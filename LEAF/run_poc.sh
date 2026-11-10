#!/usr/bin/env bash
# LEAF model-merging POC for one seed. Run from anywhere; uses GPU 0 and 1:
# experts run sequentially on GPU 0 while joint training runs on GPU 1.
#   SEED=1 STEPS=3000 RUN=runs/poc3k/seed1 bash run_poc.sh
#   DATASETS=<16 datasets> MERGE_STEPS=1600 MERGE_EVAL=160 SEED=0 STEPS=3000 RUN=runs/poc16/seed0 bash run_poc.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE=${BASE:-checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt}
DATASETS=${DATASETS:-MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI}
SEED=${SEED:-0}
STEPS=${STEPS:-1000}             # optimizer steps per dataset (experts and joint)
MERGE_STEPS=${MERGE_STEPS:-500}  # total alpha updates; multiple of the number of datasets
MERGE_EVAL=${MERGE_EVAL:-50}     # alpha validation cadence
EXTERNAL=${EXTERNAL:-MI_Weibo2014}
DIRECT_DS=${DIRECT_DS:-}         # empty = c_direct_inference.py default (all tasks.yaml datasets)
RUN=${RUN:-runs/poc}
mkdir -p "$RUN/logs"
echo "seed=$SEED steps-per-dataset=$STEPS merge-steps=$MERGE_STEPS run=$RUN start=$(date -Is)"
echo "datasets=$DATASETS"

# Independent experts; each loads the same BASE.
(
  for DS in ${DATASETS//,/ }; do
    python d_downstream_qformer_ce.py \
      --ckpt "$BASE" --ds "$DS" --steps-per-dataset "$STEPS" \
      --bs 128 --seed "$SEED" --gpu 0 --outdir "$RUN/experts/$DS" > "$RUN/logs/expert_$DS.log" 2>&1
  done
) &
EXPERTS=$!

# Joint training: STEPS rounds x |DATASETS| optimizer steps.
python d_downstream_qformer_ce.py \
  --ckpt "$BASE" --ds "$DATASETS" --steps-per-dataset "$STEPS" \
  --bs 128 --seed "$SEED" --gpu 1 --outdir "$RUN/joint" > "$RUN/logs/joint.log" 2>&1 &
JOINT=$!

wait "$EXPERTS"
echo "experts done $(date -Is)"
# The three merges share the same experts and calibration indices; uniform does not optimize.
MERGE_ARGS=(--base "$BASE" --experts-dir "$RUN/experts" --datasets "$DATASETS"
            --calibration-per-dataset 512 --steps "$MERGE_STEPS" --eval-every "$MERGE_EVAL" --lr 1e-2
            --bs 128 --seed "$SEED")
python merge_poc.py "${MERGE_ARGS[@]}" --method uniform --gpu 0 --outdir "$RUN/merge_uniform" \
  > "$RUN/logs/merge_uniform.log" 2>&1
python merge_poc.py "${MERGE_ARGS[@]}" --method global --gpu 0 --outdir "$RUN/merge_global" \
  > "$RUN/logs/merge_global.log" 2>&1 &
GLOBAL=$!
wait "$JOINT"
echo "joint done $(date -Is)"
python merge_poc.py "${MERGE_ARGS[@]}" --method block --gpu 1 --outdir "$RUN/merge_block" \
  > "$RUN/logs/merge_block.log" 2>&1
wait "$GLOBAL"
echo "merges done $(date -Is)"

# Official test: c_direct_inference.py on every checkpoint.
RUN="$RUN" DATASETS="$DATASETS" DIRECT_DS="$DIRECT_DS" BASE="$BASE" bash direct_poc.sh
echo "direct inference done $(date -Is)"
# Supplementary: Kappa/AUROC/wF1/CE with the training readout, plus external MI.
python eval_poc.py \
  --base "$BASE" --run-dir "$RUN" --datasets "$DATASETS" --external "$EXTERNAL" \
  --bs 128 --gpu 0 --output "$RUN/results.csv" > "$RUN/logs/eval.log" 2>&1
echo "seed=$SEED done=$(date -Is)"
