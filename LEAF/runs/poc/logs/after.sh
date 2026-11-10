#!/usr/bin/env bash
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
DATASETS=MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI
RUN=runs/poc
while kill -0 1343620 2>/dev/null; do sleep 20; done
for DS in ${DATASETS//,/ }; do test -f "$RUN/experts/$DS/best.ckpt" || { echo "expert $DS missing"; exit 1; }; grep -q best_round "$RUN/experts/$DS/run.json" || { echo "expert $DS unfinished"; exit 1; }; done
python merge_poc.py --base "$BASE" --experts-dir "$RUN/experts" --datasets "$DATASETS" --method uniform   --calibration-per-dataset 512 --steps 500 --lr 1e-2 --bs 128 --seed 0 --gpu 0 --outdir "$RUN/merge_uniform" > "$RUN/logs/merge_uniform.log" 2>&1
python merge_poc.py --base "$BASE" --experts-dir "$RUN/experts" --datasets "$DATASETS" --method global   --calibration-per-dataset 512 --steps 500 --lr 1e-2 --bs 128 --seed 0 --gpu 0 --outdir "$RUN/merge_global" > "$RUN/logs/merge_global.log" 2>&1 &
GLOBAL=$!
while kill -0 1343632 2>/dev/null; do sleep 20; done
python merge_poc.py --base "$BASE" --experts-dir "$RUN/experts" --datasets "$DATASETS" --method block   --calibration-per-dataset 512 --steps 500 --lr 1e-2 --bs 128 --seed 0 --gpu 1 --outdir "$RUN/merge_block" > "$RUN/logs/merge_block.log" 2>&1
wait $GLOBAL
python eval_poc.py --base "$BASE" --run-dir "$RUN" --datasets "$DATASETS" --external MI_Weibo2014   --bs 128 --gpu 0 --output "$RUN/results.csv" > "$RUN/logs/eval.log" 2>&1
echo PIPELINE_DONE
