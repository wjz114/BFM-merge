#!/usr/bin/env bash
# poc16 softmax round: softmax_global / softmax_block on each seed's existing 16 experts (no retraining).
# Calibration subsets, budget (1,600 steps, validation every 160), lr, batch size and selection rule are
# the same as the poc16 Block merge of that seed. Official test = c_direct_inference.py on the 16 sources;
# supplementary = eval_poc.py (Base and Joint re-evaluated as a consistency check, plus external Weibo).
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
DS16=MI_OpenBMI,MI_BCIC_IV2a,MI_ShanghaiU,MI_BCIC_Upperlimb,MI_HighGamma,MI_Cho2017,MI_Shin2017A,MI_PhysioNet,EMO_FACED,EMO_SEED_3_seg4,EMO_SEED_4_seg4,EMO_SEED_5_seg4,EMO_SEED_7_seg4,SSVEP_OpenBMI,Workload,ADHD_AliMotie
METHODS=(softmax_global softmax_block)
for s in 0 1 2; do
  RUN=runs/poc16/seed$s
  echo "seed$s softmax merges start $(date -Is)"
  PIDS=()
  for i in 0 1; do
    M=${METHODS[$i]}
    python merge_poc.py --base "$BASE" --experts-dir "$RUN/experts" --datasets "$DS16" --method "$M" \
      --calibration-per-dataset 512 --steps 1600 --eval-every 160 --lr 1e-2 --bs 128 --seed "$s" --gpu "$i" \
      --outdir "$RUN/merge_$M" > "$RUN/logs/merge_$M.log" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  for M in "${METHODS[@]}"; do
    cmp "$RUN/merge_$M/calibration_indices.npz" "$RUN/merge_block/calibration_indices.npz"
  done
  echo "seed$s merges done, calibration indices identical to merge_block $(date -Is)"
  PIDS=()
  for i in 0 1; do
    M=${METHODS[$i]}
    python c_direct_inference.py --ckpt "$RUN/merge_$M/merged.ckpt" --ds "$DS16" --gpu "$i" --bs 256 \
      --level 0,1,2 --output "$RUN/direct_inference/$M.csv" > "$RUN/direct_inference/$M.log" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  python eval_poc.py --collect-direct "$RUN/direct_inference" --datasets "$DS16" \
    --models softmax_global,softmax_block --output "$RUN/direct_results_softmax.csv" \
    > "$RUN/logs/direct_collect_softmax.log" 2>&1
  echo "seed$s direct inference done $(date -Is)"
  python eval_poc.py --base "$BASE" --run-dir "$RUN" --datasets "$DS16" \
    --models base,joint,softmax_global,softmax_block --external MI_Weibo2014 \
    --bs 128 --gpu 0 --output "$RUN/results_softmax.csv" > "$RUN/logs/eval_softmax.log" 2>&1
  echo "seed$s done $(date -Is)"
done
python eval_poc.py --aggregate runs/poc16/seed{0,1,2}/direct_results_softmax.csv \
  --output runs/poc16/direct_results_seeds_softmax.csv > runs/poc16/aggregate_direct_softmax.log 2>&1
python eval_poc.py --aggregate runs/poc16/seed{0,1,2}/results_softmax.csv \
  --output runs/poc16/results_seeds_softmax.csv > runs/poc16/aggregate_results_softmax.log 2>&1
echo "poc16 softmax 3-seed aggregation done $(date -Is)"
