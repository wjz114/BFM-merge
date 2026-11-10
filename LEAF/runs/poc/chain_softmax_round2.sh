#!/usr/bin/env bash
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
DS5=MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI
while kill -0 1502671 2>/dev/null; do sleep 30; done   # current round (poc16 seed 0 chain)
echo "current round finished $(date -Is); installing staged merge_poc.py / eval_poc.py"
mv merge_poc.py.next merge_poc.py
mv eval_poc.py.next eval_poc.py
for i in 0 1; do
  M=$([ $i = 0 ] && echo softmax_global || echo softmax_block)
  python merge_poc.py --base "$BASE" --experts-dir runs/poc/experts --datasets "$DS5" --method "$M" \
    --calibration-per-dataset 512 --steps 500 --eval-every 50 --lr 1e-2 --bs 128 --seed 0 --gpu $i \
    --outdir runs/poc/merge_$M > runs/poc/logs/merge_$M.log 2>&1 &
  eval "P$i=\$!"
done
wait "$P0"; wait "$P1"
for M in softmax_global softmax_block; do
  cmp runs/poc/merge_$M/calibration_indices.npz runs/poc/merge_block/calibration_indices.npz
done
echo "calibration indices identical to the previous POC"
python eval_poc.py --base "$BASE" --run-dir runs/poc --datasets "$DS5" \
  --models base,joint,uniform,global,block,softmax_global,softmax_block --external MI_Weibo2014 \
  --bs 128 --gpu 0 --output runs/poc/results_round2.csv > runs/poc/logs/eval_round2.log 2>&1 &
PE=$!
for M in softmax_global softmax_block; do
  python c_direct_inference.py --ckpt runs/poc/merge_$M/merged.ckpt --ds "$DS5" --gpu 1 --bs 256 --level 0,1,2 \
    --output runs/poc/direct_inference/$M.csv > runs/poc/direct_inference/$M.log 2>&1
done
wait "$PE"
echo "round 2 done $(date -Is)"
