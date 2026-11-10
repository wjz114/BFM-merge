#!/usr/bin/env bash
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
for DS in MI_OpenBMI MI_Cho2017 MI_PhysioNet EMO_SEED_3_seg4 SSVEP_OpenBMI; do
  python d_downstream_qformer_ce.py --ckpt "$BASE" --ds "$DS" --steps-per-dataset 1000 \
    --bs 128 --seed 0 --gpu 0 --outdir "runs/poc/experts/$DS" > "runs/poc/logs/expert_$DS.log" 2>&1
done
