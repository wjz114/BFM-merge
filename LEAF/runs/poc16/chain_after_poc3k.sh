#!/usr/bin/env bash
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
DS5=MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI
DS16=MI_OpenBMI,MI_BCIC_IV2a,MI_ShanghaiU,MI_BCIC_Upperlimb,MI_HighGamma,MI_Cho2017,MI_Shin2017A,MI_PhysioNet,EMO_FACED,EMO_SEED_3_seg4,EMO_SEED_4_seg4,EMO_SEED_5_seg4,EMO_SEED_7_seg4,SSVEP_OpenBMI,Workload,ADHD_AliMotie
while kill -0 1449941 2>/dev/null; do sleep 30; done
for s in 0 1 2; do test -f runs/poc3k/seed$s/results.csv || { echo "poc3k seed$s incomplete"; exit 1; }; done
mv run_poc.sh.next run_poc.sh
echo "poc3k finished; starting poc16 seed0 $(date -Is)"
DATASETS=$DS16 SEED=0 STEPS=3000 MERGE_STEPS=1600 MERGE_EVAL=160 RUN=runs/poc16/seed0 \
  bash run_poc.sh > runs/poc16/seed0_driver.log 2>&1 &
P16=$!
# Official test for the three 5-dataset seeds (low priority, alongside poc16 training).
for s in 0 1 2; do
  RUN=runs/poc3k/seed$s DATASETS=$DS5 DIRECT_DS=$DS5 nice -n 10 bash direct_poc.sh
done
python eval_poc.py --aggregate runs/poc3k/seed{0,1,2}/results.csv --output runs/poc3k/results_seeds.csv \
  > runs/poc3k/aggregate_results.log 2>&1
python eval_poc.py --aggregate runs/poc3k/seed{0,1,2}/direct_results.csv --output runs/poc3k/direct_results_seeds.csv \
  > runs/poc3k/aggregate_direct.log 2>&1
echo "poc3k official test + aggregation done $(date -Is)"
wait "$P16"
echo "poc16 seed0 done $(date -Is)"
