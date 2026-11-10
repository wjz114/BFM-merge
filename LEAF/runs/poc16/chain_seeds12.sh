#!/usr/bin/env bash
# poc16 seeds 1 and 2 (same settings as seed 0), official test on the 16 sources only, then 3-seed aggregation.
set -euo pipefail
cd /home/jingzhou/BFM-merge/LEAF
DS16=MI_OpenBMI,MI_BCIC_IV2a,MI_ShanghaiU,MI_BCIC_Upperlimb,MI_HighGamma,MI_Cho2017,MI_Shin2017A,MI_PhysioNet,EMO_FACED,EMO_SEED_3_seg4,EMO_SEED_4_seg4,EMO_SEED_5_seg4,EMO_SEED_7_seg4,SSVEP_OpenBMI,Workload,ADHD_AliMotie
for s in 1 2; do
  echo "poc16 seed$s start $(date -Is)"
  DATASETS=$DS16 DIRECT_DS=$DS16 SEED=$s STEPS=3000 MERGE_STEPS=1600 MERGE_EVAL=160 RUN=runs/poc16/seed$s \
    bash run_poc.sh > runs/poc16/seed${s}_driver.log 2>&1
  echo "poc16 seed$s done $(date -Is)"
done
# Seed 0 was tested on all 17 tasks.yaml datasets; drop its non-source (CS_BCIC_Speech) rows so the
# three per-seed tables have identical rows. Instruction choice does not depend on --ds, so the
# 16 source results are the same either way.
python - <<'PY'
import csv
rows = list(csv.DictReader(open('runs/poc16/seed0/direct_results.csv')))
keep = [r for r in rows if r['role'] != 'not a source']
assert len(rows) - len(keep) == 21 * 3, (len(rows), len(keep))  # 21 single models (Specialist has no such row) x 3 levels
with open('runs/poc16/seed0/direct_results_16sources.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(keep)
PY
python eval_poc.py --aggregate runs/poc16/seed0/direct_results_16sources.csv runs/poc16/seed{1,2}/direct_results.csv \
  --output runs/poc16/direct_results_seeds.csv > runs/poc16/aggregate_direct.log 2>&1
python eval_poc.py --aggregate runs/poc16/seed{0,1,2}/results.csv --output runs/poc16/results_seeds.csv \
  > runs/poc16/aggregate_results.log 2>&1
echo "poc16 3-seed aggregation done $(date -Is)"
