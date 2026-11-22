> 2026-11-15: The user has set a stronger active goal: a publishable contribution, an experimentally established mechanism, and a substantial reduction of the Joint gap. The prior narrow completion below does not satisfy that goal. Current work: `../paper_merge_v2/STATUS.md`.

# Research result and current status

The goal of identifying an approach with potential is achieved at the macro
performance level. The candidate is **class-balanced tensor merging with a
fixed average of late coefficient states** (`tensor_balanced_avg`). It improves
official macro balanced accuracy in all three seeds against original Block,
selected balanced Block, averaged balanced Block, and original tensor-CE.
The gains are modest and coexist with substantial task regressions.

## Fixed method and result

Fit6,128 signed expert/tensor coefficients using the original512 calibration
examples per task,16 experts,Adam0.01,batch128 and1,600 updates. Each example
gets weight N_cal/(C*n_class). Export the arithmetic mean of coefficients at
steps800/960/1120/1280/1440/1600. The window was fixed before its outputs were
evaluated; there is no window search or additional optimizer update.

| Official L2, source16 macro BAcc | Seed0 | Seed1 | Seed2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Original Block |57.2700%|56.7738%|57.1700%|57.0713 ±0.2624%|
| Averaged balanced Block |57.1138%|56.8006%|57.6425%|57.1856 ±0.4255%|
| Averaged balanced tensor |57.4525%|57.1931%|57.8825%|57.5094 ±0.3482%|

Paired candidate gains (pp,mean ± sample SD): original Block
**+0.438±0.265**; matched averaged balanced Block **+0.324±0.077**;
selected balanced Block **+0.398±0.140**; original tensor **+0.305±0.127**.
Against the selected balanced tensor source, averaging gives+0.195±0.225 pp
and improves2/3 seeds. Its benefit is not uniform in isolation.

All three averaged tensor validation outputs pass the predeclared resource
gate: gains+1.114/+0.563/+1.075 pp over original Block, with worst task
changes-0.833/-2.652/-2.392 pp. The paired validation mean gain is
+0.917±0.307 pp. Official scores use the unchanged L2 seed42,batch256 evaluator;
the CSV summary row is excluded from the16-task macro.

Full evidence: [trajectory averaging report](analysis/trajectory_average_report.md),
[three-seed class-balancing report](analysis/class_balance_replication_report.md),
[original tensor report](analysis/tensor_replication_report.md),
and the append-only [progress log](PROGRESS.md).

## Limits and behavior

Official ADHD regressions versus original Block are5.02 pp in seed0 and
4.89 pp in seed1; seed0 Workload is down4.17 pp. Seed2's worst task decline
is OpenBMI1.29 pp. The candidate therefore has macro-level potential, not
uniform task improvement. Joint remains ahead by2.635±0.327 pp.

Class balancing alone gives tensor-CE+0.110±0.110 official pp. The fixed
average adds a small mean gain and reduces actual weight displacement from
base in both backbone and interface relative to each selected balanced
tensor source. Those geometry observations do not prove a generalization
mechanism. All source trajectories, per-task losses, coefficients, actual
parameter displacements and negative results remain available.

These are development experiments on an exposed benchmark: earlier official
results were known when later hypotheses were fixed. Source validation may
share subjects with training; official splits retain the original held-out
protocol. Predefined HDF5 splits lack original subject/session IDs, so stable
split-row IDs are retained. Three seeds are not strong significance evidence.
Tensor granularity, class-weighted CE and weight averaging are established
ideas; this is not a CPSC or algorithmic novelty claim.

## Verification and deployment

All six averaging outputs have exact mean coefficients, original calibration
IDs,matching source hashes,unchanged architecture keys/shapes/dtypes and13
exact base buffers/inactive tensors. Full validation functional/export logit
errors are at most4.94e-6, with no changed predictions and identical BAcc.
All six official receipts match checkpoints/run/evaluator hashes frozen
after every export completed. Audits:
`inputs/trajectory_average_deployment_review.json` and
`inputs/trajectory_average_official_review.json`.

Deployment is one ordinary26,424,208-parameter LEAF checkpoint, including
25,751,552 active downstream parameters, with fixed text readouts. It needs
no expert bank,router,task-specific head or fitted coefficient file at runtime.
The unchanged official evaluator successfully loaded each static export.

The original fit takes about585–588 seconds; averaging plus full paired
validation/export adds79–80 seconds,zero new optimizer updates. Total
pipeline time is about664–667 seconds,excluding separately reported
expert-bank training and official tests. Concurrent wall times do not imply
a measured hardware speedup. Cost tables retain all controls and the earlier
interrupted unweighted seed2 attempt.

## Plan scope and reproducibility

P0/P1 audits,math checks and implementation are complete. P2's required seed0
matrix and prescribed CPSC sensitivities are complete and negative; all
semantic/logit KD,repair,projection,hidden-alignment,RegMean and TIES
adaptations remain in the seed0 report. Tensor,balanced CE and fixed averaging
have complete three-seed paired experiments. The narrower requested outcome
is an evidence-backed candidate with potential; it is not full paper-scale
completion.

Full-space CPSC rotation,matched-time extension and128-sample CPSC sensitivity
remain `not_run` because the CPSC prerequisite was negative. P3 second
interface/backbone and P4 continual consolidation are `not_run` follow-up
stages. They were not used to hide negative static results.

Static artifacts: `seed{0,1,2}/tensor_balanced_avg/merged.ckpt`.
The frozen rule is in `inputs/trajectory_average_plan.json`; source snapshots,
commands,launch receipts and full source-trajectory pointers are retained.
To reproduce an average into a new destination:

```bash
python average_merge_trajectory.py --source runs/aligned_merge_v1/seed0/tensor_balanced_ce --outdir runs/aligned_merge_v1/seed0/tensor_balanced_avg_reproduction --gpu 0
python report_trajectory_average.py
```

All planned jobs and evaluations are complete. No additional sweep is queued.
