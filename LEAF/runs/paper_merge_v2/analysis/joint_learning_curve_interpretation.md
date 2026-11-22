# Historical Joint curves and the remaining baseline question

All three historical Joint histories were reconciled against their NPZ
records and every task's metrics. No new forward pass or data read was
needed. The curves use the historical BF16 protocol; intermediate
checkpoints were not retained and cannot now be replayed in FP32.

| Optimizer updates | Three-seed mean BAcc (%) | Sample SD (pp) |
| --- | ---: | ---: |
| 1600 | 60.364609 | 1.106277 |
| 3200 | 61.475453 | 0.456416 |
| 4800 | 61.596556 | 1.008776 |
| 9600 | 63.560327 | 0.253709 |
| 19200 | 65.486211 | 0.259649 |
| 24000 | 66.069521 | 0.526185 |
| 48000 | 67.683407 | 0.320412 |

The fixed24000-to48000 interval adds1.613886±0.266660pp across seeds. SSVEP
adds6.8497pp; Workload3.6894pp; OpenBMI2.7257pp. Eight tasks improve in all
three runs, while Shin2017A declines in all three. The increase is not
uniform across tasks, and late training still contributes to the endpoint.

These observations do not isolate training length as a cause. Time is
coupled to warmup/cosine rates, accumulated optimizer states and changing
task representations. They also do not establish how long the different
balanced-CE repair recipe needs to train.

The first1600updates lie inside the historical2400-update warmup. Therefore
the early historical curve is not an optimized short-budget Joint control.
The existing flat-rate direct-base controls are distinct, useful comparators;
neither is automatically the best possible short-budget recipe. The staged
3200-update control spends half its updates on original512 rows per task,
so it is not a3200-update full-pool trajectory either.

Selected Joint checkpoints occur at48000,43200,44800updates for seeds0,1,2.
They are distinct from the fixed48000 endpoints above. Their FP32-minus-
historical selected BAcc changes are+0.009513,-0.000066,+0.000429pp. The
existing FP32 selected-checkpoint reference remains authoritative for current
repair-gap tables; no endpoint/selection or precision substitution is made.

Historical curves first exceed their bank's full-pool-repair reference at
19200,14400,11200logged updates, respectively. This is a retrospective
cross-protocol description, not a compute-speedup result: expert training,
coefficient fitting and precision remain part of the cost accounting. The
complete expert-bank+coefficient+repair pipeline records51200updates versus
48000for Joint, with different per-update computation and data exposure.

Sources:`historical_joint_learning_curve_audit.json`,
`merge_pipeline_budget.json`, `bank_coverage_replication_summary.json`.
The standalone figure `figures/historical_joint_learning_curves.{png,pdf}`
was inspected. No new training schedule or candidate was selected from
these curves.
