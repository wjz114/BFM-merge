# Class-balanced CE replication

Methods: `block_balanced_ce` and `tensor_balanced_ce`.
Original experts and exact existing calibration IDs; matched CE, optimizer, 1600 updates, and validation schedule.
Block has 528 coefficients; the tensor variant has 6,128. Both export the same full LEAF architecture and fixed text readout.
Checkpoint selection uses validation macro BAcc, then CE. Official L2 uses instruction seed42 and batch256.

| Protocol | Seed | Block | Tensor | Difference (pp) | Worst task difference (pp) | Worst task | Tasks down |
|---|---:|---:|---:|---:|---:|---|---:|
| validation | 0 | 0.627437 | 0.638608 | +1.117 | -1.667 | ADHD_AliMotie | 6 |
| official_l2 | 0 | 0.570163 | 0.575006 | +0.484 | -3.610 | Workload | 7 |
| validation | 1 | 0.627368 | 0.632197 | +0.483 | -1.980 | MI_Cho2017 | 7 |
| official_l2 | 1 | 0.566787 | 0.569569 | +0.278 | -3.840 | ADHD_AliMotie | 7 |
| validation | 2 | 0.626902 | 0.633938 | +0.704 | -1.818 | ADHD_AliMotie | 5 |
| official_l2 | 2 | 0.576406 | 0.574856 | -0.155 | -6.110 | Workload | 7 |

| Protocol | Completed pairs | Block mean ± sample SD | Tensor mean ± sample SD | Paired difference, pp ± sample SD |
|---|---:|---:|---:|---:|
| validation | 3/3 | 0.627236 ± 0.000291 | 0.634914 ± 0.003315 | 0.767872 ± 0.321919 |
| official_l2 | 3/3 | 0.571119 ± 0.004880 | 0.573144 ± 0.003097 | 0.202500 ± 0.326327 |

The completed official paired gain is +0.202 pp, versus +0.768 pp on validation. The worst official regression is Workload in seed2: -6.110 pp.

Training seeds are repetitions; heterogeneous tasks are not independent seeds. Seed0 was exploratory; seeds1/2 use the predeclared paired configuration.
Class weights are N_cal/(C*n_class), fixed from the original calibration labels; validation CE is unweighted. The earlier unweighted official results were known during development.
These candidates do not establish a CPSC effect or a new semantic-alignment mechanism.

Pending: none.

| Seed | Tensor above Block at same update | Median same-update gain (pp) | Final gain (pp) | Tensor final minus selected (pp) |
|---|---:|---:|---:|---:|
| 0 | 10/10 | +1.152 | +0.966 | -0.483 |
| 1 | 9/10 | +0.490 | +0.877 | +0.000 |
| 2 | 10/10 | +0.779 | +0.332 | -0.372 |

These checkpoint comparisons describe within-run behavior, not independent statistical evidence.

![Validation trajectories](class_balance_replication_dynamics.png)


Existing POC references and compute context:

| Seed | Base L2 | Joint L2 | Expert-bank training updates / summed seconds | Joint updates / seconds |
|---|---:|---:|---:|---:|
| 0 | 0.540656 | 0.603875 | 48000 / 5273.1 | 48000 / 5255.5 |
| 1 | 0.540656 | 0.598762 | 48000 / 5069.8 | 48000 / 4928.5 |
| 2 | 0.540656 | 0.601681 | 48000 / 5075.7 | 48000 / 4931.6 |

The expert bank is reused by both mergers. Its original training cost is included above, not charged again per merge. Summed process wall time is not elapsed calendar time or measured device utilization; concurrent jobs and hardware affect comparisons.

| Seed | Method | Coefficients | Merge seconds, including validation/export | Peak allocated CUDA GiB |
|---|---|---:|---:|---:|
| 0 | block_balanced_ce | 528 | 586.6 | 34.44 |
| 0 | tensor_balanced_ce | 6128 | 584.8 | 34.44 |
| 0 | block_ce | 528 | 593.4 | 34.44 |
| 0 | tensor_ce | 6128 | 587.8 | 34.44 |
| 1 | block_balanced_ce | 528 | 587.4 | 34.44 |
| 1 | tensor_balanced_ce | 6128 | 587.6 | 34.44 |
| 1 | block_ce | 528 | 615.5 | 34.44 |
| 1 | tensor_ce | 6128 | 588.2 | 34.44 |
| 2 | block_balanced_ce | 528 | 587.1 | 34.44 |
| 2 | tensor_balanced_ce | 6128 | 586.4 | 34.44 |
| 2 | block_ce | 528 | 586.6 | 34.44 |
| 2 | tensor_ce | 6128 | 584.1 | 34.44 |

Each deployed merger is one ordinary full checkpoint; expert tensors and fitted coefficients are needed only during merging. The candidate uses no teacher cache or alignment/SVD stage. Detailed phase times, checkpoint sizes and test times are in `class_balance_replication_compute.csv`.

Official per-task changes versus Base and Joint are in `class_balance_replication_reference_tasks.csv`; paired granularity changes are in `class_balance_replication_tasks.csv`. References predate this experiment.

Interrupted attempt `seed2/interrupted/block_ce_attempt1/interruption_summary.json`: last logged update 800; optimization/validation time through its last checkpoint 257.3s; elapsed time to observed termination at most 450.1s. This is extra experiment cost, excluded from completed-run performance and timing comparisons.

Class-balanced CE versus the same coefficient layout with ordinary CE:

| Protocol | Seed | Candidate | Control | Difference (pp) | Worst task | Worst difference (pp) |
|---|---:|---|---|---:|---|---:|
| validation | 0 | block_balanced_ce | block_ce | +0.023 | SSVEP_OpenBMI | -1.754 |
| official_l2 | 0 | block_balanced_ce | block_ce | -0.254 | SSVEP_OpenBMI | -2.160 |
| validation | 0 | tensor_balanced_ce | tensor_ce | +0.161 | MI_BCIC_IV2a | -1.455 |
| official_l2 | 0 | tensor_balanced_ce | tensor_ce | +0.237 | MI_ShanghaiU | -1.710 |
| validation | 1 | block_balanced_ce | block_ce | -0.041 | MI_Shin2017A | -3.152 |
| official_l2 | 1 | block_balanced_ce | block_ce | -0.095 | MI_Shin2017A | -2.500 |
| validation | 1 | tensor_balanced_ce | tensor_ce | -0.130 | MI_PhysioNet | -2.090 |
| official_l2 | 1 | tensor_balanced_ce | tensor_ce | +0.049 | Workload | -2.500 |
| validation | 2 | block_balanced_ce | block_ce | +0.172 | MI_HighGamma | -1.788 |
| official_l2 | 2 | block_balanced_ce | block_ce | +0.471 | MI_BCIC_IV2a | -0.780 |
| validation | 2 | tensor_balanced_ce | tensor_ce | +0.060 | ADHD_AliMotie | -1.818 |
| official_l2 | 2 | tensor_balanced_ce | tensor_ce | +0.044 | MI_HighGamma | -1.660 |

| Protocol | Candidate | Completed pairs | Mean difference (pp) | Sample SD (pp) |
|---|---|---:|---:|---:|
| validation | block_balanced_ce | 3/3 | +0.052 | 0.109 |
| official_l2 | block_balanced_ce | 3/3 | +0.041 | 0.381 |
| validation | tensor_balanced_ce | 3/3 | +0.030 | 0.148 |
| official_l2 | tensor_balanced_ce | 3/3 | +0.110 | 0.110 |

Only completed pairs are included. Exact calibration IDs, task-update order, experts and training settings are matched. Per-task weighting effects are saved in `class_balance_effect_tasks.csv`.
