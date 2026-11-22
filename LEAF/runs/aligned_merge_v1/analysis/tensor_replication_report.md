# Tensor-granularity replication

Methods: `block_ce` and `tensor_ce`.
Original experts and exact existing calibration IDs; matched CE, optimizer, 1600 updates, and validation schedule.
Block has 528 coefficients; the tensor variant has 6,128. Both export the same full LEAF architecture and fixed text readout.
Checkpoint selection uses validation macro BAcc, then CE. Official L2 uses instruction seed42 and batch256.

| Protocol | Seed | Block | Tensor | Difference (pp) | Worst task difference (pp) | Worst task | Tasks down |
|---|---:|---:|---:|---:|---:|---|---:|
| validation | 0 | 0.627208 | 0.636997 | +0.979 | -0.660 | EMO_SEED_5_seg4 | 2 |
| official_l2 | 0 | 0.572700 | 0.572631 | -0.007 | -11.390 | Workload | 5 |
| validation | 1 | 0.627773 | 0.633499 | +0.573 | -2.652 | ADHD_AliMotie | 3 |
| official_l2 | 1 | 0.567738 | 0.569081 | +0.134 | -3.610 | MI_Shin2017A | 4 |
| validation | 2 | 0.625180 | 0.633342 | +0.816 | -2.438 | MI_ShanghaiU | 5 |
| official_l2 | 2 | 0.571700 | 0.574412 | +0.271 | -2.430 | MI_OpenBMI | 5 |

| Protocol | Completed pairs | Block mean ± sample SD | Tensor mean ± sample SD | Paired difference, pp ± sample SD |
|---|---:|---:|---:|---:|
| validation | 3/3 | 0.626720 ± 0.001364 | 0.634613 ± 0.002066 | 0.789236 ± 0.204464 |
| official_l2 | 3/3 | 0.570712 ± 0.002624 | 0.572042 ± 0.002714 | 0.132917 ± 0.139068 |

The completed official paired gain is +0.133 pp, versus +0.789 pp on validation. The worst official regression is Workload in seed0: -11.390 pp.
The validation improvement is repeatable; a robust test improvement is not established.

Training seeds are repetitions; heterogeneous tasks are not independent seeds. Seed0 was exploratory; seeds1/2 use the predeclared paired configuration.
The seed0 tensor-scale control is reported separately; it has not been replicated across seeds.
These candidates do not establish a CPSC effect or a new semantic-alignment mechanism.

Pending: none.

| Seed | Tensor above Block at same update | Median same-update gain (pp) | Final gain (pp) | Tensor final minus selected (pp) |
|---|---:|---:|---:|---:|
| 0 | 10/10 | +0.914 | +0.915 | -0.064 |
| 1 | 9/10 | +0.679 | +1.289 | -0.346 |
| 2 | 9/10 | +0.850 | +1.110 | +0.000 |

These checkpoint comparisons describe within-run behavior, not independent statistical evidence.

![Validation trajectories](tensor_replication_dynamics.png)


Existing POC references and compute context:

| Seed | Base L2 | Joint L2 | Expert-bank training updates / summed seconds | Joint updates / seconds |
|---|---:|---:|---:|---:|
| 0 | 0.540656 | 0.603875 | 48000 / 5273.1 | 48000 / 5255.5 |
| 1 | 0.540656 | 0.598762 | 48000 / 5069.8 | 48000 / 4928.5 |
| 2 | 0.540656 | 0.601681 | 48000 / 5075.7 | 48000 / 4931.6 |

The expert bank is reused by both mergers. Its original training cost is included above, not charged again per merge. Summed process wall time is not elapsed calendar time or measured device utilization; concurrent jobs and hardware affect comparisons.

| Seed | Method | Coefficients | Merge seconds, including validation/export | Peak allocated CUDA GiB |
|---|---|---:|---:|---:|
| 0 | block_ce | 528 | 593.4 | 34.44 |
| 0 | tensor_ce | 6128 | 587.8 | 34.44 |
| 0 | tensor_scale_ce | 383 | 586.3 | 34.44 |
| 1 | block_ce | 528 | 615.5 | 34.44 |
| 1 | tensor_ce | 6128 | 588.2 | 34.44 |
| 2 | block_ce | 528 | 586.6 | 34.44 |
| 2 | tensor_ce | 6128 | 584.1 | 34.44 |

Each deployed merger is one ordinary full checkpoint; expert tensors and fitted coefficients are needed only during merging. The candidate uses no teacher cache or alignment/SVD stage. Detailed phase times, checkpoint sizes and test times are in `tensor_replication_compute.csv`.

Official per-task changes versus Base and Joint are in `tensor_replication_reference_tasks.csv`; paired granularity changes are in `tensor_replication_tasks.csv`. References predate this experiment.

Interrupted attempt `seed2/interrupted/block_ce_attempt1/interruption_summary.json`: last logged update 800; optimization/validation time through its last checkpoint 257.3s; elapsed time to observed termination at most 450.1s. This is extra experiment cost, excluded from completed-run performance and timing comparisons.