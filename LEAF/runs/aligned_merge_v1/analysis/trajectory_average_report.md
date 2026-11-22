# Fixed trajectory averaging

One arithmetic mean of coefficient states at800/960/1120/1280/1440/1600, evaluated as one ordinary model. Zero new optimizer updates; no selection among averaging windows. All earlier official tests were known during development.

| Protocol | Method | Seeds | Macro BAcc mean | Sample SD |
|---|---|---:|---:|---:|
| validation | block_balanced_avg | 3/3 | 0.626648 | 0.003537 |
| validation | tensor_balanced_avg | 3/3 | 0.635893 | 0.002468 |
| validation | block_balanced_ce | 3/3 | 0.627236 | 0.000291 |
| validation | tensor_balanced_ce | 3/3 | 0.634914 | 0.003315 |
| validation | block_ce | 3/3 | 0.626720 | 0.001364 |
| validation | tensor_ce | 3/3 | 0.634613 | 0.002066 |
| official_l2 | block_balanced_avg | 3/3 | 0.571856 | 0.004255 |
| official_l2 | tensor_balanced_avg | 3/3 | 0.575094 | 0.003482 |
| official_l2 | block_balanced_ce | 3/3 | 0.571119 | 0.004880 |
| official_l2 | tensor_balanced_ce | 3/3 | 0.573144 | 0.003097 |
| official_l2 | block_ce | 3/3 | 0.570712 | 0.002624 |
| official_l2 | tensor_ce | 3/3 | 0.572042 | 0.002714 |
| official_l2 | base | 3/3 | 0.540656 | 0.000000 |
| official_l2 | joint | 3/3 | 0.601440 | 0.002565 |

| Protocol | Seed | Candidate | Control | Difference (pp) | Worst task | Worst difference (pp) | Tasks down |
|---|---:|---|---|---:|---|---:|---:|
| validation | 0 | tensor_balanced_avg | block_balanced_avg | +1.476 | EMO_SEED_5_seg4 | -0.336 | 2 |
| validation | 0 | tensor_balanced_avg | tensor_balanced_ce | -0.026 | ADHD_AliMotie | -1.970 | 6 |
| validation | 0 | tensor_balanced_avg | block_balanced_ce | +1.091 | ADHD_AliMotie | -3.636 | 3 |
| validation | 0 | block_balanced_avg | block_balanced_ce | -0.385 | ADHD_AliMotie | -5.530 | 9 |
| validation | 0 | tensor_balanced_avg | block_ce | +1.114 | ADHD_AliMotie | -0.833 | 3 |
| validation | 0 | tensor_balanced_avg | tensor_ce | +0.135 | ADHD_AliMotie | -1.061 | 8 |
| validation | 0 | block_balanced_avg | block_ce | -0.362 | ADHD_AliMotie | -2.727 | 10 |
| official_l2 | 0 | tensor_balanced_avg | block_balanced_avg | +0.339 | ADHD_AliMotie | -4.460 | 5 |
| official_l2 | 0 | tensor_balanced_avg | tensor_balanced_ce | -0.048 | EMO_FACED | -1.420 | 6 |
| official_l2 | 0 | tensor_balanced_avg | block_balanced_ce | +0.436 | Workload | -4.170 | 5 |
| official_l2 | 0 | block_balanced_avg | block_balanced_ce | +0.097 | Workload | -3.060 | 7 |
| official_l2 | 0 | tensor_balanced_avg | block_ce | +0.182 | ADHD_AliMotie | -5.020 | 5 |
| official_l2 | 0 | tensor_balanced_avg | tensor_ce | +0.189 | ADHD_AliMotie | -2.380 | 9 |
| official_l2 | 0 | block_balanced_avg | block_ce | -0.156 | Workload | -3.060 | 8 |
| official_l2 | 0 | tensor_balanced_avg | base | +3.387 | Workload | -6.940 | 3 |
| official_l2 | 0 | tensor_balanced_avg | joint | -2.935 | SSVEP_OpenBMI | -15.100 | 13 |
| official_l2 | 0 | block_balanced_avg | base | +3.048 | EMO_FACED | -8.600 | 2 |
| official_l2 | 0 | block_balanced_avg | joint | -3.274 | SSVEP_OpenBMI | -16.040 | 14 |
| validation | 1 | tensor_balanced_avg | block_balanced_avg | +0.289 | MI_Shin2017A | -1.783 | 9 |
| validation | 1 | tensor_balanced_avg | tensor_balanced_ce | +0.121 | MI_HighGamma | -0.723 | 6 |
| validation | 1 | tensor_balanced_avg | block_balanced_ce | +0.604 | ADHD_AliMotie | -1.742 | 5 |
| validation | 1 | block_balanced_avg | block_balanced_ce | +0.315 | MI_BCIC_Upperlimb | -1.570 | 6 |
| validation | 1 | tensor_balanced_avg | block_ce | +0.563 | ADHD_AliMotie | -2.652 | 6 |
| validation | 1 | tensor_balanced_avg | tensor_ce | -0.009 | MI_Shin2017A | -1.611 | 9 |
| validation | 1 | block_balanced_avg | block_ce | +0.275 | ADHD_AliMotie | -1.818 | 5 |
| official_l2 | 1 | tensor_balanced_avg | block_balanced_avg | +0.392 | ADHD_AliMotie | -5.310 | 5 |
| official_l2 | 1 | tensor_balanced_avg | tensor_balanced_ce | +0.236 | ADHD_AliMotie | -1.550 | 8 |
| official_l2 | 1 | tensor_balanced_avg | block_balanced_ce | +0.514 | ADHD_AliMotie | -5.390 | 4 |
| official_l2 | 1 | block_balanced_avg | block_balanced_ce | +0.122 | MI_HighGamma | -1.280 | 9 |
| official_l2 | 1 | tensor_balanced_avg | block_ce | +0.419 | ADHD_AliMotie | -4.890 | 3 |
| official_l2 | 1 | tensor_balanced_avg | tensor_ce | +0.285 | MI_ShanghaiU | -1.890 | 5 |
| official_l2 | 1 | block_balanced_avg | block_ce | +0.027 | MI_Shin2017A | -3.330 | 7 |
| official_l2 | 1 | tensor_balanced_avg | base | +3.128 | ADHD_AliMotie | -6.360 | 3 |
| official_l2 | 1 | tensor_balanced_avg | joint | -2.683 | SSVEP_OpenBMI | -15.250 | 13 |
| official_l2 | 1 | block_balanced_avg | base | +2.735 | EMO_FACED | -7.740 | 3 |
| official_l2 | 1 | block_balanced_avg | joint | -3.076 | SSVEP_OpenBMI | -15.940 | 14 |
| validation | 2 | tensor_balanced_avg | block_balanced_avg | +1.009 | MI_ShanghaiU | -2.030 | 3 |
| validation | 2 | tensor_balanced_avg | tensor_balanced_ce | +0.199 | MI_Cho2017 | -0.928 | 4 |
| validation | 2 | tensor_balanced_avg | block_balanced_ce | +0.903 | ADHD_AliMotie | -1.818 | 4 |
| validation | 2 | block_balanced_avg | block_balanced_ce | -0.107 | ADHD_AliMotie | -4.545 | 7 |
| validation | 2 | tensor_balanced_avg | block_ce | +1.075 | MI_ShanghaiU | -2.392 | 3 |
| validation | 2 | tensor_balanced_avg | tensor_ce | +0.259 | ADHD_AliMotie | -1.818 | 4 |
| validation | 2 | block_balanced_avg | block_ce | +0.066 | ADHD_AliMotie | -0.909 | 11 |
| official_l2 | 2 | tensor_balanced_avg | block_balanced_avg | +0.240 | Workload | -3.340 | 6 |
| official_l2 | 2 | tensor_balanced_avg | tensor_balanced_ce | +0.397 | EMO_SEED_4_seg4 | -0.330 | 3 |
| official_l2 | 2 | tensor_balanced_avg | block_balanced_ce | +0.242 | Workload | -4.170 | 4 |
| official_l2 | 2 | block_balanced_avg | block_balanced_ce | +0.002 | EMO_FACED | -1.170 | 6 |
| official_l2 | 2 | tensor_balanced_avg | block_ce | +0.713 | MI_OpenBMI | -1.290 | 3 |
| official_l2 | 2 | tensor_balanced_avg | tensor_ce | +0.441 | MI_HighGamma | -1.960 | 6 |
| official_l2 | 2 | block_balanced_avg | block_ce | +0.473 | SSVEP_OpenBMI | -0.660 | 7 |
| official_l2 | 2 | tensor_balanced_avg | base | +3.817 | EMO_FACED | -6.270 | 3 |
| official_l2 | 2 | tensor_balanced_avg | joint | -2.286 | SSVEP_OpenBMI | -14.770 | 12 |
| official_l2 | 2 | block_balanced_avg | base | +3.577 | EMO_FACED | -7.680 | 2 |
| official_l2 | 2 | block_balanced_avg | joint | -2.526 | SSVEP_OpenBMI | -16.520 | 11 |

| Protocol | Candidate | Control | Pairs | Mean difference (pp) | Sample SD (pp) |
|---|---|---|---:|---:|---:|
| validation | tensor_balanced_avg | block_balanced_avg | 3/3 | +0.925 | 0.598 |
| validation | tensor_balanced_avg | tensor_balanced_ce | 3/3 | +0.098 | 0.114 |
| validation | tensor_balanced_avg | block_balanced_ce | 3/3 | +0.866 | 0.245 |
| validation | block_balanced_avg | block_balanced_ce | 3/3 | -0.059 | 0.353 |
| validation | tensor_balanced_avg | block_ce | 3/3 | +0.917 | 0.307 |
| validation | tensor_balanced_avg | tensor_ce | 3/3 | +0.128 | 0.134 |
| validation | block_balanced_avg | block_ce | 3/3 | -0.007 | 0.325 |
| official_l2 | tensor_balanced_avg | block_balanced_avg | 3/3 | +0.324 | 0.077 |
| official_l2 | tensor_balanced_avg | tensor_balanced_ce | 3/3 | +0.195 | 0.225 |
| official_l2 | tensor_balanced_avg | block_balanced_ce | 3/3 | +0.398 | 0.140 |
| official_l2 | block_balanced_avg | block_balanced_ce | 3/3 | +0.074 | 0.063 |
| official_l2 | tensor_balanced_avg | block_ce | 3/3 | +0.438 | 0.265 |
| official_l2 | tensor_balanced_avg | tensor_ce | 3/3 | +0.305 | 0.127 |
| official_l2 | block_balanced_avg | block_ce | 3/3 | +0.114 | 0.323 |
| official_l2 | tensor_balanced_avg | base | 3/3 | +3.444 | 0.348 |
| official_l2 | tensor_balanced_avg | joint | 3/3 | -2.635 | 0.327 |
| official_l2 | block_balanced_avg | base | 3/3 | +3.120 | 0.426 |
| official_l2 | block_balanced_avg | joint | 3/3 | -2.958 | 0.388 |

Pending: none.

Seed pairs measure replication; task differences and checkpoints are not independent repetitions. The dashed lines below are measured performance of the averaged weights, not the arithmetic mean of checkpoint scores.

![Source trajectories and averaged-model validation](trajectory_average_dynamics.png)

Source fitting and additional export/validation costs are separated in `trajectory_average_costs.csv`. The source fits are reused, not rerun. Base/Joint references and expert-bank costs remain in [the class-balanced report](class_balance_replication_report.md).

The combined candidate improves official macro BAcc over original Block in 3/3 seeds and over the matched averaged Block control in 3/3 seeds. This supports modest potential on the macro objective; it does not establish a uniform task benefit.

The worst original-Block task regression is ADHD_AliMotie in seed0: -5.020 pp. Averaging alone beats the selected balanced tensor source in 2/3 seeds. The per-task tables and Joint comparisons bound the claim.

The arithmetic averaging rule is related to [Izmailov et al., SWA](https://arxiv.org/abs/1803.05407v3); here the saved states come from Adam fitting of merge coefficients. This is not a reproduction of the SGD experiments or a novelty claim. Earlier test results informed development, so these are not untouched test confirmations. Three training seeds do not establish strong statistical significance.

Actual backbone/interface displacements are in `trajectory_average_parameter_geometry.json`. Export/state and frozen official-receipt audits are in `../inputs/trajectory_average_deployment_review.json` and `../inputs/trajectory_average_official_review.json`.
