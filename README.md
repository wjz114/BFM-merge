# BFM-merge

Experiments on fine-grained model merging for EEG foundation models, based on
[LEAF](https://github.com/Jiang-Muyun/LEAF). The experiments compare specialized
models, joint fine-tuning, uniform merging, global and block-wise task-vector
merging, and softmax-constrained weighted averaging.

## Code and experiment plans

- [LEAF implementation and setup](LEAF/README.md)
- [Model-merging experiment plan](LEAF_Model_Merging_POC_Plan.md)
- [Softmax-merging experiment plan](Model_softmax_merging_POC.md)
- Training and evaluation: [run_poc.sh](LEAF/run_poc.sh),
  [merge_poc.py](LEAF/merge_poc.py), and [eval_poc.py](LEAF/eval_poc.py)

## Recorded results

| Experiment | Artifacts | Report |
| --- | --- | --- |
| Five source datasets, seed 0 | [poc](LEAF/runs/poc/) | [Initial report](LEAF_Model_Merging_POC_Report.md), [softmax report](LEAF/runs/poc/Softmax_Merging_Experiment_Report.md) |
| Five source datasets, 3,000 steps per dataset, seeds 0–2 | [poc3k](LEAF/runs/poc3k/) | Aggregate `results_seeds.*` and `direct_results_seeds.*` in the artifact directory |
| Sixteen source datasets, seeds 0–2 | [poc16](LEAF/runs/poc16/) | [Three-seed report, including softmax merging](LEAF/runs/poc16/POC16_Three_Seed_Report.md) |

The repository includes result tables, run configurations, training histories,
metric arrays, figures, logs, calibration indices, and learned merging
coefficients. The small text prototypes and saved text readouts are retained
for reproducibility. Model checkpoints, raw EEG data, data caches, and Python
caches are excluded by `.gitignore`.

## Reproducing experiments

Follow [LEAF's setup instructions](LEAF/README.md), configure your dataset paths
in [configs/env.yaml](LEAF/configs/env.yaml), and download the released checkpoint
with `bash download_checkpoints.sh` from the `LEAF` directory. Existing reports
and run metadata retain the original machine paths and experiment dates.

For example, to run the five-dataset experiment in a new output directory:

```bash
cd LEAF
SEED=0 STEPS=3000 RUN=runs/reproduction/seed0 bash run_poc.sh
```

The script uses two GPUs by default. Training and inference require local
datasets and checkpoints; the committed tables, figures, and reports can be
inspected without them. Experiment-specific commands and settings are recorded
in the reports and run directories.

## Upstream source

`LEAF/` contains the upstream source at commit `1e03d69` with the local
model-merging changes and recorded experiments. It is included as ordinary
files so cloning this repository includes the implementation and results.
The upstream [CBCR License 1.0](LEAF/LICENSE) is preserved.
