# Task: LEAF POC — Direct Weighted Model Averaging with Softmax

## 1. Background

We are conducting a research POC for the paper:

**Specialize, Then Consolidate: Fine-Grained Model Merging for EEG Foundation Models**

Project location:

`/home/jingzhou/BFM-merge/LEAF`

Current branch: `poc-model-merging`

The previous experiment is completed. Read the following files first:

- `/home/jingzhou/BFM-merge/LEAF_Model_Merging_POC_Plan.md`
- `/home/jingzhou/BFM-merge/LEAF_Model_Merging_POC_Report.md`
- `merge_poc.py`
- `eval_poc.py`
- `d_downstream_qformer_ce.py`

The previous experiment trained five specialised LEAF models from the same pretrained checkpoint and compared them against joint fine-tuning.

Previous source macro test BAcc:

| Method | BAcc |
|---|---:|
| Joint Fine-Tuning | 0.7603 |
| Uniform Merging | 0.6895 |
| Global Task Vector Merging | 0.6973 |
| Block-wise Task Vector Merging | 0.7015 |

Previous learnable merging used:

\[
\theta_M^{(g)}=\theta_0^{(g)}+\sum_{k=1}^{K}\alpha_{k,g}(\theta_k^{(g)}-\theta_0^{(g)})
\]

with unconstrained real-valued coefficients.

This experiment investigates whether **direct weighted averaging of expert parameters with softmax-normalised coefficients** performs better.

Important mathematical clarification: when the coefficients sum to 1, direct weighted parameter averaging is mathematically equivalent to task-vector merging with the same normalised coefficients. Therefore, this experiment tests the effect of **convex-combination constraints**, not an inherently different parameter representation.

## 2. New merging methods

Implement two additional methods.

### Method A: Global Softmax Weighted Averaging

\[
\alpha_k=\operatorname{softmax}(\beta)_k
\]

\[
\boxed{\theta_M=\sum_{k=1}^{K}\alpha_k\theta_k}
\]

- One learnable coefficient per expert.
- Five experts, five trainable logits \(\beta_k\).
- Initialise all logits to zero, resulting in uniform weights of 0.2.
- Softmax ensures nonnegative weights that sum to 1.
- Do not apply an additional division by the number of experts.

### Method B: Block-wise Softmax Weighted Averaging

\[
\alpha_{k,g}
=
\frac{\exp(\beta_{k,g})}
{\sum_{j=1}^{K}\exp(\beta_{j,g})}
\]

\[
\boxed{
\theta_M^{(g)}
=
\sum_{k=1}^{K}\alpha_{k,g}\theta_k^{(g)}
}
\]

- Reuse the existing 33 parameter groups from `merge_poc.py`.
- Five expert weights per group, resulting in 165 trainable logits.
- Apply softmax independently across experts for each group.
- Initialise all logits to zero.
- Do not change the existing parameter grouping.
- Keep the existing treatment of buffers and unused parameters.

Use method names:

- `softmax_global`
- `softmax_block`

Preserve the original `uniform`, `global`, and `block` methods and their behaviour.

## 3. Training and optimisation settings

Reuse the existing expert checkpoints:

`runs/poc/experts/<dataset>/best.ckpt`

Reuse the original common initialisation:

`checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt`

Five datasets:

```text
MI_OpenBMI
MI_Cho2017
MI_PhysioNet
EMO_SEED_3_seg4
SSVEP_OpenBMI
```

Use exactly the same supervised objective as the previous POC:

\[
\mathcal L
=
\frac{1}{K}
\sum_d
\mathbb E_{(x,y)\sim\mathcal C_d}
\operatorname{CE}
\left(
f_{\theta_M}(x,u_d)P_d^\top,y
\right).
\]

Reuse the existing implementation of the instruction-conditioned LEAF forward, normalisation and fixed text-prototype readout.

**Do not add any classification head.** Keep the original EEG Tower + Q-Former architecture unchanged.

Experiment settings:

- Calibration samples: 512 per dataset, using the same fixed calibration indices as the previous POC.
- Batch size: 128.
- Equal-frequency dataset sampling.
- Optimiser: Adam.
- Learning rate: `1e-2`.
- Total optimisation steps: 500.
- Seed: 0.
- Validate at step 0 and every 50 steps.
- Select the best coefficients by source validation macro-BAcc, breaking ties using macro CE.
- Keep bf16 training and fp32 evaluation consistent with the previous experiment.

Do not tune additional hyperparameters in this round.

Note that softmax logits have a different gradient scale from unconstrained coefficients. Record the optimisation curves so we can distinguish poor convergence from the effects of the constraint.

## 4. Implementation requirements

Make minimal modifications to the existing code.

Prefer extending `merge_poc.py` rather than introducing a new merging framework.

Reuse:

- Existing parameter grouping.
- Expert checkpoint loading.
- Calibration dataset loading.
- Differentiable model merging using `torch.func.functional_call`.
- Validation and checkpoint-selection logic.
- Final checkpoint folding and saving.
- Existing evaluation functions.

Only the merging coefficient parameterisation and parameter composition need to change.

For the softmax methods, the coefficients must remain differentiable during merging optimisation.

Maintain the previous policy:

- Merge selected trainable model parameters only.
- Copy buffers from the original pretrained checkpoint.
- Keep unused pretrained parameters unchanged.
- Do not subtract tensors containing infinite causal-mask values.
- Do not change BatchNorm behaviour.

Save the final merged result as a normal LEAF checkpoint, requiring no experts during inference.

Use new output directories:

```text
runs/poc/merge_softmax_global/
runs/poc/merge_softmax_block/
```

Do not overwrite any previous results.

## 5. Evaluation

Extend `eval_poc.py` only as needed to evaluate the new checkpoints.

Evaluate the following models:

1. Original LEAF.
2. Joint Fine-Tuning.
3. Uniform Merging (previous result).
4. Global Unconstrained Task Vector Merging (previous result).
5. Block-wise Unconstrained Task Vector Merging (previous result).
6. Global Softmax Weighted Averaging (new).
7. Block-wise Softmax Weighted Averaging (new).

Use exactly the same source test splits as the previous experiment.

Also evaluate on external `MI_Weibo2014`, using the existing evaluation protocol.

Do not use test data for coefficient optimisation or model selection.

Report:

- Per-dataset test balanced accuracy.
- Source macro-BAcc.
- External Weibo BAcc.
- Difference relative to Joint.
- Validation curves.
- Learned merging coefficients.

Save the combined result to:

`runs/poc/results_round2.csv`

For both softmax methods, save coefficient tables and heatmaps using the existing output conventions.

## 6. Analysis

After completing the experiment, create:

`runs/poc/Softmax_Merging_Experiment_Report.md`

The report should include:

### A. Main comparison

A table comparing all previous merging methods and the two new methods, including Joint.

### B. Global vs Block-wise

Determine whether block-wise softmax improves over global softmax.

### C. Softmax vs Unconstrained Merging

Compare the new methods against their corresponding unconstrained task-vector baselines.

Analyse whether restricting merging to convex combinations improves or worsens the results.

### D. Dataset-specific behaviour

Pay particular attention to SSVEP_OpenBMI and MI_PhysioNet, which suffered substantial losses in the previous experiment.

Determine whether softmax merging reduces those losses and whether improvements come at the expense of other datasets.

### E. Learned coefficients

Examine whether some experts dominate particular blocks.

Report the coefficient distributions and any notable patterns in the EEG Tower and Q-Former.

### F. Conclusion

Answer these questions directly:

1. Does softmax-constrained parameter averaging improve over the previous unconstrained merging?
2. Does block-wise merging remain beneficial?
3. Does either new method approach or exceed Joint?
4. What is the most likely remaining bottleneck?
5. What single change should we investigate next?

Distinguish observed results from interpretations and hypotheses. Do not invent conclusions if the results do not support them.

## 7. Research POC constraints

This is research code, NOT production software.

**Avoid over-engineering.**

- Do not refactor the entire training pipeline.
- Do not build a general-purpose merging library.
- Do not add configuration frameworks or abstract interfaces.
- Do not add extensive defensive programming.
- Do not write large test suites.
- Do not run multiple smoke-test stages.
- Do not retrain the five experts or Joint.
- Do not change the original dataset preprocessing.
- Do not add extra regularisation, distillation, Fisher weighting or other algorithms.
- Do not introduce new trainable modules or classification heads.

Keep the implementation short, readable and scientifically interpretable.

Inspect the code, make the changes, and run the actual two merging experiments. Fix issues directly if execution fails.

After execution, summarise the actual results, relevant modified files, and the location of the report.

**The objective is to isolate the effect of softmax-normalised weighted averaging, without confounding it with other methodological changes.**