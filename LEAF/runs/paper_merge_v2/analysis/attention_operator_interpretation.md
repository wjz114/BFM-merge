# Attention operators: correspondence, activation effects and fixed ablation

The new evidence does not identify attention-factor cross terms as a useful
removal target. The fixed training-only intervention worsens balanced CE in
all three aggregate comparisons. No native-rank candidate or validation run
is justified by these results alone.

## Exact objects and restricted matching

LEAF has 32 attention modules with 8 heads of width 32. With column-vector
weight convention, query augmentation absorbs the query bias and value
augmentation absorbs the value bias:

\[
C_{QK}=Q_a^\top K/\sqrt{32},\qquad C_{VO}=O V_a.
\]

Key bias adds a constant across valid keys for each query and cancels in
softmax. Output bias is retained separately. The implementation respects the
native causal mask, and also handles Q-Former cross-attention's distinct
query and key inputs. It reproduces actual native attention outputs with
maximum relative L2 error 6.893656e-7 and maximum absolute error 9.566545e-6.

All 16 expert-to-base comparisons across the 32 modules have strict identity
row minima in their 8-by-8 squared-weight-distance matrices: no heads move.
This tests the concatenated Q/K/V/O and bias signatures with external
coordinates fixed. It does not eliminate continuous gauges or certify
activation-optimal alignment.

## Uniform factor means versus uniform operator means

For uniform expert averaging, the difference between averaging composed
operators and composing averaged factors is exactly their cross covariance.
The FP64 covariance identity replay has maximum relative error 3.279429e-13.

| Frobenius norm ratio | QK | VO |
| --- | ---: | ---: |
| Factor discrepancy / full operator mean | 0.184916% | 0.104177% |
| Factor discrepancy / operator mean change from base | 6.017900% | 4.092205% |
| Best rank-32 residual / factor discrepancy | 81.209717% | 82.687064% |

The rank-32 comparison is an unweighted SVD bound for this constructed target.
It neither uses activation geometry nor measures classification performance.
The uniform mean is not the objective of the learned tensor-coefficient merger.

On all original 512 training examples per task, with local inputs supplied by
the unchanged learned merger, the two uniform constructions differ by
0.563545% in centered attention-score L2 norm and 0.361821% in attention-output
L2 norm. Mean attention total variation is 0.00233940; the largest query/head
value is 0.05421375. Per-task output-relative differences span 0.2397–0.4920%.
These pooled norms weight by activation element count; the report also keeps
all per-task/per-module values. Neither construction is a teacher ensemble.

## Learned-merger quadratic terms

Writing each factor as base plus its learned merged delta gives an exact
linear-plus-quadratic expansion, for example

\[
C_{QK,M}-C_{QK,0}
=\frac{Q_{a,0}^\top\Delta K+\Delta Q_a^\top K_0}{\sqrt{32}}
+\frac{\Delta Q_a^\top\Delta K}{\sqrt{32}}.
\]

The analogous VO quadratic term is delta-O times delta-V. These terms are
9.873303% and 7.654506% of their respective operator changes in pooled
Frobenius norm. Their exact expansion replay error is at most 5.220872e-15.
They include both within-expert and between-expert products. Their size is
coordinate-dependent and does not classify them as harmful interference.

At fixed merger inputs, removing both terms changes attention outputs by
3.125167% and centered scores by 4.055289%, with mean attention TV 0.01451725.
That observation motivated one fixed end-to-end training-only ablation.
Removal strength is one, in every attention module; there is no fitted scale,
layer selection, rank selection or validation selection. Softmax and all
other model components remain nonlinear.

| Training-only condition | Macro BAcc (%) | Delta (pp) | Balanced CE | CE delta |
| --- | ---: | ---: | ---: | ---: |
| Native merger | 85.835969 | — | 0.746885842 | — |
| Composed-operator replay | 85.835969 | 0 | 0.746885832 | numerical replay |
| Remove QK quadratic term | 85.855307 | +0.019338 | 0.747005802 | +0.000119960 |
| Remove VO quadratic term | 85.722571 | -0.113398 | 0.747477770 | +0.000591928 |
| Remove both | 85.676675 | -0.159294 | 0.747695981 | +0.000810139 |

QK-only has 6 tasks up, 6 down and 4 ties; its leave-one-task-out BAcc change
spans -0.018080 to +0.046875 pp. VO-only has 2 up, 8 down and 6 ties; all
task-deletion means are negative. Both removed has 2 up, 9 down and 5 ties;
its task-deletion range is -0.183995 to -0.117624 pp. These are sensitivity
ranges, not confidence intervals. Balanced CE improves on only 6/3/4 tasks
for QK/VO/both, respectively.

The composed replay has identical BAcc on every task and maximum logit error
2.950430e-6. All native metrics replay against the earlier gradient cache.
Every model-state tensor remains unchanged. The ablated ambient operators
need not have native rank 32, and no checkpoint was exported. There were
8192 observational and 40960 intervention/replay forward-example exposures,
zero optimizer updates, and no validation or official-test inference.

This result rejects the proposed fixed removal as a useful calibration
repair. It does not refute optimizing new operator targets, activation-aware
operator alignment, or tangent-space expert training. It also supplies no
generalization estimate. The paper-level goal remains unachieved.

## Prior art and artifacts

[SLOA](https://arxiv.org/html/2610.06415v1) already merges composed QK/VO
operators and aligns local behavior on progressively merged inputs, including
affine augmentation. [GeoMerge](https://arxiv.org/html/2604.27155v1) explicitly
treats factorization symmetry and quotient geometry. Operator composition or
gauge invariance is therefore not a new contribution here.
[Task Arithmetic in the Tangent Space](https://arxiv.org/html/2305.12827v3)
distinguishes post-hoc linearization from training linearized models; its
nonlinear-advantage results caution against equating nonlinear terms with
harmful interference. The local attention ablation is neither whole-network
linearization nor a reproduction of those methods.

Sources and hashes are in `attention_operator_summary.json`. Detailed data:
`attention_operator_geometry.json`,
`../seed0/attention_activation_diagnostic/run.json`, and
`../seed0/attention_quadratic_training_ablation/run.json`.
Figure: `figures/attention_operator_diagnostics.{png,pdf}`.
