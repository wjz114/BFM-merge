# Specialist output compression on the existing fitting stream

The cached specialist outputs vary mostly between classes. This provides a
concrete description of the output bottleneck, but it does not establish that
instance variation is dispensable for training or explain causally why the
existing KL control adds only 0.095397 percentage points over ordinary CE.

`teacher_output_compression.json` was computed in 11.29 CPU seconds from
existing caches. It reuses the original 204800 fitting exposures and their
full-pool class weights, normalized within each task. It performs no model
inference, optimization, validation evaluation or official-test access.
Weighted teacher accuracy and teacher-to-initial KL exactly replay the prior
decision audit. All variance and KL decomposition identities pass.

## What is compressed

For normalized row weights w_i, outputs z_i and group g_i, the weighted
centroid m_g minimizes within-group squared error. The audit checks

    sum_i w_i ||z_i - m||²
      = sum_i w_i ||z_i - m_g(i)||² + sum_g W_g ||m_g - m||².

This is applied separately to centered logits, probabilities and final
normalized features. Centering each logit row removes its irrelevant common
offset. Groups use the true class, the teacher argmax, or their pair.

For teacher probabilities q_i and their group means q_g, the audit also checks

    sum_i w_i KL(q_i || q_bar)
      = sum_i w_i KL(q_i || q_g(i)) + sum_g W_g KL(q_g || q_bar).

This describes the teacher's categorical output on the empirical row
distribution. It does not measure all task knowledge. Group means estimated
from teacher outputs remain teacher-derived targets; conditioning on the
teacher argmax preserves its decisions, including its mistakes.

## Results

The following are means of the 16 separate task-level ratios, not a ratio of
pooled cross-task energies. All tasks are retained in the JSON and figure.

| Grouping | Centered-logit variance remaining within groups | Probability information remaining within groups | Relative L2 change of the initial KL output gradient |
|---|---:|---:|---:|
| True class | 8.5806% | 8.4642% | 43.2938% |
| Teacher argmax | 1.9540% | 1.6492% | 17.1582% |
| True class and teacher argmax | 1.6367% | 1.3632% | 15.3909% |

For the gradient calculation, g_i = p_initial,i - q_teacher,i and the
replacement uses q_group,i. The reported quantity is
sqrt(sum w_i ||g_replacement,i - g_i||² / sum w_i ||g_i||²), not the
difference between scalar gradient norms. No parameter Jacobian or optimizer
state enters this comparison. The figure's first version used an ambiguous
axis label; the corrected `teacher_output_compression_v2` PNG/PDF is primary.

Within teacher-prediction groups, centered-logit residual variance averages
0.9153% for the eight binary tasks and 2.9926% for the eight multiclass tasks.
Their corresponding output-gradient changes are 8.5843% and 25.7320%.
Thus a small share of output variance can still matter relative to the
student's current mismatch. SEED3/4/5/7 retain material gradient changes.
ADHD's almost constant within-class logits are a special extreme, not a
description of every task.

## Binary geometry and a teacher-free reference

For two fixed prototypes p_0,p_1 and a unit feature u, let d=p_1-p_0.
The score difference is u^T d. Binary CE for class1 decreases monotonically
with this difference, so its optimum over unit features is u=d/||d||;
class0 uses the negative direction. This is a direct geometric identity,
not a new optimization method. The two resulting probability vectors can be
selected using the true label without reading an expert model.

Selecting those vectors using the teacher's prediction instead gives a
different, teacher-dependent reference. Across the binary tasks:

| Template selector | Teacher-to-template KL | Relative L2 gradient change | Pooled gradient cosine |
|---|---:|---:|---:|
| True label, teacher-free | 0.011280 | 31.5587% | 0.942007 |
| Teacher prediction | 0.001373 | 8.8033% | 0.993551 |

Teacher errors have mean weighted mass 2.1281% on these binary fitting
streams. They contribute 0.010433 of the 0.011280 macro KL to the true-label
template. This explains why replacing expert decisions by ground truth is
substantially different from removing the remaining continuous output detail.
It does not establish the effect of either replacement on validation.

Geometric margin saturation must not be described as nearly one-hot
confidence. The six binary MI tasks have attainable correct-class softmax
probability 0.757761 at native scale; Workload and ADHD have 0.777795 and
0.764988. Across all16 tasks, teacher entropy divided by log(K) averages
0.866679. The architecture bounds the logits, so high accuracy and a margin
near its geometric bound coexist with soft output probabilities.

## Relation to existing work and next decision

[Yuan et al., CVPR2020](https://arxiv.org/pdf/1909.11723), Sections3–4,
connect KD to learned smoothing and study manually constructed targets.
[Tang et al.](https://arxiv.org/pdf/2002.03532), Sections3–4,
separate class relationships from instance-dependent gradient effects using
partial distillation targets. These are direct precedents for generic
label-conditioned targets and diagnostic target replacement.

[Chandrasegaran et al., ICML2022](https://proceedings.mlr.press/v162/chandrasegaran22a/chandrasegaran22a.pdf),
Sections3–4, distinguish information erasure from other geometric and
temperature effects. Their setting does not establish a LEAF result, but
it is a reason not to equate compressed outputs with ineffective KD.

The audit supports a narrower claim: the final specialist interface carries
little within-prediction variation on these fitting examples, especially for
binary tasks, while even that residual can change the initial KD gradient.
A causal claim would require a controlled target-replacement intervention;
no such training experiment is registered or launched here. First complete
the ordinary CE duration comparison. This analysis is mechanism groundwork,
not a differentiated method, independent confirmation, or achieved paper goal.
