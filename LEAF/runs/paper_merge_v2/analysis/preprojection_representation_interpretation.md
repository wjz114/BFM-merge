# Last-projection structure survives; usefulness remains untested

The frozen diagnostic completed on all16 tasks, the same128 original and128
replacement training rows per task, and four models: own expert, initial
merger, historical1600-step CE mean and historical1600-step CE+KL mean.
The new4800-step models are not part of this diagnostic. Both subsets are
development-exposed training data; ADHD has9 overlapping selected rows.
There were16384 main forward examples and no optimizer updates or backward
passes. All native logits reproduce the preceding cache exactly, and model
states remain unchanged. The two cache shards took44.73s and27.84s; the CPU
report took33.45s. The PNG was visually inspected after completion.

The prehead representation is the2048-dimensional input to the final Linear
layer, after query LayerNorm, flattening and disabled dropout. Raw and unit
representations are its768-dimensional output before and after normalization.

| Teacher statistic, task-macro mean | Original | Replacement |
|---|---:|---:|
| Within-true-class / total variance, prehead |17.8424%|19.1994%|
| Within-true-class / total variance, raw |6.0471%|7.5432%|
| Within-true-class / total variance, unit |6.6402%|7.9954%|
| Within-true-class / total variance, centered logits |5.5757%|7.2157%|
| Own-head null fraction of prehead within-class variance |43.7202%|43.8660%|

Every task in both pools has a larger within-class variance fraction before
projection than in its unit output. These ratios use different denominators
and representations; their difference is not a quantity of lost information.
The separate row/null decomposition establishes exactly which prehead
variation the current Linear map annihilates. All19 distinct head matrices
have rank768; the FP64 decomposition, energy identity and annihilation checks
pass. Null variation is invisible through that fixed head, but may contain
nuisance, noise or useful structure. A subsequently trained head need not
retain the same null space. None of these measurements labels it useful
knowledge or proves that recovering it improves classification.

Class-balanced linear CKA was computed after removing true-class means,
with32 within-class permutations retained as descriptive reference values.
The following statistic subtracts the permutation mean from paired CKA:

| Teacher versus student, prehead | Original | Replacement |
|---|---:|---:|
| Initial merger |0.239856|0.234212|
| CE |0.279250|0.266652|
| CE+KL |0.291469|0.285935|

KL-minus-CE is+0.012220 on the original pool, with8 positive/8 negative
tasks, and+0.019283 on replacement, with11 positive/5 negative. Their
leave-one-task-out mean ranges are[0.008265,0.014067] and
[0.013059,0.022450]. These ranges are sensitivity checks, not confidence
intervals. They show that output KL already changes intermediate paired
structure. They do not establish that structural agreement caused its
small validation gain, or that stronger agreement would improve accuracy.
Common initialization and common inputs may also explain paired structure.

Intermediate/relational KD has direct precedents. LELP additionally uses
class-conditioned PCA after removing classifier directions, then constructs
pseudo-subclass targets. Its native student predicts expanded subclasses
and sums them at inference, which does not directly preserve the ordinary
LEAF prototype interface. This limits direct reproduction under our export
constraint; it does not make generic prehead/null-space distillation novel.
Primary-source read scope is recorded in `../literature/review.md`.

Decision: this is a rationale for a bounded causal control, not a successful
method. Before a training launch, specify a paired ordinary prehead-feature
control and a class-preserving target intervention, with fixed data, loss
scale, update budget and checkpoint window. Compare against ordinary CE and
retain all tasks. No rank, layer, temperature or coefficient sweep is
registered. No training pilot has yet been frozen or launched by this note.

The prior input-Jacobian worker read whole subject arrays before selecting
training rows, including unused validation EEG in host memory. Its actual
inference remained training-only. The correction is documented in
`input_response_read_scope_correction.md`; this new cache directly requests
selected training EEG rows. Frozen sources and numeric results are preserved.

Numerical source: `preprojection_representation_summary.json`.
Design: `../inputs/preprojection_design.json`.
Figure: `figures/preprojection_representations.png`.
