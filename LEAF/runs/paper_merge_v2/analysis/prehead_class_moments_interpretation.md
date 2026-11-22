# Class moments resolve the raw-MSE ambiguity

The completed diagnostic separates changes in class centers from changes
within classes. It does not support calling the paired model's variance
reduction harmful feature collapse, or prioritizing a residual-only loss.
The paper-contribution and substantial stable Joint-gap objective remains
active and unachieved.

All four fixed means were evaluated on the same original fitting multiset.
This added474520 native forward examples, zero teacher forwards and zero
optimizer updates. Only selected training EEG rows were read. Both original
workers completed; model states were unchanged. Repeated CE, BAcc and all
three target-MSE values agree exactly with the preceding fitting audit.
The GPU1 log contains an allocation warning followed by normal completion;
no retry or replacement run was launched.

## Exact endpoint decomposition

For each true class, let`mu_s` and`mu_t` be exposure-weighted feature means,
`D=||mu_s-mu_t||²/d`, `V_s=E||h-mu_s||²/d`,
`V_t=E||t-mu_t||²/d`, and`C=E<(h-mu_s),(t-mu_t)>/d`.
Each component uses the original class weight and exposure mass. Then
`Q_paired=D+V_s+V_t-2C`. Class-mean-target MSE is exactly
`Q_mean=D_stored+V_s`, where`D_stored` uses the actual FP32 stored teacher
mean. Moments are accumulated in FP64. The largest loss-identity error is
5.829e-15; FP32 mean-rounding corrections are retained, not assumed zero.

| Fixed mean | Centroid MSE D | Student variance V_s | Covariance C | True-class-centered cosine |
| --- | ---: | ---: | ---: | ---: |
| CE |0.299388|0.350509|0.111049|0.408689|
| Paired |0.207709|0.274417|0.106623|0.439383|
| Class mean |0.233382|0.224354|0.083438|0.396424|
| Class shuffle |0.234086|0.225242|0.083610|0.396270|

Numbers are task-macro averages. Cosine is computed per task before macro
averaging, not reconstructed from macro moments. It centers each model at
its own true-class mean, unlike the preceding teacher-mean-centered cosine.

Relative to CE, paired-target MSE changes by

`−0.158918 = −0.091679 (centroids) −0.076092 (student variance) +0.008853 (−2C)`.

Both centroid discrepancy and student variance decrease on all16tasks.
Raw covariance decreases on12tasks and increases on4; centered cosine
increases on13tasks. A larger cosine does not imply greater raw covariance.
This identity describes endpoints, not parameter-gradient effects or a
causal account of the optimization trajectory.

Teacher variance has task-macro value0.189663. The paired student's macro
variance is still1.447times that value and exceeds teacher variance on
12of16tasks. Mean/shuffle ratios are1.183/1.188 and exceed the teacher on
8tasks each. Variance reduction alone therefore does not establish collapse
or explain weak accuracy. These are trace-like second moments; rank loss or
loss of useful semantic directions was not measured by this audit.

## What the pairing controls show

Relative to both mean and shuffle, the paired model has higher within-class
variance, covariance and centered cosine on every task, and smaller centroid
discrepancy on every task. The paired intervention changes sample-related
geometry and preserves more variation than the controls. Yet its validation
advantage remains−0.019391pp versus mean and+0.038905pp versus shuffle; both
task-deletion sensitivity ranges cross zero. Those ranges are not confidence
intervals or an equivalence test.

The paired model's+0.103475pp versus matched CE closes3.7034% of the Joint
gap remaining after CE and still leaves2.690554pp. One bank and exposed
validation data do not supply the requested stable paper-level evidence.
The generic centering/variation-transfer idea also has direct precedents in
DIST and IFVD; detailed read scopes are in`../literature/review.md`.

## Decision

Retire this fixed raw-prehead MSE recipe as the current paper candidate.
No residual-only objective, learning-rate sweep or auxiliary-scale sweep is
selected from these outcomes. A subsequent distillation candidate needs
evidence for a useful signal or a specific optimization obstruction beyond
the fact that raw MSE mixes centers and within-class variation. Existing
results neither rule out all teacher information nor establish such an
obstruction. The strongest replicated practical gain remains full-pool CE
repair, which is insufficient by itself as the intended new contribution.

Sources:`prehead_class_moments_summary.json`,
`prehead_class_moments_integration.json`,`prehead_distillation_summary.json`.
Plan:`../inputs/prehead_class_moments_plan.json`.
Figure:`figures/prehead_class_moments.{png,pdf}`.
The new audit is subsequent to the four-arm results, not preregistered before
them. Source workers and historical outputs were preserved unchanged.
