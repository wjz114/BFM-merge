# Completed native prehead target controls

The fixed paired-prehead recipe does not currently establish a material
advantage or a sample-pairing mechanism sufficient for the paper objective.
All four1600-update arms completed, all396 tensors of each six-state mean
were reconstructed exactly, and native export checks passed.

| Fixed mean | Validation macro BAcc | Difference from matched CE |
| --- | ---: | ---: |
| CE |65.265175%|0|
| Paired prehead |65.368650%|+0.103475pp|
| Teacher class mean |65.388041%|+0.122865pp|
| Class-preserving shuffle |65.329745%|+0.064569pp|

Paired versus CE improves10tasks and declines6, with task-deletion means
from+0.049767 to+0.272819pp. Paired versus class mean is−0.019391pp
(7up,8down,1tie), and versus shuffle is+0.038905pp(10up,5down,1tie).
Both target-control task-deletion ranges cross zero; they are sensitivity
analyses, not confidence intervals or proof of equivalence. One expert bank
and the already exposed validation set do not establish replication.

The paired model retains a2.690554pp gap to historical FP32 Joint. It closes
35.1141% of the initial merger-to-Joint gap in total, but ordinary CE already
closes32.6187%. Its additional gain closes only3.7034% of the gap remaining
after CE. The class-mean arm closes4.3974% of that remaining gap. This is not
evidence of a substantial new method advantage. Fresh CE differs from its
historical native run by+0.030838pp; this is reported, not subtracted as noise
or used to invent a significance threshold.

## What the fitting diagnostic establishes

All four fixed means were evaluated on the exact original fitting multiset,
with training-exposure multiplicities and original full-pool class weights.
The audit used474520 native forward examples, cached teacher/initial states,
zero updates, and selected training EEG only. Both shards passed unchanged
model-state and native-head checks. All16initial target-loss replays passed.

Each auxiliary model lowers its own target MSE relative to CE on all16tasks.
Paired-target task-macro MSE falls from0.617462(CE) to0.458544(paired), a
25.7373% reduction of those macro values. The auxiliary loss is active and
its target is fitted; the weak performance result cannot be explained simply
by a zero or disconnected auxiliary gradient.

For the actual stored teacher class mean`mu`, define`a=h-mu` and`r=t-mu`.
The exact endpoint identity is`Q_paired=Q_mean+V_teacher-2*C`, where`C` is the
weighted inner product of`a` and`r`. From CE to paired distillation, the macro
change is

`−0.158918 = −0.167771 + 0.008853`.

Thus the decrease in distance to teacher class means accounts for the net
MSE reduction, while the raw residual inner-product term offsets part of it.
The same decomposition with the actual task scales is
`−0.079011 = −0.084636 + 0.005625`. This is endpoint algebra, not attribution
of training dynamics or parameter-gradient effects.

Raw paired residual inner product falls from0.111049(CE) to0.106623(paired);
only4tasks increase it. Normalized residual cosine rises from0.325558 to
0.361190(13tasks increase). The paired model exceeds both mean/shuffle
controls in normalized pairing on all16tasks, so the intervention does alter
sample-related geometry. That geometric difference does not yield a clear
performance advantage over the controls. A larger normalized cosine must
not be reported as a larger raw residual signal.

`Q_mean` includes both student within-class dispersion and student/teacher
class-centroid discrepancy. The stored loss summaries do not separate them.
Consequently, neither feature collapse nor useless teacher residual content
has been demonstrated. A further mechanism audit would need classwise
student moments to resolve that ambiguity before designing a residual-only
objective. No such new training condition is registered or launched here.

## Scope and next decision

This experiment supports retiring this fixed raw-prehead MSE recipe as the
current paper candidate. It does not rule out every intermediate-feature
distillation method. No learning-rate, auxiliary-scale, or seed search is
allocated merely to rescue this small increment. Generic feature distillation,
class-conditioned targets, and residual-directed repair have direct prior
art in the literature ledger.

The primary and mechanism figures were visually inspected:
`figures/prehead_target_controls.{png,pdf}` and
`figures/prehead_fitting_mechanism.{png,pdf}`.
Authoritative numeric sources are`prehead_distillation_summary.json`,
`prehead_fitting_summary.json`,`prehead_pairing_decomposition.json`, and
`prehead_mechanism_integration.json`. Total scientific training6400updates;
engineering96updates includes the preserved reference-selection correction.
Teacher/initial caching and analysis forwards remain separate from training
cost. Expert construction48000updates and original coefficient fit1600updates
remain part of the pipeline ledger; no end-to-end efficiency claim follows.

The full paper-contribution, clear-mechanism and substantial stable Joint-gap
objective remains active and unachieved. Independent confirmation is unresolved.
