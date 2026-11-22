# Fixed task mixing: what the control can establish

At a fixed parameter vector, let `g_t` denote the class-weighted mean
gradient of task `t`, and let `g_bar = (1/T) sum_t g_t`. Define
`V = (1/T) sum_t ||g_t - g_bar||^2`. A uniformly sampled subset of `q`
distinct tasks has mean-gradient variance

`E ||(1/q) sum_{t in S} g_t - g_bar||^2 = (T-q)/(q(T-1)) V`.

This follows by expanding the squared norm: each centered gradient appears
with probability `q/T`, and each distinct pair with probability
`q(q-1)/(T(T-1))`; the sum of all centered gradients is zero. For `T=16`
and `q=4`, the ratio is `0.2`. This is a finite-population identity, not a
new optimization result or a generalization bound.

The frozen schedule uses random task permutations already present in the
native128 control. Every16-step block is regrouped into four task groups;
each group's128-row batches are split into four32-row chunks. Every four
mixed updates visit all16 tasks, and every16 updates preserve exactly the
original task/example multiset. Across100blocks the recorded cached-gradient
variance ratios are0.201219,0.200393,0.199039 at the three existing states.

The reference micro4 arm also performs four32-row forward/backward passes,
but uses one task for all four. It averages their losses and applies Adam
once. Thus total examples, update count, optimizer, class weights, data
multiset, microbatch size and accumulation implementation are matched.
The older native128 arm additionally measures sensitivity to floating-point
batch grouping. All runs keep LEAF in eval mode. The output remains an
ordinary static checkpoint.

The identity describes task means. With independent within-task sampling,
equal total examples can preserve the average within-task variance term,
while reducing the between-task component. The actual frozen stream uses
without-replacement rows, shared blocks and temporally correlated chunks;
neither independence nor a measured within-task covariance is established.
Existing gradients use the old512 calibration rows and weights, not the
full training pool used by this experiment.

Adam is nonlinear in the gradient history. Mixing changes its first and
second moments and the frequency with which each task affects updates.
Consequently a positive result would support this specific batch-organization
intervention. It would not uniquely attribute the result to variance, prove
gradient surgery unnecessary, or establish a novel optimizer. A negative
result would not refute the finite-population identity.

Primary comparison is the fixed six-checkpoint mean, mixed versus same-task
micro4. Native128, initial merger, Joint, all16 task effects, reference512
fit and measured costs remain visible. The protocol performs no task,
snapshot, group-size or learning-rate selection from these outcomes.

Sources: `inputs/task_mixing_design.json`,
`analysis/parameter_gradient_conflicts.json`,
`analysis/task_mixing_cached_variance.json`.
