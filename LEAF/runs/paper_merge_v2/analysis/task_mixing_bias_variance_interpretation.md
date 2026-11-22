# Actual fitting-stream check on the shared output bias

The initial task-mean variance calculation omits within-task sampling.
`analysis/task_mixing_initial_bias_gradient_variance.json` closes this gap
for one specific parameter block: the768-dimensional bias of the final
linear output layer. At a fixed model, summing the raw-output derivatives
is exactly its bias gradient. Existing cached initial outputs therefore
allow both full1600-step sample schedules to be checked without another
model forward or backward pass.

The calculation uses every204800fitting-stream exposure, full-pool class
weights and the prescribed task grouping. The old512 empirical bias
gradients independently replay the existing native full-network cache,
with maximum relative L2 error8.724419e-7 across16tasks. The small error
reflects FP64 readout derivatives over cached FP32 forward outputs.

Mixed-task divided by same-task values are:

| Quantity | Ratio |
| --- | ---: |
| Total batch-gradient variance | 0.474966 |
| Between-task-mean component | 0.202603 |
| Within-task-sampling component | 1.046448 |
| First-moment deviation, all1600steps | 0.579355 |
| First-moment deviation, last800steps | 0.588365 |

The variance decomposition includes the observed covariance between task
means and sampling residuals; it does not assume independence. The
between-task term follows the approximately0.2ratio expected from the
grouping, while the sampling residual does not fall. Consequently, the
20%task-mean result and the earlier6–8%fixed-task-mean momentum result
cannot be used as estimates of total batch-gradient or moment variance.

This is evidence that grouping changes actual initial bias-gradient
fluctuations. It covers768parameters out of25,751,552trainable parameters,
at one frozen model. The first-moment recursion still holds the model
fixed and omits Adam's second-moment preconditioner. No claim about whole
network optimizer variance, training trajectories or validation accuracy
follows from it alone. The paired training experiment remains the test of
whether the intervention produces useful performance.
