# What gradient-mismatch theory can explain here

The current tensorwise merger cannot be assigned the published scalar
task-arithmetic error identity with its residual terms silently set to zero.
The theory supplies useful questions, but it does not yet explain the local
Joint gap or establish a new repair method.

The primary-source method audit covers
[Uncertainty-Based Gradient Matching v2](https://arxiv.org/html/2310.12808v2),
Sections3–3.3. Its task-arithmetic derivation assumes stationary solutions
with a shared quadratic penalty centered on the pretrained model. A Taylor
expansion leads to Hessian/Fisher preconditioning; it also discusses
iterative refinement. These directions are established prior art. The
appendix proofs and official implementation were not audited in this pass.

[On Task Vectors and Gradients v6](https://arxiv.org/html/2508.16082v6),
Sections2–3.3, analyzes fixed-step full-batch gradient descent. Its first
“epoch” is one full-gradient update. The later-iterate expansion depends on
curvature and a small-step approximation; the paper explicitly limits exact
transfer to Adam. Early checkpoint merging and repeated short training/merge
rounds also have cited precedents. Reading these statements is not a
verification of all bounds or their applicability to LEAF.

## Exact accounting for arbitrary checkpoints

The following is a direct algebraic identity, not a novelty claim. Let
`theta0` be the released base, `theta_t` expert checkpoints, and `thetaJ` the
historical Joint checkpoint. Write `delta_t = theta_t - theta0`. For a chosen
common differentiable evaluation loss `L_t`, let `g_t = grad L_t`. Choose a
positive-definite matrix `H` and scalar task weights `alpha_t`. Define

```
r_t = g_t(theta_t) + H delta_t
r_J = sum_t alpha_t g_t(thetaJ) + H (thetaJ - theta0)
M   = theta0 + sum_t A_t delta_t
```

Here `A_t` is diagonal with a constant coefficient within each parameter
tensor, exactly representing tensorwise merging. Adding and subtracting the
defined terms gives

```
H (thetaJ - M)
  = r_J - sum_t alpha_t r_t
    - sum_t alpha_t [g_t(thetaJ) - g_t(theta_t)]
    + H sum_t (alpha_t I - A_t) delta_t.
```

The final term is the difference between scalar task weights and tensorwise
weights. If all `A_t = alpha_t I` and both sets of stationarity residuals
vanish, the expression reduces to the familiar scalar identity. Those are
additional conditions, not consequences of having trained checkpoints.

This identity has a useful negative implication: measuring only the middle
gradient-difference term does not identify the cause of the parameter gap.
It can omit equally large stationarity and coefficient terms. Evaluating
the whole equality would verify accounting, not prove a causal mechanism
or a relationship to balanced accuracy.

## Local mismatches that must remain explicit

Historical experts and Joint used ordinary CE, minibatch AdamW, nonzero
decoupled weight decay, a tokenizer/rest LR split, warmup/cosine scheduling,
BF16, and training dropout with fixed BN. Existing gradient caches instead
use original512 rows per task, empirical class-balanced CE, FP32 and eval
mode. Those gradients need not be stationary even at a historical endpoint.
Decoupled AdamW decay is not automatically the assumed shared quadratic
penalty centered on the released base.

The current supervised merger additionally has tensor-dependent weights
and a fixed checkpoint mean. The repair runs optimize a different empirical
objective and use a much shorter trajectory. No common `H`, stationary
residual bound, Taylor remainder bound or BAcc guarantee has been established.
Fitting such quantities to Joint would be an oracle diagnostic and must not
enter candidate construction.

No new Joint/own-expert gradient cache or curvature method is launched from
this reading. The next concrete control isolates an already documented
training-recipe difference: the historical10:1 tokenizer/rest LR ratio,
with identical data and two initializations. This is an optimizer control,
not a proposed paper contribution.
