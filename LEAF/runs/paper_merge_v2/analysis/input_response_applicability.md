# Scope of the input-response diagnostic

The diagnostic compares frozen initial, full-pool CE and full-pool CE+KL
models against each task's frozen expert. All selected EEG rows come from
the existing training pools. It does not measure independent generalization
or use Joint weights to construct a model.

For K classes, the Helmert matrix Q has K-1 orthonormal rows and satisfies
Q^T Q = I - 11^T/K. We differentiate Q times the native cosine-logit vector.
This retains every centered-logit contrast and removes the common-logit
direction, which cannot change softmax probabilities or argmax decisions.
The reported Jacobian is the full native-autograd local branch derivative
over these contrasts; it is not a random projection estimate.

The three input metrics use the identity, the empirical channel-support
projector, or the square root of the original512 channel covariance,
normalized by sqrt(trace(C)/65). This is a channel metric replicated at
every time point. It assumes no temporal covariance and is not an estimate
of an EEG data manifold, physiological intervention, or deployment noise.
Projecting onto channel support removes directions with no empirical
variance; it does not establish that remaining perturbations are realistic.

## Why a local derivative is not a finite-noise mechanism

The following is a direct Taylor calculation, not a new theorem. Write
d(x) = f_student(x) - f_teacher(x), with zero-mean noise z of covariance C.
Under twice differentiability and sufficient integrability for taking
expectations, the second-order expansion is

E ||d(x + sigma z)||^2 = ||d(x)||^2
+ sigma^2 [tr(J_d C J_d^T) + sum_k d_k tr(C H_dk)]
+ o(sigma^2).

With stronger smoothness and symmetric noise, the remainder can be bounded
at fourth order. The residual-times-curvature term is generally of the
same order as the Jacobian term. For d(x)=1+x^2 at x=0 with standard Gaussian
z, the exact expectation is 1+2 sigma^2+3 sigma^4 although J_d(0)=0.
Thus noisy-output matching does not isolate Jacobian matching for an
arbitrary nonlinear residual. Local affine regions remove the curvature
term, but finite perturbations can cross their boundaries. LEAF includes
GELU, softmax attention, normalization and max pooling.

The failed v2 smoke illustrates that distinction. On the first OpenBMI
teacher example, a relative input step1e-4 gives a directional finite
difference near194.03 while autograd gives141.63. The discrepancy persists
in float64, and the two-example probe changes43/47 pool selections on the
positive/negative side. At steps1e-8 and1e-9, no selections change and
float64 differences agree with autograd to relative error below4e-9.
Native and float64 gradients agree to about2e-6 in that probe. The v3
engineering gate therefore checks float64 local derivatives, retains the
original finite-step diagnostics, and leaves scientific FP32 Jacobians
unchanged. This verifies a derivative implementation, not finite-noise
linearity or a cause of the BAcc gap.

## Prior work and decision boundary

Derivative supervision and random-projection approximations already appear
in [Sobolev Training](https://papers.nips.cc/paper_files/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf),
Section2. Input-Jacobian distillation and its noise interpretation appear
in [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a/srinivas18a.pdf),
Sections3–4.1. Choosing a channel metric alone does not establish novelty.

[What Mechanisms Does Knowledge Distillation Distill?](https://proceedings.mlr.press/v243/wu24a/wu24a.pdf)
defines mechanisms through interventions on controlled latent variables.
Our derivative differences are not equivalent evidence. A large mismatch,
even when teacher and student predictions agree, would identify an unfit
quantity. A targeted intervention would still need to improve performance
and demonstrate that the relevant response change explains that effect.
No derivative-based candidate or loss coefficient has been selected.
