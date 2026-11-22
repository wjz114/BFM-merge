# Frozen teacher coordinates: what the active comparison can establish

The active experiment changes teacher feature targets while preserving the
teacher's represented decision function. It compares two ordinary LEAF
students with the same M0 initialization, fitting stream, balanced CE,
original auxiliary weights, Adam settings, 1600 updates and six-checkpoint
mean. It is an exploratory single-bank comparison on exposed development
validation; the paper objective remains unachieved.

For each query row, let the teacher prehead representation be
`h_t = q_t diag(gamma_t) + beta_t`. A fixed orthogonal matrix `R_t`, shared by
all eight query slots, is fitted from the existing paired training cache.
It fixes the all-ones axis. The target presented to the unchanged student is

`h_aligned = ((h_t - beta_t) / gamma_t) R_t diag(gamma_0) + beta_0`,

where `gamma_0, beta_0` are M0's final LayerNorm affine parameters. Every
denominator is checked nonzero. For each corresponding teacher head block,

`W_aligned = W_t diag(gamma_t) R_t diag(gamma_0)^(-1)`.

The head bias adds the original `W_t beta_t` contributions and subtracts
the new `W_aligned beta_0` contributions across query blocks. Substitution
recovers the original teacher raw output exactly in real arithmetic.
The transformed head is used only to audit the cached target conversion;
the student retains its original architecture and native head. At inference
there is no task-dependent rotation or extra adapter.

The preparation checks all118630cached rows, including845zero-weight rows:
maximum FP64 affine raw-output error4.761e-13, target-FP32-rounding cosine
error1.184e-8, and zero prediction changes. Actual fitting uses117785unique
rows with204800scheduled exposures. Source/cache/rotation hashes are retained
in `aligned_query_target_preparation.json` and the frozen design.

The representation diagnosis is narrower than an accuracy mechanism. In the
normalized query frame, all32heldout-training-half total MSE checks improve.
Class means account for75.7433% of average absolute error reduction, while
within-class error decreases11.0541% on average, also32/32. Means-only fitting
typically harms within-class agreement. These halves are held out only from
the rotation fit; they are not unseen by the expert or merger training.
The75.7433% decomposition is in normalized query coordinates, not a claimed
identity for the native affine-prehead training objective.

The original auxiliary weights are unchanged. Consequently, initial auxiliary
gradient RMS becomes0.73785–0.94725 of the original. Mean prehead CE-gradient
cosine increases0.36598 to0.39277, but negative per-example fractions increase
2.69287% to3.08984%, and the fixed-scale CE dot product decreases on average.
These cached feature gradients are not parameter gradients or actual Adam
updates. They do not establish uniformly reduced interference.

If the primary comparison is materially positive and stable, class-mean,
shuffled-pair, auxiliary-strength and fresh CE controls are required before
claiming sample-specific transfer or expanding seeds. A small gain cannot
be promoted simply because the representation fit improves. The completed
fixed-coefficient weight-merging counterfactual already illustrates the
distinction: response alignment improves MSE but reduces macro BAcc to35.806%.

Orthogonal and Procrustes feature distillation are established by
[VkD](https://arxiv.org/abs/2403.06213) and
[geometry-aware representational alignment](https://arxiv.org/html/2509.25253v1).
The present target parameterization and exact source-function audit do not,
by themselves, establish novelty. The official latter implementation was
read only in the recorded measure and dispatch sections; no reproduction
or complete proof audit is claimed.
