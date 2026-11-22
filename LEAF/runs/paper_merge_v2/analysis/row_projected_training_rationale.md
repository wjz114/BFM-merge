# Actual optimization in row and shuffled expert spaces

The previous fixed-update diagnostic found a useful row-structure signal,
but its row projection remained below full CE. This experiment asks whether
optimization within that space can provide a stronger solution. It does not
use the already learned CE displacement to construct or initialize the bases.

There are three fixed conditions: ordinary full CE, row-space updates, and
same-dimension shuffled-space updates. The initial merger, complete fitting
schedule, class weights,1600updates, batch128, Adam1e-5/no decay, nativeFP32,
dropoutoff/BNfixed and six-state average are shared. All16 exposed validation
tasks are retained. The total scientific budget is4800updates;96additional
updates are reserved for engineering. No teacher forwards are required.

## Why update projection instead of latent-coordinate Adam

Different orthonormal bases describe the same subspace. Adam's coordinatewise
second moments can nevertheless produce different physical updates when
applied to those latent coordinates. We keep the Adam calculation in native
weight coordinates in all conditions, then apply a fixed projector to its
proposal. This makes the effect of the constraint explicit.

Let each group's orthonormal row basis be`Q`, so`P=Q^T Q`. At the current
materialized weights, ordinary Adam proposes`u`. The constrained conditions
compute

`c=Q u`, `v=Q^T c`, `gamma=||u||/||v||`,

where the norm sums over all active parameters, not per layer or row. If the
projection is zero, the update is zero and the event is logged. Coordinates
accumulate as`beta <- beta + gamma*c`, and weights are materialized as
`theta_initial + Q^T beta`. Shuffled conditions apply the fixed coordinate
permutation before encoding and undo it after decoding. Full CE uses the
original Adam directly and bypasses all projection machinery.

The restoration factor matches the projected step to that arm's own current
Adam proposal. It does not assert that different arms take identical step
norms once their trajectories diverge. It also changes the algorithm relative
to unscaled projected Adam; that rule is fixed before scientific outcomes.
Both the proposal and actual rounded parameter-step norms are recorded every
update, alongside the factor. No factor cap, rank or learning-rate search is
defined in this experiment.

For an ideal orthonormal basis, replacing`Q` by`R Q` for orthogonal`R` leaves
`P` unchanged. Coordinates transform as`beta -> R beta`; the projected update,
restoration factor and materialized weights remain unchanged by induction.
This is invariance to the choice of basis *inside the same fixed space*, not
rotation invariance of Adam with respect to arbitrary model coordinates.
The FP64 toy check verifies this distinction numerically. It is not a
convergence theorem or a bitwiseFP32 guarantee.

## Construction and implementation

Bases use original expert-minus-base weights and the previously fixed
architecture-defined partitions. Every group retains rank16 at the fixed
Gram eigenvalue cutoff1e-10. Thin QR supplies an orthonormal basis; FP64
orthogonality, source-span reconstruction and FP32 basis-Gram errors are
recorded. Each condition has1,132,560 coordinates. No Joint parameters,
teacher activations or full-CE displacement enter basis construction.

The existing `train_prehead_distillation.py` runs unchanged with balanced CE
and auxiliary scale0. A process-local proxy replaces only its Adam constructor
for the two constrained arms; other modules retain the real torch module.
The wrapper records both its hash and the inherited trainer's hash, labels
the scientific condition explicitly, and hashes the completed native run.
The inherited worker still loads its existing teacher-feature cache, although
those features contribute nothing to the zero-auxiliary objective. This is
cache I/O, not new teacher inference or a teacher-supervised condition.

Full Adam moments and the expert bases remain in memory. Consequently this
is not a memory-saving or speedup claim. Final exported checkpoints have only
ordinary LEAF weights. Averaging already rounded weights is the primary
output; a separate check measures its numerical difference from averaging
the stored displacement coordinates.

## Evidence required before interpreting results

Engineering requires deterministic32-step full CE to reproduce the existing
same-worker reference tensors, histories and metrics. Both projected smokes
must pass native export, finite-gradient and unchanged-fixed-tensor checks.
Actual/proposed step norms must agree within0.2%, and weight-mean versus
coordinate-mean maximum difference must be below2e-6. These are numerical
implementation checks, not accuracy thresholds. The verifier and tolerances
were saved before smoke outcomes.

Scientific interpretation compares both constrained means with fresh full CE
and each other, reporting all task effects and deletion sensitivity. Row
beating shuffled alone supports the structural prior but does not meet the
requested Joint-gap goal. Superiority over full CE would still need replication,
a differentiated contribution and independent confirmation. Negative results
would limit this fixed optimizer/constraint recipe, not prove a capacity bound.

Generic fine-grained merging, chunking and projected optimization have prior
art. The literature ledger includes NP Merge, Expert Merging++, SubGD and
the Adam rotation study. No paper-level novelty is claimed by this design.

Frozen design:`../inputs/row_projected_training_design.json`.
Frozen runtime plan:`../inputs/row_projected_training_plan.json`.
