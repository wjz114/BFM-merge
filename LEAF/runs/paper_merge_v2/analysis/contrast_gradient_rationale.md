# Prototype-contrast gradient diagnostic

The fixed row-space recipe has finished. Its fitting CE is higher than full
CE on every task; it does not improve validation macro BAcc. The subsequent
query-slot check enumerates all8! assignments and finds identity optimal for
every expert under both weight anchors, raw responses and centered responses.
Neither result selects another row-space or query-permutation training trial.

The next hypothesis concerns the shared cosine readout, not parameter-space
row constraints. Let z be the raw768-dimensional output, r=||z||, u=z/r,
p_c the fixed normalized text prototypes, and C the span of their centered
contrasts. For one example, v=sum_c(softmax(logits)_c-y_c)p_c lies in C.
The ordinary raw-output CE gradient is

    g = (v - u (u^T v))/r.

For the orthogonal projector P onto C,

    (I-P)g = -(u^T v)(I-P)u/r.

Thus this component is caused by normalization, although unnormalized class
logit differences depend only on Pz. On a correctly oriented binary example,
it tends to reduce feature energy outside that task's contrast direction;
on an incorrectly oriented one, it can increase that energy. Directions
outside one task's contrast space can belong to another task's contrast
space. This supplies a concrete possible coupling mechanism, not evidence
that the coupling is harmful in LEAF.

A prospective intervention would preserve the native forward computation
and project only the raw-output backward signal to Pg. No task head, routing,
expert bank or altered inference computation would be exported. It is a
modified backward rule, not the exact gradient of the unchanged CE in general.
For the final linear head, the projected aggregate gradient is P times the
original head gradient, so its inner product with the original head gradient
is nonnegative. This does not imply descent for the full network or Adam.
No norm-restoration factor or new training trial has been selected.

First reuse the archived initial raw outputs and prehead features on the
fixed fitting stream. Measure per-example normalization-component energy,
aggregate final-head component energy, and ordered cross-task directional
derivatives against each other task's ORIGINAL CE head gradient. Compare
directions after diagnostic norm matching to avoid conflating direction and
magnitude. Projected-versus-projected gradient cosines alone would not test
interference with the original objectives. Validate formulas with autograd
and replay initial logits. This stage requires no EEG reads, model inference,
training, validation data or checkpoint change.

If the cached head diagnostic motivates continuation, a separate full-network
gradient intervention is needed before attributing interference to this
component. A reduction in pairwise conflict alone is not a performance result
or a proof of useful knowledge preservation. Existing gradient surgery and
normalization work remain prior art; no novelty claim is established by a
bounded search or by this derivation.
