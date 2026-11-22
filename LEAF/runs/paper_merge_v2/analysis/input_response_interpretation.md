# Completed frozen-model input-response diagnostic

All16 tasks completed on bank0:128 class-stratified original-training rows
and128 label-matched replacement-training rows per task, three shared models
and each own expert. Replacement rows are available to full-pool repair and
are not independent evaluation. The selected ADHD pools share9 rows.
No optimization, new checkpoint, validation inference or official-test read
occurred. The reporter recomputed every saved energy aggregate and all
agreement-conditioned aggregates. The standalone PNG/PDF was inspected.

The comparison below is CE+KL relative to CE. Ratios are geometric means
across16 tasks of the relative-Jacobian-mismatch ratio, using empirical
channel support. Lower is better for this diagnostic.

| Training pool | Error decreases | Geometric error ratio | Mean cosine change |
| --- | ---: | ---: | ---: |
| Original | 14/16 | 0.897355 | +0.007205 |
| Replacement | 12/16 | 0.802865 | +0.024349 |

Raw-channel ratios are0.898445/0.806125; covariance-weighted ratios are
0.897829/0.802735. Each log-ratio contrast stays negative after deleting any
single task. Those deletion checks are sensitivity analyses, not confidence
intervals. The existing exposed-validation BAcc values remain65.234338% for
CE and65.329735% for CE+KL: only+0.095397pp, with7 task improvements,8 declines
and1 tie. This diagnostic adds no new performance result or independent
replication. It shows that ordinary KL already changes input derivatives;
the magnitude of that change alone does not explain a useful BAcc gain.

## Norm, direction and denominator effects

For teacher energy A, student energy B and their inner product D, the exact
identity is E_error=A+B-2D. With r=sqrt(B/A) and cosine c, relative squared
error is1+r^2-2rc. All saved measurements satisfy both identities.

The KL-minus-CE error change decomposes into deltaB-2deltaD. The median
fraction |deltaB|/(|deltaB|+2|deltaD|) is97.0250% on original rows and97.6834%
on replacement rows. This is an algebraic decomposition: the inner-product
term itself depends on both direction and norm, so these percentages are
not a causal attribution to smoothing. Directional agreement remains low:
KL median cosine is0.05599/0.06283 across tasks. The largest mismatch ratios
are strongly affected by tiny teacher energies, e.g. support energy around
1e-10 for ADHD. Absolute and relative measurements must be shown together.

Teacher energy is also concentrated in few rows. Across tasks, the median
share carried by the highest-energy10% of rows is96.5277%/96.1693%. The
median largest-row share is66.9037%/36.7406%. Consequently128 rows can yield
unstable pooled magnitudes even though every class contrast is computed.
Per-row quantiles and concentrations are retained in the report.

## Binary cosine-head geometry

A post hoc, cache-only check covers all8 binary tasks. For normalized
feature u and prototype difference d, the margin is u^T d, bounded by
||d||. The squared tangent factor is
||P_(u-perp)d||^2/||d||^2=1-(margin/||d||)^2.
Feature-to-input Jacobians and the inverse raw-feature norm remain additional
factors, so this identity is not a complete input-derivative explanation.

Teacher median normalized margin slack is small on every binary task:
about1.2e-5 to1.3e-3 on original rows, and1.3e-5 to1.1e-3 on replacement
rows. For ADHD it is1.2059e-5/1.3473e-5. Thus the cosine readout is already
close to its geometric margin bound on typical expert training examples;
small teacher logit derivatives need not imply absence of useful internal
features. The shared models have appreciably larger slack. This check
uses saved logits and prototypes only, reports floating-point overshoots,
and introduces no saturation threshold or training choice.

## Decision

Do not claim that preserving the measured Jacobians is the missing causal
mechanism. A generic Jacobian penalty is established prior art, and the
present proxy mixes direction, magnitude, near-saturated readout geometry
and a few influential examples. No derivative-loss scale sweep, new expert
bank, candidate training or extra independent-evaluation claim follows.
The next candidate needs a specific target that survives these distinctions
and a controlled performance intervention. The paper objective remains
active and unachieved; full-pool CE remains the replicated conventional
comparator.

Primary artifacts: `input_response_summary.json`,
`input_response_geometry.json`, `binary_margin_saturation.json`.
Numerical gate and literature applicability: `input_response_v3_gate.json`,
`input_response_numerics.json`, `input_response_applicability.md`.
