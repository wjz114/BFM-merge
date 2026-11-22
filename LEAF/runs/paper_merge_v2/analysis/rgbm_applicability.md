# Applicability to the current LEAF experiment

This is a local inference note, not a reproduction. Primary-paper read scope
and attribution are recorded in `../literature/review.md`, entry2026-11-16
on residual-governed repair.

The original LEAF merger learns a coefficient for each expert and parameter
group. Write its update as

\[
\Delta\theta(\alpha)=\sum_{g,d}\alpha_{gd}P_g\Delta\theta_d,
\]

where `P_g` retains only group `g`. At a fixed input and base checkpoint,
the first-order response of a differentiable scalar model output is

\[
\Delta f(\alpha)\simeq\sum_{g,d}\alpha_{gd}v_{gd},\qquad
v_{gd}=\nabla f(\theta_0)^\top P_g\Delta\theta_d.
\]

Observing only each whole expert's endpoint response provides, at best in
this approximation, the sum `sum_g v_gd`. It does not identify every
`v_gd`. For example, two groups can have responses1 and−1: the whole-expert
response cancels, whereas coefficients(1,0) give response1. Whole-expert
output mixtures therefore cannot generally certify a tensor-coefficient
merger's linearity. A groupwise Jacobian calculation or a restricted
coefficient family would be a different, explicitly specified analysis.

This algebra does not establish whether LEAF is locally linear, whether a
residual gate is useful, or whether a repair direction improves accuracy.
Those are empirical questions. The completed three-bank CE experiment
already shows useful shared-weight repair under its own recipe; it cannot
be replaced by a blanket interference narrative. Its out-of-span repair
effects also need functional tests rather than a norm-only argument.

The live four-arm prehead experiment is unchanged. First compare its fixed
means and fitting objectives. Generic residual-directed branch repair would
need differentiation from existing work and a data/compute-matched ordinary
repair comparison before it could support the requested contribution.
