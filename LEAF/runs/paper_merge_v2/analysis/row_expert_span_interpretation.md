# Operator rows contain a useful part of the fixed CE repair

The completed intervention supports a narrower structural result: at equal
dimension, operator-row expert spans capture more of an existing CE update
and retain more validation benefit than shuffled coordinate groups. It does
not establish a better independently trained method, a capacity limit, or
the requested substantial improvement over full-pool CE toward Joint.

## Fixed construction

The target is the historical1600-update full-pool CE checkpoint mean minus
the original merger. All16 original experts define the basis; no Joint
parameters enter construction. The126 eligible tensors are Linear weights,
fused/separate attention projection weights, and Conv2d weights flattened
within output channels. Each output row has16 expert directions. Norms,
biases, positions and learned queries retain their old tensorwise projection.

The control permutes coordinates within each eligible tensor, applying the
same fixed permutation to all experts and the target, then uses exactly the
same group count and width. Both spaces have1,132,560 coefficients and actual
rank1,132,560. Original whole-tensor merging has6,128 coefficients. This is
an equal-dimension structural comparison, not a claim that extra coefficients
are free. The broader space has about4.4% of the active weight dimension.

FP64 Gram solves use the fixed relative cutoff1e-10. Normal equations,
Pythagorean decomposition, containment of the original tensorwise projection,
and recomposition were checked. Toy cases include rank deficiency, a zero
basis, independent SVD least-squares agreement and permutation restoration.
The plan was fixed before projection or evaluation outcomes.

| Projection space | Fraction of squared CE-update norm captured |
| --- | ---: |
| Whole parameter tensor |1.757562%|
| Operator row |31.296584%|
| Shuffled coordinate groups |4.497584%|

The corresponding projected norms differ. Equal dimension does not make this
a norm-matched comparison. These parameter-space fractions do not measure
semantic information, explain all optimization dynamics or certify accuracy.

## Native single-checkpoint results

All four predeclared projected/complement interventions were evaluated at
scale1, alongside fresh initial and full-update comparators. Every model is
an ordinary LEAF checkpoint with the same state keys and unchanged inactive
tensors. All16 previously exposed validation tasks and original512 training
rows were included; there were no official-test reads.

| Checkpoint | Validation macro BAcc | Gain over initial |
| --- | ---: | ---: |
| Initial merger |63.912610%|0|
| Full CE repair |65.234338%|+1.321728pp|
| Row projection only |64.346020%|+0.433410pp|
| Row complement only |65.032741%|+1.120131pp|
| Shuffled projection only |64.009559%|+0.096949pp|
| Shuffled complement only |65.172162%|+1.259552pp|

Row projection exceeds shuffled projection by0.336461pp:12tasks improve and
4decline. Leave-one-task-out means range0.263213–0.419498pp. Row complement
is0.139421pp below shuffled complement:5higher,10lower,1tie, with deletion
means−0.212149 to−0.074459pp. These are sensitivity ranges, not confidence
intervals; the tasks and one expert bank do not establish independent replication.

Row projection retains32.7912% of the full CE accuracy increment; its
complement retains84.7475%. These do not sum to100% because the finite
checkpoint interaction is−0.231813pp. Shuffled counterparts retain7.3350%
and95.2959%, with interaction−0.034773pp. Such endpoint interactions are not
Hessian or gradient-causality estimates. Most of the useful existing repair
remains outside even the finer row space.

The row-projection checkpoint still trails Joint by3.713183pp and full CE
by0.888317pp. It is not an improved merger. Projecting an optimized full-space
endpoint also does not establish what optimization confined to that space
could attain.

## Verification and cost

Construction completed in29.11s. Both original evaluation workers completed
in194.13s and166.10s. Cost:411630 new forward examples, zero teacher forwards,
zero new optimizer updates. The original CE training that supplies the target
remains a prerequisite. All32 native initial/full logit replays are exact
with no prediction changes; every evaluated state remained unchanged.
Allocation warnings were followed by normal completion; no restart occurred.

For non-eligible tensors, the complement subtracts the stored FP32 historical
parallel update from the full model. It differs from the historical complement
rounded from its FP64 projection by at most1.192e-7. This rounding difference
is identical for row and shuffled conditions. The numerical audit is in
`row_expert_span_rounding_audit.json`; no checkpoint was changed after evaluation.

The PNG/PDF figure is`figures/row_expert_span.{png,pdf}`; the PNG was inspected.
Authoritative numbers:`row_expert_span_summary.json` and
`row_expert_span_integration.json`. Source and input hashes are fixed in
`../inputs/row_expert_span_plan.json`.

## Research decision

This gives a concrete hypothesis for a constrained-optimization comparison:
operator-row expert directions may be a more useful prior than equally sized
arbitrary coordinate groups. A subsequent test must optimize both spaces from
the same merger, control the update/optimizer geometry and data budget, and
compare against the full-pool CE baseline. It must not infer superiority from
this post-hoc projection result or tune a coefficient rule against Joint.

Generic fine-grained coefficients are not new: NP Merge learns elementwise
interpolations, and Expert Merging++ uses flattened contiguous chunks.
An operator row is a contiguous chunk, so grouping alone is not a distinct
parameterization claim. The current evidence warrants investigating the
structural hypothesis; it does not establish differentiated paper contribution
or sufficient Joint-gap reduction. No training condition is launched by this
diagnostic. The original research goal remains active and unachieved.
