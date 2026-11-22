# Decision disagreements and the next controlled comparison

`diagnose_decision_disagreements.py` completed using only previously saved
training logits and prototypes. It replays the original204800-example
fitting multiset and the two128-row-per-task input-response subsets. The
full-stream weighted accuracies exactly reproduce the earlier raw-output
gradient audit. There is no new model inference or fitting in this audit.

On the full fitting multiset, the task-macro weighted teacher accuracy is
96.840881%, compared with81.311025% for the initial merger. Expert-correct,
merger-wrong examples occupy16.548183% of weighted mass; the reverse occupies
1.018327%. Their difference exactly equals the15.529856pp accuracy gap.
Both-correct mass is80.292699%; remaining mass consists of both-wrong cases.
These weights are the original full-pool class weights applied to the
sampled multiset and normalized within each task. They are not independent
validation BAcc or a newly estimated generalization ceiling.

Large expert-correct/merger-wrong masses occur for IV2a39.154%, ShanghaiU
31.632%, SSVEP28.983% and Upperlimb26.608%. On most binary tasks, the teacher's
normalized top-two margin on those examples is close to its geometric
bound. This supports the previously observed readout saturation; it does
not measure distance to an input-space boundary. All tasks and all margin
quintiles remain in `analysis/decision_disagreements.json`.

Ordinary KL is already focused on many of these disagreements. Averaging
within-task fractions,72.8691% of its initial loss lies on teacher-correct,
merger-wrong rows. Their macro mean KL contribution is0.045774 out of
0.054048 total. Reweighting solely because such rows exist therefore needs
more justification than an assertion that KL ignores them.

The smaller original/replacement subsets have remaining expert-minus-CE
training gaps of9.2265/12.3573pp. Replacing CE by CE+KL changes their BAcc
by-0.6882/+0.3419pp respectively. These subsets are descriptive, dependent
training data; the already-known full-validation difference remains
+0.095397pp. The audit does not support selecting one of these subsets as
the favorable result.

## Fixed optimization-budget control

The new design `inputs/duration4800_ce_design.json` repeats the exact existing
1600-step schedule three times, with one continuous Adam state. Each arm
uses4800 updates,614400 example exposures and the same117785 unique fitting
rows. Every row multiplicity triples exactly; no new training rows or
teacher targets are added. One arm starts from the initial merger and the
other from the released base. Both retain Adam1e-5, FP32, balanced CE,
dropout off and fixed BN.

The prescribed primary output averages steps2400/2880/3360/3840/4320/4800,
the same six relative locations in the second half as before. The comparison
therefore concerns the longer training procedure with its corresponding
fixed averaging window. It does not separately identify effects of update
count and averaging-window placement. Selected checkpoints are secondary.

Before crossing the original1600-update boundary, each worker must reproduce
all six existing800/960/1120/1280/1440/1600 state dictionaries bitwise and
their validation metrics exactly. The new worker changes bookkeeping and
this replay check, not the gradient/update path. It stops on a mismatch;
neither live plans nor workers may be edited to bypass a failed gate.

This is a conventional baseline motivated by remaining training errors and
the historical Joint learning curves. It is not an optimized short Joint
recipe or a new algorithm. The merger also incurs1600 coefficient-fitting
updates and48000 expert-bank updates; no end-to-end savings claim follows.
The paired reporter is prepared and must run only after both workers finish.

Boundary-sample distillation, paired crossing samples, mixup-based function
matching and synthetic-input merge repair have direct precedents recorded
in the literature ledger. Any next method needs a specific distinction and
controlled performance evidence beyond those generic constructions.
