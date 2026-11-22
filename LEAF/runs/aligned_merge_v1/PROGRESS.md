# Language-aligned merging research ledger

## 2026-11-14 — resume and evidence audit

The previous work made concrete progress: CPSC was implemented and 16 seed-0
experts were folded; five 1,600-update merge runs completed. No experiment
process was live at resumption. Original `runs/poc16` assets remain read-only.
The repository is `LEAF/` (the parent is not a Git repository); existing dirty
files and untracked experiment code are preserved. No applicable AGENTS.md
was found in the workspace or its ancestor directories.

Observed validation macro BAcc (training readout, **not official L2 test**):

| Run | Best step | Validation BAcc |
|---|---:|---:|
| Original Block, merge seed 0 | 1600 | 0.6259108571 |
| CPSC eta=1 Block, merge seed 0 | 1600 | 0.6263395643 |
| Original Block, merge seed 1 | 1120 | 0.6250294884 |
| Original Block, merge seed 2 | 640 | 0.6274788867 |
| Original Block, merge seed 3 | 480 | 0.6279064165 |

CPSC's paired improvement is only **0.0429 percentage points**, below the
plan's 0.5 pp pilot threshold. The merge-seed runs also change calibration
indices, so they are not pure optimizer-noise repeats and are not independent
expert-training seeds. No positive CPSC claim is supported yet.

Next implementation: reuse the existing functional merger, add explicit
calibration-index input, GroupScale and semantic/logit KD with fingerprinted
teacher caches, per-update dynamics, gradient checks and reload-logit checks.
Complete the seed-0 required matrix, then the prescribed eta=0.1/10 sensitivity
checks and matched 400-step shared-interface CE repair if pure merging remains
weak. Decisions use validation; test is reserved for fixed candidates.

Pending milestones: full P0 asset/split audit, extended P1 correctness checks,
seed-0 P2 matrix and official evaluation. Three expert seeds, external paper
baselines, P3 interface/backbone and P4 continual experiments are not run;
their plan gates remain in force.

### Implemented and checked in this continuation

- `merge_poc.py`: explicit calibration file, 33-parameter GroupScale,
  CE+SemKD and CE+logitKD, live validation/coefficient histories, per-update
  CE/KD/gradient/coefficient norms, finite-gradient and FP32 export checks.
- `aligned_support.py`: content-addressed teacher caches bound to checkpoint,
  sample IDs, EEG/label payloads, instruction/prototypes, preprocessing and
  forward code. Original embeddings are retained; CPSC targets use saved R.
- `repair_interface.py`: shared Q-Former CE repair, frozen backbone checked
  tensor-by-tensor; 400 steps, validation every 40, AdamW 1e-4/1e-3.
- `audit_aligned.py`: `inputs/seed0/audit.json` plus train/val/test/calibration
  ID files. All 18 source checkpoints load strictly, share the expected base,
  and preserve base buffers/inactive parameters. 33 groups found. Subject IDs
  are retained where HDF5 provides them; predefined splits lack those IDs.
- `run_aligned_stage.py`: reproducible named jobs, dry-run and exit ledger.
- `report_aligned.py`: validation/per-task CSVs and learning/loss figures.
- `verify_aligned_outputs.py`: extended real CPSC checks for both instructions
  and train/validation samples; implemented, full execution still pending.

Three numerical tests passed (`correctness_tests.log`): geometry/folding/CE,
degenerate complements, and stopped-teacher/differentiable-student KD.
Real MI_OpenBMI + MI_Cho2017 smoke runs passed (4 updates, not performance
evidence). All 33 coefficient groups received gradients (smallest peak group
norm 0.01424 in SemKD); maximum functional/reloaded logit discrepancy was
2.09e-7. GroupScale and Block step-zero validation both equal 0.7730960711.
Repair smoke preserved every backbone tensor and reproduced validation on
reload. The repair forward was subsequently made explicitly identical to
LEAF's normalization order under autocast; full runs use that version.

Full runs started: seed-0 Block-SemKD on GPU 0; GPU 1 queue contains GroupScale
then Base/Block/CPSC matched repairs. All use the existing seed-0 calibration
IDs. Additional queued plans: CPSC-SemKD, prescribed eta=0.1/10, CPSC Uniform,
extended invariance checks and official base/expert reevaluation. No new full
run result has been claimed before its `run.json` completion artifact exists.

Reproducible next queue command (only after the current GPU 0 run completes):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python run_aligned_stage.py --gpu 0 --jobs cpsc_block_semkd,eta01,eta10,cpsc_uniform
```

Numerical-protocol correction: full GroupScale and SemKD agree at step zero,
but their uniform BAcc differs slightly from legacy Block (0.58575 vs 0.58592)
after explicitly disabling TF32. Add fresh `block_ce` and `cpsc_block_ce`
controls using this same implementation before interpreting KD deltas.
The original repair inputs remain the prespecified legacy checkpoints; their
step-zero reevaluation records the current numerical baseline. Also, the
legacy CPSC gain hides 10/16 declining tasks and +4.63 pp on Workload; final
projection delta norms increase by 1.05–6.81×. These are observations, not a
causal explanation for the weak macro effect.

Completed P0 readout spot-check: unmodified official `c_direct_inference.py`,
MI_OpenBMI, level 2, seed 42, CPU FP32, batch 32 reproduces the stored Base
73.54% and seed-0 specialist 79.96% BAcc (CSV precision). Artifacts are in
`inputs/seed0/{base,expert}_MI_OpenBMI.{csv,json,log}`.

Extended CPSC check completed on MI_OpenBMI, 512 calibration + 512 validation
trials under each of the training and official L2 instructions. Zero prediction
changes in all four checks; maximum centered-logit error 8.40e-9, maximum
fold-vs-rotation embedding error 4.32e-8. Validation complement residual falls
from 0.8003 to 0.6498 (training instruction) and 0.7956 to 0.6448 (official L2).
This confirms prediction preservation and held-out output alignment for this
expert; it does not establish improved merging. Full report:
`inputs/seed0/invariance_MI_OpenBMI.json`. Other 15 experts' extended checks
remain pending; their original fold-time validation checks already exist.

Runtime note: `utils.seed_all()` enables TF32 and the module fixes 16 CPU
threads, overriding OMP's intent for Torch. Concurrent CPU verification caused
contention during the first two full runs; the audit processes were given lower
scheduling priority and have now completed. Include this contention in wall
time interpretation, not an algorithmic cost claim. Backend-specific precision
settings and installed versions are saved in `inputs/seed0/environment.json`.

### First full-budget new results (completed, 2026-11-14 04:14 SGT)

| Method | Validation macro BAcc | Selected update | FP32 reload max logit error | Total wall time |
|---|---:|---:|---:|---:|
| GroupScale CE | 0.6145779173 | 1440 / 1600 | 9.62e-7 | 847.9 s |
| Block SemKD, weight 1 | 0.6250746711 | 1600 / 1600 | 5.97e-7 | 955.1 s |

Both completed successfully, preserved all base buffers/inactive tensors, and
had finite nonzero gradients across all 33 groups. Reloaded validation BAcc
matches selected BAcc exactly. Peak allocated GPU memory was approximately
36.98 GB for each. SemKD teacher preparation took 11.53 s within setup; total
times include cold data loading and the documented CPU contention.

Observed dynamics: GroupScale plateaus well below legacy Block, while SemKD
has larger validation oscillations and recovers at update 1600. Calibration
CE and KD decline without a clear validation advantage yet. Do not conclude
equivalence against legacy Block until the matched precision rerun finishes.

The two automatic queues have advanced, verified by their live child processes:

- GPU 0: **Block CE running**, then CPSC CE, CPSC SemKD, eta=0.1/10 fold+merge,
  and CPSC Uniform. Driver session began with a wait on the actual first
  SemKD PID and only advanced after successful `run.json` creation.
- GPU 1: **Base interface repair running**, then Block and CPSC interface
  repairs, each with the same calibration pool and 400-update schedule.

Live/replayable commands:

```bash
python run_aligned_stage.py --gpu 0 --jobs block_ce,cpsc_block_ce,cpsc_block_semkd,eta01,eta10,cpsc_uniform
python run_aligned_stage.py --gpu 1 --jobs group_scale,repair_base,repair_block,repair_cpsc
python report_aligned.py
```

Do not launch these again while their current child processes are live. The
runner skips completed outputs but refuses incomplete directories. Process
state and `queue_events.jsonl` should be checked on continuation. Summary CSVs,
per-task changes and plots are in `seed0/analysis/`; per-update raw dynamics
remain in each run directory. The full P2 matrix, official tests of new methods,
remaining 15 extended invariance checks and decision gate are still pending.
No newly promising method has been established yet, so the goal stays active.

## 2026-11-14 — continuation: matched controls and mechanism analysis

Previous goal turn classification: **progress** (new implementations, completed
full-budget GroupScale/SemKD runs, real correctness evidence and launched
queues). Revalidated both queues and their live children before continuing;
no run was restarted.

Added `analyze_merge_geometry.py` and
`seed0/analysis/parameter_geometry.json`: actual active-parameter displacement
norms and directions relative to the original uniform task vector, per group.
For the legacy selected checkpoints, Block backbone/interface displacement is
18.008/13.689 versus CPSC 18.121/11.337. The output-group displacement is
12.541 versus 9.753. Thus the individual aligned experts' enlarged final-layer
task vectors do **not** imply a larger final merged output-group displacement.
GroupScale's backbone displacement is only 6.325 (interface 10.389). These
measurements constrain explanations; they do not establish a causal mechanism.

Extended `report_aligned.py` with a per-task change heatmap and validation
trajectory ranges. Within-run trajectory ranges are explicitly not seed
uncertainty. Archived 21 source/config files plus the plan and package versions
in `inputs/source_snapshot_20261114`; archived merge/support/model hashes
exactly match both completed full runs.

Started the full 16-expert, two-instruction, calibration/validation invariance
check on CPU at priority 19 with **one Torch thread**, avoiding the earlier
contention. Live process at launch: PID 397918; output target
`inputs/seed0/invariance_all_eta1.json`, log of the same stem. This check has
not yet completed and must be polled rather than restarted.

Completed **Base + interface CE repair**: validation BAcc **0.6251455077**,
selected step **320/400**, versus step-zero 0.573243. Frozen backbone equality
and reload validation checks passed; total wall time 401.1 s. No candidate
test score was used. Block repair has now started from its prespecified legacy
checkpoint, followed by CPSC repair. The base repair result is a necessary
control: ordinary shared-interface training alone can reach approximately the
legacy Block level, so an advantage from merging must be demonstrated against
this matched repair baseline.

Matched numerical-protocol **Block CE completed** at validation BAcc
**0.6272082324** (`seed0/block_ce/run.json`), selected update 1600.
The report now uses this run as its primary Block reference. Semantic KD is
approximately **−0.213 pp** against the matching control; the positive-increment
claim is unsupported. CPSC CE is now running with the same precision settings.

The one-thread full CPU invariance pass was deliberately terminated (SIGTERM,
exit 143), after observed throughput of several minutes per expert. This was
a resource scheduling decision for a live verification process, not a training
failure or an observation-timeout restart. Its log is retained. The identical
full check is queued behind the verified live GPU-1 repair driver; it will write
`inputs/seed0/invariance_all_eta1_gpu.json` and `.log`. The already completed
MI_OpenBMI CPU check remains valid. No training process was interrupted.

### A provisional positive repair result; confirmation prioritized

The default full-interface controls completed:

| Initialization + 400-step repair | Selected update | Validation BAcc |
|---|---:|---:|
| Base | 320 | 0.6251455077 |
| Legacy Block | **0** | 0.6257321754 |
| Legacy CPSC Block | 400 | **0.6313324119** |

Block repair selected its unchanged initialization: training all interface
parameters did not help that run. CPSC + repair improves **+0.5600 pp** over
matched Block repair, with worst task change **−1.8182 pp**, satisfying the
plan's practical magnitude/stability thresholds **within the repair family**.
It improves +0.6187 pp over Base repair, but EMO_FACED is −4.4753 pp against
that different control. Against the newly rerun pure Block CE it gains only
+0.4124 pp, below the pure-merge pilot threshold. This is a promising repair
interaction on one expert seed, not evidence that pure CPSC merging works.

Indeed, matched strict-FP32-composition CPSC CE finished at **0.6252557677**,
versus Block CE **0.6272082324** (−0.1952 pp). All default export checks passed.
To check whether the repair signal depends on the legacy numeric trajectory,
started **repair_block_fp32** then **repair_cpsc_fp32**, using the fresh merge
checkpoints and otherwise exactly the same 400-step repair protocol. Their
results are pending. CPSC SemKD and prescribed eta checks remain on GPU 0.

Full real CPSC invariance verification now **passed on all 16 experts**, 512
calibration + 512 validation examples, each under training and official L2
instructions: **64 checks, zero prediction changes**. Largest centered-logit
error 5.99e-8; softmax error 1.25e-8; CE error 4.67e-10; margin error 7.65e-8;
fold-vs-rotation output error 1.68e-7. Residual alignment improves in 58/64
split/instruction checks, so generalization of the fitted alignment is not
universal. Results and full parameter-angle matrices are saved in
`inputs/seed0/invariance_all_eta1_gpu.json` (76.15 seconds).

Prepared, **not run as an experiment**: `projection_repair.py` supports a
zero-initialized update to the existing shared final linear layer, restricted
to the union of task-class contrast spaces, with full-output and random-rank
controls. The actual union rank is **22**, giving **45,078** update parameters
versus 1,573,632 for a full-output update. Five numerical tests passed, including
zero-init equivalence, subspace restriction, ordinary-checkpoint export, and
excluding between-task centroids from the contrast basis. This alternative
was motivated by the negative full-Block repair trajectory. It is currently
held at preparation stage while the positive CPSC-repair signal is checked;
it has no empirical performance claim and is not integrated into the trainer.

Latest live observation before yielding: `repair_block_fp32` has reached
validation step 360, with a provisional best **0.6297249998 at step 240**;
`repair_cpsc_fp32` is still queued behind it. This higher fresh Block-repair
control reinforces the need for the matching fresh CPSC-repair run before
claiming robustness of the initial +0.56 pp result. `cpsc_block_semkd` reached
validation step 1280, with current best 0.6208542497 at step 960; it is still
running to the fixed 1600-update budget. No new-method official test or
three-expert-seed confirmation has run yet.

Current live drivers: GPU-0 stage session continues CPSC SemKD → eta=0.1 →
eta=10 → CPSC Uniform. GPU-1 stage session runs
`python run_aligned_stage.py --gpu 1 --jobs repair_block_fp32,repair_cpsc_fp32`.
The full invariance GPU pass is complete (exit 0). On continuation, inspect
these actual processes and completion artifacts before launching anything.
Regenerate summaries with `python report_aligned.py`; it now exports
`alignment_checks.csv` and includes the six residual-alignment regressions.

## 2026-11-14 — confirmation and evaluation preparation

Previous goal turn: **progress** (completed full invariance audit, matched CE
and repair results, additional analysis artifacts and live confirmation runs).
Both live drivers were revalidated before continuing; no training restarted.

The required coefficient-only seed-0 matrix is complete: GroupScale 0.614578,
Block CE 0.627208, Block SemKD 0.625075, CPSC CE 0.625256, CPSC SemKD
**0.6231081982** (selected step 1440/1600). CPSC SemKD is −0.4100 pp versus
Block CE and −0.1966 pp versus ordinary SemKD. Thus neither pure CPSC merging
nor its default KD extension has demonstrated an increment. The prescribed
eta=0.1/10 sensitivity runs continue without test feedback.

`repair_block_fp32` completed at **0.6297249998**, selected step **240/400**.
The matching `repair_cpsc_fp32` is running. These use the freshly rerun merge
checkpoints; the earlier +0.56 pp legacy-input repair comparison is still a
provisional observation, not a robustness result.

Added `evaluate_aligned.py`, a thin runner for the unmodified official
`c_direct_inference.py --level 2 --seed 42 --bs 256`. It freezes checkpoint
hashes and validation-selection receipts before test inference and keeps
official tests separate from validation summaries. A real-input dry run
passed (`evaluation_dry_run.txt`); no new test evaluation has run yet.

### Confirmation failed; next repair hypothesis fixed before test

The fresh-checkpoint repair pair completed: Block repair **0.6297249998**
(step 240), CPSC repair **0.627449** (step 400; exact value in its run.json).
The earlier +0.56 pp CPSC-repair advantage did **not** repeat; the matched
fresh comparison is approximately −0.228 pp. Do not promote the earlier
single-seed result as a demonstrated method increment. Both runs passed
frozen-backbone and reload checks. Test data still has not informed choices.

Integrated the prepared projection experiment into `repair_interface.py`
after the active full-interface runs finished, preserving their archived
source hashes. The added modes are `output`, `contrast`, and `random`:

- Freeze the backbone and all interface layers except the existing final
  shared projection; update it as W_init + Q A (and the corresponding bias).
- Full-output control has 1,573,632 update parameters; contrast and random
  controls have 45,078 (rank 22 on the full task manifest).
- Contrast Q is the union of within-task prototype differences. Random Q is
  a fixed seed-0 orthonormal subspace of the same rank. No task-specific head.
- Fixed 400 updates, AdamW 1e-4/1e-3, batch 128, validation every 40 including
  step 0, original calibration IDs. Weight decay acts on zero-initialized
  update coordinates for all three projection modes.
- All projection controls use FP32 frozen preprojection features and FP32
  training/readout. This differs from BF16 full-interface training; the
  full-output and random controls isolate the restriction within the same
  numerical protocol. Frozen features are generated afresh in memory and
  preserve sample order and the subsequent training RNG state.
- Export folds the update into an ordinary LEAF checkpoint. Checks cover
  frozen parameters, cached-versus-full-forward logits, and selected reload.

Predefined run set, before any new-method test scores: Block output / contrast
/ random, Base contrast, CPSC contrast. The primary candidate must improve
against matched full-output and random controls as well as the unrepaired
Block; Base contrast tests whether expert merging adds value. No learning-rate
or rank sweep is planned. Six numerical tests passed; real-data smoke checks
are running before the full projection comparisons are launched. Projection
performance remains **not_run** at this point.

### Projection controls launched after smoke checks

Both real-data smoke checks passed. Restricted projection repair preserved all
frozen tensors, exported an ordinary checkpoint, and reproduced full-model
logits within 9.84e-7 on the two smoke tasks. The refactored default full-interface
path also passed its frozen-backbone and reload checks. The two-task contrast
rank is 1; the full task union is rank 22 as recorded above. Smoke outcomes are
engineering checks, not method-performance evidence.

Started the prescribed GPU-1 queue: `proj_block_output`, `proj_block_contrast`,
`proj_block_random`, `proj_base_contrast`, `proj_cpsc_contrast`. The source is
archived under `inputs/source_snapshot_20261114_projection` with SHA-256
manifest, retaining the earlier source snapshot. GPU 0 continues the eta
sensitivity queue. No official test score has yet informed candidate choice.

The eta=0.1 sensitivity run completed at **0.6226850408**, selected step
1600/1600 (−0.4523 pp vs strict Block). GPU 0 proceeds to the prescribed
eta=10 run. The first projection control, Block + full-output repair,
completed at **0.628269** (step 120/400), only about +0.106 pp versus Block.
Its calibration/validation CE keeps decreasing while macro BAcc peaks early;
validation CE fell from 0.942428 to 0.934951, but final BAcc was 0.627634.
This control alone does not meet the pilot threshold. Export/full-forward
checks passed on all 16 tasks (max logit error 2.94e-6). A transient CUDA
allocator warning during FP32 feature extraction recovered automatically;
the run completed normally without changing its configuration.

Existing POC official source references are recorded separately in
`seed0/analysis/poc_official_references.json`; they were already known in
the plan and have not been used to select any new checkpoint. These values
are averages of the original four-decimal official CSV task rows.

The matched Block projection trio is now complete (same frozen checkpoint,
calibration IDs, task/batch schedule, FP32 arithmetic, 400-update budget):

- `proj_block_output`: 0.6282687022, selected step 120, 1,573,632 update parameters.
- `proj_block_contrast`: 0.6283382368, selected step 280, 45,078 update parameters.
- `proj_block_random`: 0.6274058742, selected step 400, 45,078 update parameters.

Contrast repair gains only +0.1130 pp over Block, +0.0070 pp over full-output,
and +0.0932 pp over random-rank repair. Its worst task change versus Block
is −0.1914 pp (MI_BCIC_Upperlimb), compared with −0.7172 pp for full-output.
The 35-fold parameter reduction is real, but these small accuracy differences
do not satisfy the pilot accuracy gate. Base/CPSC contrast controls continue.

Next bounded diagnostic if those controls do not reveal a stronger signal:
run the already implemented, plan-listed `block_logitkd` at its preset weight
1 and the same 1600 updates. This tests whether task probabilities are more
useful teachers than full embeddings, whose non-discriminative directions
can disagree across experts. No temperature, LR, or KD-weight search; no
new official test scores have been observed at this decision.

### Projection set completed; fixed logits KD running

- `proj_block_output`: BAcc **0.6282687022**, step 120; frozen/export checks passed, maximum cached/full logit error 2.94e-06.
- `proj_block_contrast`: BAcc **0.6283382368**, step 280; frozen/export checks passed, maximum cached/full logit error 2.8e-06.
- `proj_block_random`: BAcc **0.6274058742**, step 400; frozen/export checks passed, maximum cached/full logit error 2.77e-06.
- `proj_base_contrast`: BAcc **0.5800305474**, step 360; frozen/export checks passed, maximum cached/full logit error 2.18e-06.
- `proj_cpsc_contrast`: BAcc **0.6264567897**, step 240; frozen/export checks passed, maximum cached/full logit error 2.03e-06.

Base contrast repair reaches 58.00%, well below the merged initialization.
However, contrast repair does not yield a practical gain over unrepaired
Block at this budget. CPSC contrast repair is also worse than Block contrast
repair; this second repair restriction does not rescue CPSC's contribution.
No projection variant meets the +0.5 pp validation accuracy gate.

Started `python run_aligned_stage.py --gpu 1 --jobs block_logitkd` with
unchanged preset hyperparameters and exact original calibration indices.
GPU 0 continues eta=10 followed by CPSC Uniform. Source modules used by
active trainers have not been edited. Official evaluation remains pending.

Additional analysis artifacts: `matched_projection_comparison.csv` records
all prespecified matched differences; `training_dynamics_summary.json` stores
first/last observed training-loss windows and validation trajectories.
Gradient norms from distinct parameterizations are not comparable directly.

### Fixed official evaluation queue and component diagnostic

Prepared `diagnose_components.py` for the plan's validation-only 2×2 component
analysis: B0/AM and BM/A0 for ordinary Block and CPSC, reusing previously
measured B0/A0 and BM/AM controls. No fitting, new checkpoint selection, or
test input. CLI/compilation checks passed; real execution is queued after the
current GPU-0 driver finishes. Non-additivity will be reported as interaction,
not as proof of semantic drift.

The same queue then evaluates the fixed coefficient-method set with the
unmodified official L2/seed-42 evaluator: Block CE, GroupScale, Block SemKD,
CPSC CE, CPSC SemKD, eta0.1, eta10, CPSC Uniform. All checkpoint selection
remains validation-only. The queue waits for PID 387256 to exit; it does not
interrupt or restart training. The command, logs, and checkpoint receipts
will make each evaluation auditable. Later analysis/evaluation sources are
archived in `inputs/analysis_snapshot_20261114`; earlier snapshots are intact.
At scheduling time, **no new-method official test result has been read**.

## 2026-11-14 — fixed-test phase and component evidence

Previous goal turn: **progress**. It completed the five preset projection
controls and their export checks, rejected a practical projection/CPSC gain
at the current budget, launched fixed logit KD, and prepared the component
diagnostic and official evaluation queue. Both GPU training processes and
the waiting evaluation queue were revalidated as live on continuation.

Scheduled GPU-1 official evaluations after its logit-KD driver (PID 461980)
exits: `block_logitkd`, `repair_base`, `repair_block_fp32`,
`repair_cpsc_fp32`, and all five `proj_*` runs. This covers the preset
repair comparisons, including negative controls, with no selection from
test results. Legacy-initialization repair runs remain in the validation
record but are not promoted as the fresh matched comparison. The two GPU
evaluation queues use disjoint method outputs and the same unmodified
official L2/seed-42/batch-256 protocol.

### Bounded extension: matched full-model repair (fixed before test review)

The projection controls did not clear the validation gate, and the strongest
fresh full-interface repair improves Block by only +0.2517 pp. Existing Joint
validation is 0.680497 versus Block 0.627208; this indicates remaining model
capacity, not the cause of the gap or a matched-budget result. A bounded next
hypothesis is that repair must allow the merged backbone and interface to
adapt jointly. This extends the plan's default backbone-frozen repair and
will be reported separately from coefficient-only merging.

Prespecified three controls: `repair_base_full`, `repair_block_full`,
`repair_cpsc_full`, initialized from Base and the fresh strict Block/CPSC
checkpoints. All use the identical original calibration indices, seed 0,
400 total updates, batch 128, AdamW lr=1e-4/weight_decay=1e-3, eval mode
(dropout off, BN statistics fixed), BF16 forward/FP32 CE, validation step 0
and every 40. Train all classification-active parameters; inactive tensors
and all buffers stay unchanged. No LR, duration, or scope sweep is planned.

Evidence sought: a practical gain over unrepaired Block, an advantage over
matched Base full-model repair (to establish value from merging), and the
separate CPSC-vs-Block initialization comparison. Merely improving after
extra parameter training does not establish a CPSC contribution.

Extended the existing repair trainer with `--mode full`; no second trainer
or deployment module. Added backbone/interface gradient evidence, frozen
tensor checks, and retained ordinary checkpoint reload checks. Sources are
archived in `inputs/source_snapshot_20261114_full_repair`. A four-update,
two-task smoke run gates the full queue.

Reordered only the waiting GPU-1 queue: terminated waiting shell PID 464733
(SIGTERM; it had only a `sleep 5` child), leaving logit-KD training untouched.
The replacement waits for logit KD, runs the smoke and three full-repair
controls, then evaluates all fixed repair/logit/projection checkpoints.
GPU-0's existing component/official queue remains intact. No new official
test result had been read when this extension and all settings were fixed.

Eta=10 completed at **0.6262040318**, selected step 1600/1600, or
−0.1004 pp versus Block. Together with eta=1 (−0.1952 pp) and eta=0.1
(−0.4523 pp), the prescribed CPSC strength checks all remain below Block.
The negative main result is not rescued by the preset sensitivity range.
The GPU-0 driver has advanced to the non-trained CPSC Uniform reference.

CPSC Uniform completed at **0.5855907590**, with zero optimization updates.
All jobs in the original GPU-0 coefficient queue completed successfully.

The waiting shell sessions 16605 and 16717 both returned terminal exit 143
before starting their dependent commands. Process inspection confirmed both
wrappers absent, no diagnostic/evaluation outputs created, GPU 0 free, and
the logit-KD trainer still live on GPU 1. The cause of these two wrapper
exits is not established; do not conflate them with the earlier deliberate
termination of the replaced GPU-1 waiting shell. No training was restarted.
Launched the not-yet-started component diagnostic directly on GPU 0; remaining
stages will now be launched explicitly after verified predecessor completion.
The fixed methods and hyperparameters are unchanged.

### Component swaps and logits KD completed

- `block_ce`: B0/AM=0.5905168752, BM/A0=0.6231521844, BM/AM=0.6272082324; interaction=-1.3218 pp.
- `cpsc_block_ce`: B0/AM=0.5921154298, BM/A0=0.6207794842, BM/AM=0.6252557677; interaction=-1.4397 pp.

Logit KD completed at **0.6267716351**, selected step **800**; ordinary Block remains higher. Export/reload checks passed.

The Block merged backbone retains most of the full merge's validation score
when paired with the Base interface. This limits the evidence for a large
benefit from output-interface alignment; it is not a causal proof about
semantic drift. The full component results and per-task metrics are in
`seed0/analysis/component_swaps.json`.

After verified completion of both predecessor jobs, started the fixed GPU-0
official coefficient evaluation queue directly. Also started the GPU-1
full-model repair smoke directly. The full-model repair configuration was
already fixed before this first new-method official test launch.

The full-model repair smoke passed: finite nonzero gradients reached both
backbone and interface, all frozen/inactive tensors and buffers were unchanged,
and the selected ordinary checkpoint reloaded with identical validation
BAcc. The four-update smoke selected step 0 (accuracy declined during the
four updates); this is an engineering pass, not performance evidence, and
did not trigger hyperparameter changes. Source hash matched the archived
trainer. Launched the three fixed 400-update full-model controls on GPU 1.
GPU 0 is running official inference; the first evaluator completed normally
at the fixed batch size 256.

The eight-method coefficient official evaluation queue completed successfully
(exit 0). All evaluations use the original official L2/seed-42/batch-256
evaluator, full 16-source manifest, and pre-recorded checkpoint receipts.
`seed0/official_l2/summary.csv` and `per_dataset.csv` contain the results.
The first new-test inspection is timestamped in
`inputs/first_new_test_review.json`; full-model repair settings predate it.

Used the now-free GPU 0 to start the already-prespecified `repair_cpsc_full`
control while GPU 1 runs Base full repair then Block full repair. The original
GPU-1 stage driver will reuse the completed CPSC output when it reaches that
job; it never overwrites an incomplete run. This changes scheduling only.
All full-repair methods, data, schedules, and source hashes remain fixed.

### Official coefficient result and continuation checkpoint

Fresh Block CE scores **0.572700** official source macro BAcc. CPSC CE
scores **0.569250** (−0.3450 pp; 12/16 tasks down), CPSC SemKD 0.56543125,
Block SemKD 0.56974375, GroupScale 0.54990625, eta0.1 0.57011875,
eta10 0.56853750, and CPSC Uniform 0.52969375. Thus the official test
results agree with the negative validation finding for the fixed coefficient
methods. These are averages of four-decimal official task rows. Worst-task
changes and decline counts are in `analysis/official_coefficient_comparisons.json`.

The goal remains active: no new approach has yet passed the pilot gate and
replication requirement. The bounded full-model repair family is running,
with all settings fixed before the first new test review. Confirmed live
at this checkpoint: GPU-1 stage PID 479497 / child 479501 (`repair_base_full`,
then `repair_block_full`, then reuse completed CPSC), session 10846; GPU-0
stage PID 482798 / child 482799 (`repair_cpsc_full`), session 80544.
Do not edit repair_interface.py while these source-hashed runs are active.
The original coefficient and component/evaluation jobs are complete.
Official tests for logit KD, the matched interface/projection controls, and
full-model repair are still pending; launch them directly after GPUs free.
The previous waiting-shell wrappers are terminal and must not be mistaken
for active evaluation queues. Regenerate the report after completed runs.

## 2026-11-14 — full-model repair continuation

Previous goal turn: **progress**. It completed all prescribed coefficient
sensitivity runs and eight official tests, measured component swaps, and
implemented/smoke-checked the bounded matched full-model repair extension.
On continuation both full-repair stage drivers and their training children
were confirmed live; no run was restarted. Both backbone and interface
gradients are logged per update for the new family.

### Functional alignment reference and bounded adaptation preparation

Base full-model repair completed at **0.603052** (selected step 280/400),
below Base interface-only repair (0.625146). Its final validation BAcc was
0.584969. The two merged-model full controls continue; neither is complete
at this preparation point. No learning-rate changes were made to these runs.

Read the plan's [Expert Merging paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/24dbe8babeca9f372712bbf19dd3584e-Abstract-Conference.html) and
[official code](https://github.com/Littleor/ExpertMerging/tree/0670bb957d8a353c74dbb73d61b2e99d43112a39).
Pinned commit `0670bb957d8a353c74dbb73d61b2e99d43112a39`; six inspected
reference files are archived with exact Git blob identities and SHA-256 in
`inputs/external/ExpertMerging_0670bb95/manifest.json`. Its functional
objective includes intermediate-state matching and coefficient anchoring.
The released code and paper differ in coefficient constraints, regularizer
normalization, and optimizer defaults; these differences must stay explicit.
Also consulted the plan's [RegMean++ reference](https://arxiv.org/html/2508.03121v3);
no RegMean adaptation has been implemented or run.

Prepared `expert_alignment.py`: full hidden-state hooks for all 28 LEAF
transformer layers, content-addressed original-expert caches bound to exact
EEG/IDs/readouts/source hashes, and detached-teacher KL + hidden-MSE +
coefficient-anchor terms. Eight numerical tests passed, including unchanged
forward outputs, gradient flow into student layers, teacher detachment,
identical-state zero loss, and hook cleanup. A real eight-example CPU cache
smoke is running separately; no active trainer source has been changed.

`inputs/expert_alignment_plan.json` fixes two adapted runs for the next
available coefficient experiment: `expert_merging_adapted` (no hard-label
fitting loss) and `block_hidden_ce` (same objective plus hard CE). Both retain
33 groups, original experts/calibration, 1600 updates, batch128, Adam0.01,
validation every160 including step0; hidden weight1, KL weight1, anchor0.1.
Signed coefficients, supervised validation selection, EEG readout, budget,
and absence of importance-guided chunking are explicit deviations. These
are **adapted** methods, not a claim to reproduce Expert Merging++. All
future choices remain validation-based, and earlier test exposure is recorded.
Integration is pending until the current source-hashed full-repair runs end.

The real hidden-cache smoke passed on eight MI_Cho2017 calibration trials:
all 28 cached layers and final embeddings exactly matched direct FP32
forwards; cache reuse was exact and training RNG state was unchanged.
Evidence: `inputs/seed0/hidden_cache_smoke.json` (2.37 seconds on one CPU
thread). This is cache correctness evidence, not a method-performance run.

CPSC full-model repair completed with **step 0 selected**, BAcc
**0.6252557677**. No trained checkpoint beat its initialization. Frozen
tensor/buffer and reload checks passed; both components received nonzero
gradients. Block full-model repair is still running. With GPU 0 freed,
started official evaluation of logit KD, all existing matched interface/
projection controls, and the now-completed Base/CPSC full-model controls.
No official score is used to revise the prepared hidden-alignment settings.

### Full-model repair rejected; functional alignment integrated

The matched Block full-model control completed at its unchanged step-0
checkpoint (**0.6272082324**). Thus neither merged initialization benefited
from joint updates at this preset budget/rate; Base full repair also lost
to its frozen-backbone counterpart. All frozen-tensor, finite-gradient, and
reload checks passed. The separate official Block-full test has completed.
This negative family is retained, with no learning-rate sweep.

Only after every source-hashed full-repair trainer finished, extended the
existing `merge_poc.py` with `expert_adapted` and `hidden_ce` losses. It
reuses composition, scheduling, validation selection, export, and all
existing calibration IDs. Teacher cache records now also include the full
resolved model configuration (in addition to exact inputs/readouts/hashes).
All 28 layer states are retained; no pooling or token subsampling.

The fitting-only variant excludes hard-label CE from its objective, while
logging CE diagnostically. Both variants still use stratified calibration
indices and labeled validation selection; this is not a fully label-free
experimental pipeline. The first-step coefficient gradient norms are
recorded separately, when the uniform-anchor gradient is exactly zero.
Per-layer MSE, KL, anchor, CE, and coefficient trajectories are logged.

Sources are archived in `inputs/source_snapshot_20261114_hidden_alignment`.
Two real-data, four-update GPU smokes are now running before the fixed full
1600-update runs. No hidden-alignment validation/test result has yet informed
configuration. Earlier official test exposure remains explicitly recorded.

### Hidden-alignment smoke passed; matched official controls complete

Both four-update, two-task real GPU smokes completed successfully. All 33
coefficient groups had finite, nonzero first-step data gradients (the
uniform-anchor gradient is zero at initialization). All 28 hidden layers
were captured. Folded-checkpoint logits agree within 2.91e-7 for teacher
matching and 4.18e-7 for the CE variant; all eight unit tests passed.
`inputs/seed0/hidden_alignment_smoke_review.json` records exact hashes.
These are engineering checks, not evidence of method performance.

Launched the pre-specified `expert_merging_adapted` full 1600-update run on
GPU 1. The CE variant will use the same full teacher cache after setup.
The original pre-run `expert_alignment_plan.json` is preserved unchanged;
its preparation status describes when it was written. No weights or
training settings were changed after the smokes or official evaluations.

All 20 currently scheduled official evaluations finished. Selected results
(source macro BAcc; seed42, batch256, 16 source tasks):

| Method | Official BAcc | Difference from Block (pp) |
|---|---:|---:|
| Block CE | 0.572700 | 0.000 |
| Block logit KD | 0.567331 | -0.537 |
| Base interface repair | 0.564969 | -0.773 |
| Block interface repair | 0.568213 | -0.449 |
| CPSC interface repair | 0.564625 | -0.808 |
| Block full-output projection repair | 0.572169 | -0.053 |
| Block contrast projection repair | 0.573106 | +0.041 |
| Block rank-matched random projection repair | 0.573025 | +0.032 |
| CPSC contrast projection repair | 0.570938 | -0.176 |
| Base full-model repair | 0.552975 | -1.973 |
| Block full-model repair | 0.572700 | 0.000 |
| CPSC full-model repair | 0.569250 | -0.345 |

The contrast/random projection difference is only +0.0081 pp. Neither
projection efficiency nor these tiny score changes establishes a useful
accuracy improvement or a specific semantic-subspace effect. Full-model
repair selected the unchanged step-0 checkpoint for both merges. Base full
repair selected step280 (validation 0.6030523561), below Base interface
repair. No further full-repair learning-rate search is planned.

`analysis/report.md` and `official_matched_comparisons.csv` now include
matched official deltas and task regressions. Test scores do not change
the fixed hidden-alignment comparison. Pre-existing softmax POC controls
were also inspected: source-only official macros are 0.54224375 (global)
and 0.55270000 (block), versus legacy signed Block 0.567975. The CSV MEAN
row is excluded; these are previous numerical settings, not new controls.

### RegMean/RegMean++ fixed adaptations prepared

Pinned the plan-listed RegMean++ official repository at
`a9fdd97629707062a5872653dd9b34f7e25c7243`. Six reference files, including
its MIT license and default configuration, are archived with verified Git
blob hashes under `inputs/external/RegMean_plusplus_a9fdd976`.
Reference: https://arxiv.org/html/2508.03121v3 and
https://github.com/nthehai01/RegMean-plusplus .

The implementation compares expert-prefix statistics (RegMean) with
merged-prefix statistics (RegMean++), solving each layer before advancing
the merged representations. It covers all 184 attention/MLP linear maps in
LEAF's 28 transformer layers, including packed Q/K/V and attention output
projections. Every other active parameter is averaged, with buffers and
inactive heads fixed at the common base. No labels enter the solve.

`inputs/regmean_plan.json` fixes the two full EEG adaptations before any
full result: the official CLIP default alpha=0.95, 512 existing calibration
examples per task, batch32, no hyperparameter search, no gradient updates.
Final checkpoints are fixed; validation at initialization and completion
is descriptive, not a search over partially merged layers. This differs
from the previous 528-coefficient optimization and remains separately
identified. A CPU FP64 solve replaces explicit inversion; exactly
unobserved columns retain their arithmetic mean.

Three mathematical/forward tests passed. Both real two-task CPU smokes
completed all 28 layers and ordinary checkpoint export/reload. Their tiny
stratified subsets are engineering checks only. The first smoke stopped
before fitting because naive validation truncation omitted a class;
stratified smoke subsampling fixed that specific issue, and the failed log
is retained. Full 512-example calibration arrays remain unchanged.
Evidence: `inputs/seed0/regmean_smoke_review.json`; frozen sources:
`inputs/source_snapshot_20261114_regmean`. Full runs wait for the current
hidden-alignment jobs to finish and release their GPUs.

### Hidden-alignment full runs complete: pilot rejected

Both pre-specified 1600-update runs completed, with all 33 first-step
coefficient gradients finite/nonzero, matching dataset update order,
exactly matching teacher-cache fingerprints and original calibration IDs.
Final ordinary checkpoint logits agree with functional composition within
7.46e-7. Source hashes match the archived implementation.

| Method | Selected step | Validation macro BAcc | Difference from Block (pp) |
|---|---:|---:|---:|
| Block CE | 1600 | 0.6272082324 | 0.000 |
| Expert Merging adapted | 160 | 0.6002108740 | -2.700 |
| Hidden alignment + CE | 320 | 0.6026742191 | -2.453 |

Neither method passes the pilot gate. Teacher matching's mean hidden MSE
falls from 19.16985 in the first 160 updates to 18.18196 in the last 160,
while diagnostic CE is essentially unchanged (0.87921 to 0.87908).
Backbone branches account for approximately 96% of the raw hidden MSE.
This fixed objective has not converted intermediate matching into useful
classification retention. No loss-weight search is planned in this family.

Both full runs took approximately 631 seconds, including setup and export.
Detailed first/last-window task/layer losses and the exact gate calculations
are in `analysis/hidden_alignment_summary.json`,
`hidden_alignment_task_losses.csv`, and `hidden_alignment_layer_losses.csv`.
The report/plots have been refreshed. No official hidden-method test scores
have yet been observed.

RegMean++ is now running on GPU 1 and its matched RegMean control on GPU 0,
using the fixed settings in `inputs/regmean_plan.json`. These are new
closed-form merging experiments, with no changes to the completed hidden
objectives. Their full results remain pending.

### Fixed RegMean comparison complete: pilot rejected

Both full closed-form methods finished successfully. RegMean validation
BAcc is 0.57421603 and RegMean++ is 0.56566626; both are below their uniform
initialization and the 0.62720823 Block control. Merged-prefix statistics
did not improve this fixed EEG adaptation. All 184 map solves, attention
reconstruction checks, fixed-state checks, exact calibration-ID checks,
and ordinary checkpoint reloads passed. Exact diagnostics and timings are
in `inputs/seed0/regmean_full_review.json`. No alpha sweep is planned.

Official evaluation of the fixed hidden-alignment and RegMean checkpoints
is underway. Began source review of the plan-required TIES baseline at
`prateeky2806/ties-merging` commit
`44e7891fc84f3de7e4caa52664cd864ca3715e91`, with four verified source/license
files archived under `inputs/external/TIES_44e7891f`. Its pruning and sign
resolution address parameter interference directly; the current functional
alignment objectives have not passed the pilot.

### TIES bounded comparison integrated and smoke-checked

`inputs/ties_plan.json` fixes density0.2, mass-sign election, disjoint mean,
scale1, and two methods: fixed TIES and TIES vectors followed by the original
1600-update Block-CE schedule. There is no density or scale sweep. The
reference's kth-value threshold and global-majority exact-zero sign rules
are retained. Thresholds span all active parameters per expert. Exclusion
of base-fixed buffers/inactive heads, fixed scale instead of the reference
scale search, and the learned extension are explicit EEG adaptations.

The transformed expert vectors are multiplied by K/contributor-count per
coordinate, so the existing uniform 1/K coefficients exactly reproduce
the fixed TIES disjoint mean at initialization. This keeps the existing
composition, training schedule, readout, and export path. It changes the
available task-vector directions; it does not preserve specialist logits.

All 14 integrated tests passed, including new threshold/sign/tie cases,
zero vectors, tensor partitioning, and coefficient gradients. A four-update
real two-task GPU smoke passed, with nonzero gradients in all 33 groups and
functional/export agreement. Frozen sources are in
`inputs/source_snapshot_20261114_ties`. The RegMean trainers had completed
and their source hashes were checked before changing the shared merger.

All 24 official evaluations completed. The new fixed-checkpoint source
macros are 0.540375 (Expert Merging adapted), 0.53719375 (hidden+CE),
0.5317125 (RegMean), and 0.530175 (RegMean++), versus Block0.572700.
The TIES settings were fixed before these new official results were read.
No current approach has passed the pilot gate.

### TIES fixed result and coefficient-tying diagnostic

Fixed scale-1 TIES completed at validation BAcc 0.490561, below uniform and
Block. Its learned 1600-update counterpart is still running; step320 has
recovered to 0.6116. No scale/density setting has been changed.

A read-only gradient diagnostic reconstructed the selected Block checkpoint
exactly using 6,128 per-tensor coefficients, then differentiated macro CE
on the original calibration pools (all 512/task, FP32, batch32). Projecting
this gradient onto the original 33-group tied subspace leaves 84.17% of
its squared Euclidean norm outside that subspace. This quantifies a local
constraint in coefficient coordinates, not an expected validation gain.
No weights were updated. Results, raw gradients, and group statistics are
in `analysis/tensor_gradient_diagnostic/` (52.3 seconds).

Actual parameter-displacement analysis was also refreshed. RegMean and
RegMean++ move the backbone by L2 54.90 and 46.76, versus Block18.27;
fixed TIES moves it by24.82. These measurements describe the resulting
weights without interpreting coefficient sums as model displacement.

`inputs/tensor_granularity_plan.json` now fixes two next granularity checks:
per-expert/per-tensor CE (6,128 coefficients), and a tensor-scale control
(383 scales on the uniform delta). Both preserve the existing CE, data,
1600 updates, Adam0.01, batch128, and validation schedule. This directly
tests flexibility within the shared merge; it adds no deployment modules.
Integration waits for the active TIES trainer to finish so its source
receipt remains accurate. These prepared checks do not alter that run.

Fixed TIES official source macro BAcc is 0.47840625. Its pruning retains
20% of coordinates per expert before sign election, and approximately
12.31–16.18% afterward. The uniform-transformed-vector/disjoint-mean
numerical difference is at most 2.24e-8. This is the fixed scale-1 preset;
the reference README also describes scale selection, which was omitted
here and explicitly replaced by the separately learned Block-CE variant.
The latter remains below Block through step800; its fixed budget continues.

### Learned TIES completed; tensor-granularity integration checked

TIES + Block-CE completed all 1600 updates and selected step960, validation
BAcc 0.6244516298 (Block difference -0.2757 pp). Eleven tasks decrease;
the worst decrease is MI_ShanghaiU -1.8315 pp. The final step1600 score
was lower than this selected checkpoint. All 33 gradients were nonzero;
export error <=6.41e-7 and frozen-source checks passed before edits. The
fixed TIES family does not pass the pilot gate.

After that trainer finished, added `--granularity tensor` to the existing
merger for the two prepared CE variants. Grouping changes from33 structural
groups to383 active parameter tensors; all composition/optimization/readout
code is shared. The full coefficient table is retained in CSV/NPZ, while
large heatmaps omit cell text and cap their height for readability.

Both real two-task/four-update smokes passed: all383 groups have finite,
nonzero gradients and coefficients change during training. Both selected
the unchanged initial checkpoint; no smoke accuracy improvement is claimed.
Their selected checkpoint exports passed. Exact checks are recorded in
`inputs/seed0/tensor_granularity_smoke_review.json`; sources are archived in
`inputs/source_snapshot_20261114_tensor`. Full tensor-CE will run on GPU1;
the tensor-scale control will run on GPU0 after the learned-TIES official
evaluation finishes. Their pre-specified settings remain unchanged.

Learned TIES official source macro BAcc is 0.566350, versus fresh Block
0.572700 (-0.635 pp). The report now contains all26 official evaluations.
Both tensor-granularity full runs are active: `tensor_ce` on GPU1 and
`tensor_scale_ce` on GPU0. Their settings were fixed before this TIES test
result was read. Source `merge_poc.py` is frozen again for those runs.

The first tensor-CE validations are 0.627093 at step160 and 0.626859 at
step320, versus Block's 0.621147 and 0.617735 at those same update counts.
The completed Block best remains 0.627208. This shows faster early
improvement, not a passed final pilot. At step320, Workload is -3.694 pp
relative to selected Block. The tensor-scale control is 0.603188 at step160.
Both runs continue their fixed1600-update budgets; no settings change.

### Tensor candidate passes the interim pilot; replication prepared

Tensor-CE reached validation BAcc0.633870 at step480 (+0.666 pp vs Block,
worst task -0.909 pp), then0.6370 at step960. Training remains in progress;
the preset final selection and export checks still determine the actual
pilot result. The tensor-scale control remains below Block at observed
checkpoints. No tensor official test score has been read.

Prepared `inputs/tensor_replication_plan.json`: if the completed selected
seed0 checkpoint passes the existing gate, repeat the fixed tensor-CE and
fresh Block-CE pair on expert/calibration/optimizer seeds1 and2. All32
existing experts have the expected task/seed/common-base metadata, and
exact old calibration files exist. No experts will be retrained. The
seed0 tensor-scale run remains a mechanistic pilot control.

`run_aligned_stage.py --seed 1|2` now produces the paired seed-specific
commands; dry-run checks verified expert directories, calibration paths,
optimizer seed and output paths. The active training implementation was
not changed. `report_replication.py` will report paired differences, means
and sample SD across training seeds, along with per-task regressions and
explicit incomplete states. It checks matched settings, expert paths,
calibration arrays and dataset-update order before combining a pair.
Sources are archived in `inputs/replication_snapshot_20261114`.

Official tensor results will remain unread until all planned replications
finish. This prospective confirmation addresses the extensive seed0
exploration; it does not establish a CPSC effect or semantic mechanism.

### Completed tensor pilot and paired replication launch

The selected tensor-CE checkpoint is step960, validation macro BAcc
0.6369966424 versus fresh Block0.6272082324 (+0.978841 pp). Fourteen tasks
improve; the two declines are SEED-5 (-0.660419 pp) and SEED-7 (-0.572916 pp).
The completed pilot passes the predeclared +0.5 pp / worst-task -3 pp gate.
Tensor-scale selects step1600 at0.6217297502 (-0.547848 pp vs Block), with
nine tasks down and ADHD -4.545455 pp. Thus per-tensor uniform-delta scaling
alone did not reproduce the expert-specific coefficient gain in this seed.

Both runs completed1600 updates, have finite nonzero gradients for all383
parameter-tensor groups, use the exact original calibration IDs, and match
their recorded source hashes. Exported-checkpoint logits differ from the
functional model by at most1.20e-6; full folded validation BAcc is identical
to the selected result. Detailed per-task results and audit evidence are in
`inputs/tensor_granularity_full_review.json`.

Launched the fixed paired replications exactly as prepared:
`run_aligned_stage.py --seed 1 --gpu 0 --jobs block_ce,tensor_ce` and
`run_aligned_stage.py --seed 2 --gpu 1 --jobs block_ce,tensor_ce`.
Each GPU runs its fresh Block control followed by tensor-CE, with no source
or configuration changes. Commands and exits are recorded in each seed's
`queue_events.jsonl`. The global replication report currently contains the
completed seed0 pair and explicitly marks the other pairs/tests pending.

Attribution check: AdaMerging's official layer-wise implementation already
uses one task coefficient per parameter/state tensor. Its reference uses
clamped coefficients,0.3 initialization and unlabeled entropy; this LEAF
candidate uses signed coefficients,1/K initialization and labeled CE with
fixed text readouts. Therefore tensor granularity is an established design,
not a new semantic-alignment mechanism. The checked commit, source URL and
differences are recorded in `inputs/adamerging_reference.json`.

### Tensor pilot dynamics recorded

`report_replication.py` now saves focused validation BAcc/CE trajectories
for all completed paired runs and a same-update comparison CSV. On seed0,
tensor-CE exceeds Block at all10 post-initialization evaluation steps:
median gain +0.913843 pp, minimum +0.033940 pp, and final-step gain
+0.915302 pp. Its final score is only0.063539 pp below its selected peak.
This describes persistence within the exploratory run, not independent
replication evidence. Seed1/2 remain in their fresh Block training phase.

Actual parameter-displacement analysis was refreshed from exported
checkpoints. Backbone/interface L2 distances from Base are17.3920/13.4573
for tensor-CE,18.2747/13.6798 for Block, and7.2633/10.7024 for tensor-scale.
The tensor candidate's gain therefore does not coincide with a larger
overall displacement norm in this seed; the geometry alone does not
identify a causal mechanism. All35 completed seed0 checkpoint measurements
are saved in `seed0/analysis/parameter_geometry.json`.

The updated reports were executed successfully and the focused figure was
visually checked. Reporting sources are archived separately in
`inputs/replication_dynamics_snapshot_20261114`; active training sources
remain unchanged. Next: complete the two fixed Block/tensor pairs, audit
exports/settings/calibration, then evaluate the fixed checkpoints with the
unchanged official L2 protocol and update the paired three-seed report.

### Replication provenance and delivery evidence

Completed full `audit_aligned.py` runs for seeds1 and2. Each audit verifies18
source checkpoints (Base, Joint,16 experts), expected initialization hashes,
strict parameter signatures,4 fixed buffers,9 inactive tensors,33 groups,
text metadata and16 dataset split/calibration manifests. All checks passed;
source POC files were read only. The audits ran briefly on CPU alongside the
replications, so process wall times should not be read as isolated benchmarks.

Independently inspected11 main/control seed0 exported state dictionaries.
All have the Base keys/shapes/dtypes and exactly preserve all13 fixed tensors.
Recorded functional/export errors are at most1.20e-6; selected and reloaded
full-validation BAcc agree. Actual deployed capacity is26,424,208 parameters,
of which25,751,552 participate in the downstream path. Evidence is in
`inputs/seed0/deployment_review.json`.

Extended `report_replication.py` with original Base/Joint references, per-task
official differences, and phase/runtime/memory/checkpoint-size CSVs. Existing
reference comparisons verify instruction strings and label order. Each seed's
expert bank originally used48,000 updates; its sunk training cost is reported
separately from the1,600-update merge. The report executes successfully on the
currently complete seed0 pair and marks pending results explicitly.

Added `STATUS.md` as a concise requirement/evidence map, with the negative
CPSC conclusion, conditional experiments marked `not_run`, remaining
replication/test work and concrete next commands. Training sources and the
predeclared replication configuration remain unchanged. Both fresh controls
were live at step480 on the latest check; tensor runs follow automatically.

### Seed2 queue-controller interruption

Observed stage-controller PID540204 terminal with exit143; the original
trainer PID540205 remains live, now reparented to PID1. No terminal training
failure occurred and the latest Block history is step640. The seed1 stage
controller/trainer remain live. The cause of the controller's signal is not
established. Do not restart the active seed2 trainer or overwrite its output.
After it finishes and its1600-update completion/export artifact is verified,
launch only the pending seed2 tensor-CE job with the exact prepared command.
Record the observed completion separately because the terminated controller
cannot supply its child's exit status.

The subsequent process/GPU check showed trainer540205 had also disappeared,
with no `run.json`/export and no traceback. The last recorded evaluation is
step800; kernel logs are not readable, so the termination cause and trainer
exit code remain unknown. This is a terminal interrupted attempt, not a
validation-based early stop. Its complete available trace and process log
were moved intact to `seed2/interrupted/block_ce_attempt1`.

Started the exact seed2 Block→tensor queue again in a detached process session,
recorded in `seed2/queue_attempt2.json`. No training source, initialization,
seed, calibration IDs, optimizer, update budget or selection rule changed.
No resumable optimizer state was saved, so the rerun starts from the same
uniform initialization. Record interrupted compute separately from completed
method timing. Seed1 continues its original uninterrupted run.

### First replication control completed

Seed1 Block completed1600 updates and selected step800 at validation macro
BAcc0.6277729698. All33 group gradients are finite/nonzero, calibration IDs
are exact, source hashes match, maximum export-logit error is1.12e-6 and
reloaded full-validation BAcc is identical. Audit:
`inputs/seed1/block_full_review.json`. Total process time615.49s includes
setup, validation and export; the brief concurrent CPU audit is noted above.
The existing queue automatically launched seed1 tensor-CE (PID550833) with
the predeclared settings.

The detached seed2 rerun is live. At common checkpoints0/160/320/480, every
task's BAcc exactly matches the archived interrupted attempt (maximum
difference0). The comparison is saved in the archived attempt's
`rerun_trajectory_check.json`. Its800-update interrupted trace contains
257.29s of recorded optimization/validation and at most450.12s from launch
to observed termination; this extra cost is now listed separately by the
replication report. No candidate test has been run or read.

### Prospective seed1 interim signal; seed2 recovery verified

The seed1 tensor run reached0.6334989883 at step640 versus completed
Block0.6277729698 (+0.572602 pp), with worst task ADHD -2.651515 pp.
This interim checkpoint passes the pilot resource threshold; it is not yet
the completed run's selected/exported result. At the latest observation the
run had reached step800 and was continuing unchanged to1600 updates.

Seed2's entire archived800-update trajectory was reproduced exactly by the
rerun: task-update order, every recorded training CE, and per-task BAcc/CE at
0/160/320/480/640/800 all have zero difference. The saved recovery comparison
now covers the full interrupted trace. Seed2 Block has progressed to1280;
its tensor candidate remains queued. No official tensor test feedback exists.

### All three Block controls complete

Seed2 Block rerun completed1600 updates and selected step1280 at validation
macro BAcc0.6251795544. All33 gradients are finite/nonzero, source hashes and
original calibration IDs match, and reloaded validation BAcc is identical.
Maximum functional/export logit error is1.01e-6; completed process time586.57s.
The interrupted-attempt cost remains separate. Review evidence:
`inputs/seed2/block_full_review.json`.

The detached queue launched seed2 tensor-CE automatically (PID557547).
Seed1 tensor-CE remained live at step1120, with selected-so-far step640
BAcc0.6334989883. All planned fresh Block controls are now complete; the
remaining training work is the two unchanged tensor candidates, followed by
their export audits and fixed official evaluations.

### Seed1 tensor replication completed and passed

Seed1 tensor-CE completed1600 updates and selected step640, validation macro
BAcc0.6334989883 versus Block0.6277729698 (+0.572602 pp). Thirteen tasks
improve and three decline: Cho2017 -0.495050 pp, SEED-5 -1.490366 pp and
ADHD -2.651515 pp. The completed run passes the same pilot magnitude and
worst-task thresholds, without changing settings after the seed0 pilot.

All383 coefficient groups have finite nonzero gradients; source hashes and
calibration IDs match. Maximum export-logit error7.15e-7; reloaded full
validation BAcc is identical to the selected result. Evidence:
`inputs/seed1/tensor_full_review.json`. Total process time588.20s.

The paired reporter verifies matching experts/settings, calibration arrays
and task-update sequence. Tensor is above same-update Block at9/10 post-init
checkpoints in seed1; median difference+0.678725 pp, with one -0.391453 pp
checkpoint. Its final value is0.345976 pp below its selected peak. These are
within-run dynamics, not additional independent replications. The seed1 and
global reports have been refreshed successfully. Seed2 tensor remains live;
all tensor official evaluations remain pending completion of that run.

### All fixed replications complete; official testing starts

Seed2 tensor-CE completed1600 updates and selected the final step at validation
macro BAcc0.6333421983 versus Block0.6251795544 (+0.816264 pp). Eleven tasks
improve; worst decline is ShanghaiU -2.437573 pp. All383 gradients are finite
and nonzero, calibration IDs/source hashes match, maximum export-logit error
9.13e-7, and reloaded validation BAcc matches. Review:
`inputs/seed2/tensor_full_review.json`; completed process time584.11s.

All three paired audits pass. Validation mean±sample SD is Block
0.626720±0.001364 and tensor0.634613±0.002066. Paired gains are
+0.978841,+0.572602,+0.816264 pp; mean+0.789236±0.204464 pp. Every completed
seed passes the predeclared magnitude/worst-task resource gate. Seed0 remains
exploratory and the two prospective seeds use unchanged settings.

Only after all six training runs completed, launched the fixed official L2
tests: seed0 tensor and its scale control on GPU0 (existing Block reused),
seed1 Block/tensor on GPU1. Seed2 Block/tensor will follow on an available GPU.
The launch receipt `inputs/tensor_official_launch.json` records all six frozen
run hashes, commands, process IDs and the phase boundary. Test receipts from
`evaluate_aligned.py` additionally bind each evaluated checkpoint and source.
No official tensor score had been read at launch.

### Tensor official replication complete: validation gain mostly attenuates

| Seed | Block official L2 | Tensor official L2 | Difference (pp) | Worst task change (pp) |
|---|---:|---:|---:|---|
|0|0.572700|0.572631|-0.006875|Workload -11.390|
|1|0.567738|0.569081|+0.134375|Shin2017A -3.610|
|2|0.571700|0.574413|+0.271250|OpenBMI -2.430|

Official mean±sample SD is Block0.570713±0.002624 and
tensor0.572042±0.002714; paired difference+0.132917±0.139068 pp.
This is much smaller than the validation gain+0.789236±0.204464 pp.
Seed0 improves11 of16 official tasks but its Workload regression cancels the
macro gain. The completed seed0 tensor-scale control is0.554175, below Block.
Tensor granularity is a repeatable validation improvement and a modest test
trend, but it has not established a robust test improvement. Do not present
the pilot gate alone as success on the main protocol.

All six official evaluation receipts match the frozen checkpoint/run hashes,
the unchanged evaluator,16-task manifest, instruction seed42 and batch256.
All six final state dictionaries preserve the13 fixed tensors exactly.
Evidence: `inputs/tensor_replication_official_review.json` and
`inputs/tensor_replication_deployment_review.json`. Final sources for this
stage are archived in `inputs/tensor_final_snapshot_20261114` before further
training-code changes. The global replication report is complete.

### Next bounded hypothesis: class-balanced calibration CE

Fixed `inputs/class_balance_plan.json` before implementation/runs. Current
CE averages examples, whereas BAcc averages classes. Calibration counts show
Workload382/130, SEED-5 max/min ratio2.83 and SEED-7 ratio2.27; the fixed
inverse-frequency rule applies to every task, not only observed failures.
This mismatch is a candidate explanation, not an established causal finding.
Workload's training and official instruction strings are identical, so a
different instruction phrase is not an explanation for its seed0 regression.

Implemented `--class-balanced-ce`: mean minibatch loss
`N_cal/(C*n_y) * CE(logits,y)`. The full-calibration expectation is mean
class-conditional CE. It uses only existing calibration labels, preserves
the original validation/selection rule and records both weighted and raw CE,
class counts and exact weights. Original default CE remains unchanged.
Prepared matched `block_balanced_ce` and `tensor_balanced_ce` jobs with the
same1600 updates, learning rate, seeds, calibration IDs and fixed readout.
No weight/LR sweep or new architecture is planned. Existing official results
are now known and this development exposure is explicitly recorded.

Real four-update smoke runs on Workload/SEED-5 are active for both coefficient
layouts. Full runs remain gated on weights, finite gradients and export
checks. `report_aligned.py` recognizes the new methods and matched controls.

Both class-balanced smoke runs passed. All33/383 coefficient groups received
finite nonzero gradients; all original calibration IDs are exact. Recorded
weights matchN/(C*n) and give equal total class mass over the full calibration
pool. Weighted and unweighted training CE are both logged. Maximum export
errors are9.00e-7 (Block) and5.07e-7 (tensor); reloaded validation matches.
Both selected smoke step4; these small-budget scores are engineering checks,
not performance evidence. Review: `inputs/seed0/class_balance_smoke_review.json`.

Archived sources in `inputs/source_snapshot_20261114_class_balance`, then
launched `block_balanced_ce` on GPU0 and `tensor_balanced_ce` on GPU1 in
detached sessions. Exact commands/PIDs are in `inputs/class_balance_launch.json`.
Frozen training source SHA256:
`44c6232bb2736e7c6c4938c8f7fccfc1aab5f89ce3c8cc28e5c0d80f43eab0f2`.
Each full run uses the unchanged1600-update budget and original selection
rule. No balanced-CE official evaluation is planned before the fixed pilot
and, if warranted, seed1/2 replication finish.

### Class-balanced pilot observation and reporting

Both weighted pilots reproduced the original uniform step0 validation macro
BAcc0.5857470611 exactly. By step480, balanced tensor's best is step320 at
0.6325757022 (+0.536747 pp vs fresh Block; worst task SSVEP -0.964209 pp).
It passes the resource gate provisionally but is0.442094 pp below the
completed original tensor-CE best. Balanced Block's observed best remains
0.6234093290 at step160. Both runs continue the fixed1600-update budgets;
no final selection/export result is claimed yet.

`report_aligned.py` now includes a focused class-balancing comparison and
trajectory figure after full runs finish. It checks matched expert/settings,
calibration arrays and task-update order, and records first/last160-update
weighted fitting CE alongside unweighted diagnostic CE. This distinguishes
loss reweighting from improved validation. The report still separates tests
from validation; its title now correctly uses the actual seed directory.
The report executed successfully while completed balanced results remain
pending. Active training sources have not changed.

### Conditional class-balanced replication prepared

At step800, balanced Block's best is0.6274373473 (step640), essentially tied
with original Block. Balanced tensor's best remains0.6325757022 (step320).
The full pilot is still active and its completed selection/export check
remains the replication gate.

Prepared `inputs/class_balance_replication_plan.json` with exact unchanged
seed1/2 Block-balanced/tensor-balanced commands. Existing seed-specific
experts/calibration files are reused; class weights follow the same formula
on each calibration pool. No weighted official score has been observed.

`report_replication.py --family class_balanced` now creates separate
`class_balance_replication_*` artifacts and checks equal class counts/weights
within each paired layout comparison. It also compares each weighted layout
with its existing unweighted counterpart, recording per-task and paired
seed differences in `class_balance_effect_*`. This prevents a weak weighted
control from hiding the comparison with original tensor-CE. The original
report's completed statistics are unchanged; the new family is explicitly
incomplete. Both reporting modes executed successfully. No training source
or active run configuration changed.

### Class-balanced pilot completed; fixed replication launched

Balanced tensor selected step1440 at validation macro BAcc0.6386080568:
+1.139982 pp versus original Block, +0.161141 pp versus original tensor-CE,
and +1.117071 pp versus the balanced Block control. Its worst task change
versus original Block is SEED-5 -0.972808 pp, with3 tasks down. Versus the
stronger original tensor control,8 tasks decline and the worst is BCIC-IV2a
-1.455195 pp; the small macro gain must not hide these tradeoffs.
Balanced Block selected step640 at0.6274373473 (+0.022911 pp versus Block).

Both full runs completed1600 updates with finite nonzero gradients in every
coefficient group, exact original calibration IDs, correct inverse-frequency
weights, unchanged source hashes and identical selected/reloaded validation
BAcc. Maximum export-logit errors are1.52e-6 (Block) and4.36e-6 (tensor).
Review: `inputs/seed0/class_balance_full_review.json`.

The completed primary passes the predeclared pilot gate. Launched the exact
prepared seed1 and seed2 queues, each balanced Block followed by balanced
tensor, on GPU0 and GPU1 respectively. Detached launch receipt:
`inputs/class_balance_replication_launch.json`. Training code is still the
frozen44c6232b… version. No class-balanced official score has been observed;
all four new replication runs must finish before those tests.

The full seed0 comparison report and separate class-balanced replication
report executed successfully; the focused figure was visually checked.
First→last160-update raw CE is0.822156→0.756261 for original tensor and
0.823795→0.755034 for balanced tensor. Comparable final raw fitting losses
accompany a small selected-validation gain; this does not establish improved
test generalization. Full task and time-window CSVs are retained.

Across post-initialization checkpoints, tensor macro-BAcc range is1.702718 pp
without class balancing and1.117806 pp with it; Workload's range is17.570303
and10.212382 pp respectively. These ranges include learning trends and
within-run fluctuations, and are not seed uncertainty or proof of a causal
generalization mechanism (`validation_trajectory_ranges.json`).

### Per-task class-balancing loss records

Added `class_balance_task_loss_windows.csv` to the seed report:128 rows for
four completed pilot/control methods × first/last windows ×16 tasks, with
exactly10 updates per task/window. Both weighted objective CE and raw CE are
retained. Real report generation and row/update counts passed.

For Workload, original tensor raw CE is0.443055→0.385544, while balanced
tensor raw CE is0.477688→0.391166 and its weighted objective is
0.478181→0.397798. For SEED-5, late raw CE is1.036725 versus1.039158.
Thus the modest selected-validation gain does not correspond to uniformly
lower raw fitting losses on the imbalanced tasks. These are calibration
minibatch diagnostics, not held-out generalization evidence.

Both seed1/2 balanced Block controllers and trainers are live at step480 on
the latest observation; their tensor runs remain queued. Training sources
and fixed configurations remain unchanged.

### Seed1/2 balanced Block controls completed

Seed1 selects step320 at validation macro BAcc0.6273678127
(-0.040516 pp versus original Block); seed2 selects step1600 at0.6269018298
(+0.172228 pp). The worst task changes versus original Block are Shin
-3.152455 pp and HighGamma -1.787892 pp respectively. These are controls,
not evidence of a benefit from class weighting alone.

Both completed1600 updates with33 finite nonzero gradient groups,528
coefficients, exact original calibration IDs, matched task-update order,
correct inverse-frequency weights and unchanged frozen source hashes.
Maximum export-logit errors are7.45e-7 and4.25e-7. Reviews are saved in
`inputs/seed{1,2}/block_balanced_full_review.json`.

Seed2 selected/reloaded BAcc is identical. Seed1 has an explicitly recorded
SEED-7 change of+0.004409 pp after reload (+0.000276 macro pp). A full
validation margin check is prepared in `check_merge_export_margins.py` to
compare reconstructed functional, reloaded and validation-wrapper paths;
it will run on a free GPU after current training. No training helper was
changed. The existing wrapper renormalizes text vectors, which is one
possible numerical source to check, not yet a confirmed explanation.

Both controllers automatically started their planned tensor-balanced runs.
All six balanced training runs will finish before official evaluations.

### All balanced training completed; frozen official evaluations started

Tensor-balanced seed1 selects step1600 at0.6321970418 (+0.442407 pp
versus original Block, -0.130195 pp versus original tensor); seed2 selects
step1280 at0.6339380558 (+0.875850 pp and+0.059586 pp respectively).
Seed1 misses the0.5-pp resource gate. Across three seeds, weighting changes
tensor validation macro BAcc by only+0.030178 pp on average. Tensor still
beats matched balanced Block by+0.767872±0.321919 pp. All negatives remain.

All six planned runs completed before any balanced official test. State
checks pass for architecture keys/shapes/dtypes, finite parameters and13
exact base buffers/inactive tensors. The causal mask intentionally contains
negative infinity; a first overly broad all-tensors-finite audit assertion
was corrected to check finite parameters and exact original buffers.
Deployment evidence: `inputs/class_balance_replication_deployment_review.json`.

The seed1 control discrepancy is resolved by a full18542-row SEED-7 check:
reconstructed state error0; functional/reloaded logit max3.606e-6;
one prediction changes at row17324 with original top-two margin7.45e-7,
within twice the measured error. Selected/reloaded metrics reproduce
exactly. The plain reload changes the same prediction, so wrapper text
renormalization is not required to explain it. No source or checkpoint
changed (`inputs/seed1/block_balanced_export_margin_review.json`).

All six run/checkpoint hashes were frozen before launching the unchanged
L2 seed42 batch256 evaluator. Seeds0/1 official queues are active; seed2
is prepared for the next free GPU. Receipt: `inputs/class_balance_official_launch.json`.

### Class-balanced official result complete; one fixed averaging follow-up

Official balanced tensor macro BAcc is0.57500625/0.56956875/0.57485625.
Its paired gain over original Block is+0.243125±0.067129 pp, positive in
all3 seeds. Over original tensor the gain is+0.110208±0.110260 pp.
Against the equally weighted Block control, the gain is+0.202500±0.326327 pp,
including seed2 -0.155000 pp. Thus weighting gives a modest consistent gain
against original Block, but stable superiority to the matched balanced
control is not established. Worst task changes versus original Block are
ADHD -4.09 pp, ADHD -3.34 pp, and OpenBMI -1.66 pp. Seed0 Workload improves
7.78 pp versus unweighted tensor but remains3.61 pp below original Block.
All six official receipts match frozen checkpoint/run/evaluator hashes.
The source-task macro excludes the evaluator CSV's summary row.

The separate class-balanced replication report and complete per-task/source
comparisons are retained. No prior result is superseded or removed.

Fixed the next bounded hypothesis before implementation in
`inputs/trajectory_average_plan.json`: average coefficient states at
steps800/960/1120/1280/1440/1600 from the existing class-balanced tensor and
Block trajectories, with zero new optimization. The same arithmetic rule
applies to both layouts and all seeds; no averaging-window search. This
addresses the observed checkpoint fluctuations. It is related to standard
weight averaging (Izmailov et al., arXiv:1803.05407v3), adapted here to Adam
coefficient trajectories; neither novelty nor improved generalization is
assumed. All prior official results are already exposed and recorded as
such. Planned pilot/replication/test phase boundaries remain explicit.

### Fixed averaging implementation and engineering smoke passed

Added `average_merge_trajectory.py` and named stage jobs for fixed averages.
No training source changed. Arithmetic mean coefficients are checked exactly
against an independent NumPy FP64 reduction of the prescribed six states;
composing their mean differs from averaging composed states by1.99e-7.
Reloaded state error is0. All fixed base buffers/inactive tensors are exact.
The Workload/SEED-5 smoke's full-validation logit errors are1.55e-6/2.86e-6;
calibration IDs and source snapshot indices are exact. No performance claim
is made from this engineering subset. Additional smoke cost14.32 seconds.
Review: `inputs/seed0/trajectory_average_smoke_review.json`.

Both seed0 full averaging jobs are active on separate GPUs. Their source
trajectories and all original/weighted selected-checkpoint results remain
unchanged. Added the two new completed-run names to the existing seed report.
The class-balanced three-seed report and all seed reports executed; the
completed three-seed dynamics figure was visually checked.

### Fixed averaging pilot passes; same-rule replications launched

Tensor-balanced average reaches validation macro BAcc0.6383440763:
+1.113584 pp versus original Block and -0.026398 pp versus the selected
balanced tensor source. Worst task versus original Block is ADHD -0.833333
pp, with3 tasks down; the fixed primary gate passes. The averaged Block
control reaches0.6235877958, -0.362044 pp versus original Block and
-0.384955 pp versus its selected balanced source. It is a weaker control,
so both selected and averaged comparisons are retained.

Full16-task export checks pass: max errors4.46e-6/4.32e-6, no changed
predictions, identical selected/reloaded BAcc, exact fixed-state/calibration
checks. Each adds78.95 seconds of postprocessing and full paired validation,
with zero optimizer updates. Source fit costs are counted separately.
Review: `inputs/seed0/trajectory_average_full_review.json`.

Added `report_trajectory_average.py` with per-seed/task paired comparisons,
means/sample SD, source versus postprocessing cost, fixed-window/source hash
checks, and source trajectories with evaluated averaged-model scores. Its
real pilot report executed successfully. The exact averaging rule is now
running on seeds1/2, each Block followed by tensor. Receipt:
`inputs/trajectory_average_replication_launch.json`. No averaged official
score has been observed; all planned exports complete first.

### Fixed averaging replications and full state checks completed

Tensor averages in seeds1/2 score0.6334077683/0.6359278117 on validation,
+0.563480/+1.074826 pp over original Block. Their worst task declines are
ADHD -2.651515 pp and Shanghai -2.391922 pp. Together with seed0, all three
primary outputs meet the predeclared validation resource gate. Averaged
Block scores are0.6235877958/0.6305205361/0.6258346334; averaging helps that
layout only in seed1 relative to its selected balanced source.

All six outputs pass independent source/history hashes, exact fixed mean,
calibration IDs, architecture/state checks,13 fixed tensors and finite
parameters. Full validation export-logit errors are at most4.94e-6; saved
states reconstruct exactly. Evidence: `inputs/trajectory_average_deployment_review.json`.
Actual backbone/interface weight displacements, including differences from
the selected source checkpoint, are saved in
`analysis/trajectory_average_parameter_geometry.json`; no coefficient-sum
proxy is used and geometry alone is not a generalization explanation.

All six outputs were frozen before official tests. Seeds0/1 are evaluating
under the unchanged L2 seed42 batch256 protocol; seed2 will use the next
free GPU. Receipt: `inputs/trajectory_average_official_launch.json`.

### Potential candidate established; bounded search concluded

All averaged official tests are complete. Tensor-average scores are
0.574525/0.57193125/0.578825; mean0.57509375±0.00348189. Gains over
original Block are+0.182500/+0.419375/+0.712500 pp, pairedmean
+0.438125±0.265497 pp. Against averaged balanced Block the gains are
+0.338750/+0.392500/+0.240000 pp, pairedmean+0.323750±0.077349 pp.
All three also improve over selected balanced Block (+0.397500±0.140322 pp)
and original tensor (+0.305208±0.127148 pp). Averaging alone improves2/3
balanced tensor sources: -0.048125/+0.236250/+0.396875 pp.

The combined approach now meets the requested narrower outcome: a method
with potential, supported by consistent official macro gains against the
original and matched strong controls after all three validation gates pass.
This is not a claim of broad task robustness or paper-scale completion.
ADHD falls5.02/4.89 pp in seeds0/1 versus original Block; seed0 Workload
falls4.17 pp. Joint remains2.634583±0.327398 pp ahead. These limitations and
prior benchmark exposure remain prominent in STATUS and the report.

All six official receipt statuses,16-task completion lists,protocol fields,
frozen run/checkpoint/evaluator hashes and phase ordering passed.
`inputs/trajectory_average_official_review.json` records those checks.
All six full validation exports have no prediction changes and exact BAcc,
with maximum logit error4.94e-6. Source parameter states reconstruct exactly;
fixed buffers/inactive tensors remain unchanged. Average export/validation
adds79–80 seconds and zero updates; total source-fit-plus-postprocess cost
is664–667 seconds per candidate. No new speedup claim is made.

Final seed reports and the three-seed averaging report executed successfully;
the complete trajectory figure was visually checked. Per-task score tables,
first/last loss windows,coefficient histories,actual parameter geometry,
negative methods and the interrupted original attempt are retained. The
original checkpoints/POC assets remain untouched. No further sweep is queued;
P3/P4 remain explicit follow-up stages beyond the achieved current goal.
