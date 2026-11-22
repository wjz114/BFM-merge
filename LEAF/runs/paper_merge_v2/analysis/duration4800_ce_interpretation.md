# Fixed CE duration control: improvement continues, initialization advantage narrows

Both v2 trajectories completed all4800 updates. The paired reporter passes
source/input checks, reconstructs all396 tensors of each of the four fixed
averages exactly, and verifies native export. The PNG/PDF figure was inspected.
The original v1 stops and their1600 wasted updates remain recorded.

The primary comparison uses the SAME continuous Adam trajectory for each
initialization. Its1600-step average uses800/960/1120/1280/1440/1600;
its4800-step average uses2400/2880/3360/3840/4320/4800. The original1600-step
batch schedule repeats three times. Each row's multiplicity triples, with
117785 unique fitting rows unchanged and no new training rows. Both runs
retain native FP32, Adam1e-5, zero decay, dropout off, fixed BN and balanced CE.
Later averaging-window placement is part of the longer procedure.

| Initialization | Same-trajectory1600 mean BAcc |4800 mean BAcc | Gain |
|---|---:|---:|---:|
| Learned merged checkpoint |65.222645% |65.494958% |+0.272313pp |
| Released base checkpoint |64.306125% |65.102851% |+0.796725pp |

The merger duration gain improves10tasks and declines6. Its leave-one-task-out
mean range is[0.193170,0.440912]pp. The base duration gain improves14tasks and
declines2, with range[0.650218,0.927360]pp. These are task-deletion sensitivity
ranges, not confidence intervals or seed variability estimates.

Merger-minus-base shrinks from0.916519pp at1600 to0.392107pp at4800. The latter
has11improved/5declined tasks and task-deletion range[0.228339,0.595015]pp.
The initialization-by-duration interaction is-0.524412pp:3tasks positive,
13negative, with every task-deletion mean negative[-0.630361,-0.377555]pp.
Thus the advantage of merger initialization persists at this reuse budget,
while the base recovers a larger amount under the longer procedure.

Relative to the initial merger63.912610% and historical Joint68.059204%,
the4800-step merger mean gains1.582348pp and closes38.160185% of that initial
gap. The remaining gap is2.564246pp. The same-trajectory1600 mean closes
31.593031%. This is one exposed-validation bank. It must not replace the
separate three-bank1600-step replication with a claim of replicated4800-step
performance, nor be mixed with historical official-development-test scores.

The extra duration is not uniformly helpful. Merger gains include
HighGamma+1.4595pp, PhysioNet+1.2145pp, IV2a+1.1885pp and SSVEP+1.1288pp;
Shin declines2.2567pp and ADHD0.9091pp. The base also declines on Shin,
by1.1628pp, while gaining2.9943pp on Workload and1.8182pp on ADHD. Every task
is retained in the report rather than choosing only the improving domains.

Historical1600 results are separate reproducibility diagnostics. The new
short merger mean differs by-0.011693pp from its historical result and the
base by-0.160098pp. Their states are not bitwise identical. Source/data hashes
and initial predictions match; native nondeterministic computation remains
the working numerical explanation, without an isolated CUDA-operator proof.
These differences are not accepted by a post-hoc tolerance, and historical
checkpoints are not substituted into either live optimizer trajectory.

Each4800 run used614400 CE examples and took about38minutes including setup,
validation and both averages on the concurrent two-GPU execution. Merger
construction additionally used48000 expert-bank updates and1600 coefficient
updates before this4800-step repair. This is not end-to-end compute savings,
an optimized short Joint baseline, or a new method. No official test access.

Decision: retain this conventional control and the matched short averages.
No further duration or learning-rate sweep is registered from this result.
The paper goal remains unachieved: a differentiated method, a causal mechanism
and independent confirmation are still missing. The next frozen read-only
audit checks whether useful-to-investigate representation structure survives
before the final projection and is absent from the CE/KL students; geometry
alone will not establish its usefulness.
