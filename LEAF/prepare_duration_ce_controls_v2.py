"""Freeze within-trajectory duration comparisons after preserving the failedv1 gate."""
import copy
import json
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256
from prepare_duration_ce_controls import write_new


def main():
    root = Path('runs/paper_merge_v2')
    previous_path = root/'inputs/duration4800_ce_design.json'
    failure_path = root/'analysis/duration_prefix_failure.json'
    previous = json.loads(previous_path.read_text())
    failure = json.loads(failure_path.read_text())
    assert set(failure['arms']) == {'merged', 'base'}
    assert all(row['stopped_at_update'] == 800 for row in failure['arms'].values())
    worker = Path('train_duration_ce_control_v2.py')
    before = Path(previous['worker']).read_text()
    after = worker.read_text()
    start, end = '    optimizer = torch.optim.Adam(', '    fit_seconds = time.time()-fit_started'
    assert before[before.index(start):before.index(end)] == after[after.index(start):after.index(end)]
    # Code review: only recording, post-training averaging and reporting differ.
    hashes = copy.deepcopy(previous['source_sha256'])
    for name, digest in hashes.items():
        assert sha256(Path(name)) == digest, name
    hashes.update({str(p): sha256(p) for p in [worker, Path(__file__), previous_path, failure_path]})
    design = copy.deepcopy(previous)
    design.update(fixed_unix=time.time(), version=2, worker=str(worker), source_sha256=hashes,
                  short_average_steps=[800, 960, 1120, 1280, 1440, 1600],
                  previous_design={'path': str(previous_path), 'sha256': sha256(previous_path)},
                  failure_audit={'path': str(failure_path), 'sha256': sha256(failure_path)},
                  update_loop_byte_identical_to_v1=True,
                  primary='For each initialization, compare the fixed1600-step mean and fixed4800-step mean from the SAME uninterrupted4800-update trajectory. Historical1600-step results and all six prefix state differences are separate reproducibility diagnostics. Retain all16task effects, both starts, initialization-by-duration interaction and task-deletion sensitivity. Selected/final states remain secondary.',
                  engineering_gate='Frozen source/data/checkpoint/schedule hashes, exact initial validation, finite trainable states, fixed-buffer equality and native-export checks remain required. Historical-prefix bitwise equality is not a gate because the existing scientific recipe permits nondeterministic kernels. All six discrepancies are recorded. Each duration average is reconstructed from its own saved snapshots after training.',
                  amendment='Thev1 exact-prefix gate was invalidly extrapolated from deterministic engineering replay. Both failedv1 runs and their hashes remain. This version changes no scientific optimizer, input, initialization or4800-step averaging setting. It adds a same-trajectory1600-step comparator; no tolerance is chosen to excuse historical differences.',
                  scope='Single-bank exposed-validation conventional control. Same finite training support and row proportions;4800updates have three times the exposures and a later prescribed averaging window. Primary duration contrast uses one continuous trajectory per initialization. Historical-prefix drift is reported separately. No optimal-short-Joint, new-method, independent-confirmation or end-to-end compute-saving claim. No official-test access.')
    for name, arm in design['arms'].items():
        arm.update(plan=str(root/f'inputs/duration4800_ce_v2_{name}_plan.json'),
                   outdir=str(root/f'seed0/duration4800_ce_v2_{name}'),
                   log=str(root/f'seed0/logs/duration4800_ce_v2_{name}.log'))
    design['budget']['failed_v1_optimizer_updates_total'] = failure['wasted_optimizer_updates']
    design_path = root/'inputs/duration4800_ce_v2_design.json'
    write_new(design_path, design)
    for name, arm in design['arms'].items():
        old_plan_path = Path(previous['arms'][name]['plan'])
        plan = json.loads(old_plan_path.read_text())
        plan.update(fixed_unix=time.time(), design=str(design_path), design_sha256=sha256(design_path),
                    short_average_steps=design['short_average_steps'], scope=design['scope'])
        plan['source_sha256'].update(hashes)
        plan['source_sha256'][str(old_plan_path)] = sha256(old_plan_path)
        write_new(Path(arm['plan']), plan)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path), 'arms': design['arms']}, indent=2))


if __name__ == '__main__':
    main()
