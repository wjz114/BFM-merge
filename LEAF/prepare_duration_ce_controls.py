"""Freeze a three-pass CE budget control with no new training-row access."""
import copy
import json
import time
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    root = Path('runs/paper_merge_v2')
    old_path = root/'inputs/base_initialization_ce_lr1e5_plan_v2.json'
    old = json.loads(old_path.read_text())
    for path, digest in old['source_sha256'].items():
        assert sha256(Path(path)) == digest, path
    previous_schedule = Path(old['schedule'])
    assert sha256(previous_schedule) == old['schedule_sha256']
    previous = np.load(previous_schedule)
    assert previous['task'].shape == (1600,) and previous['local_indices'].shape == (1600, 128)
    task_order = np.tile(previous['task'], 3)
    local_indices = np.tile(previous['local_indices'], (3, 1))
    for start in [0, 1600, 3200]:
        assert np.array_equal(task_order[start:start+1600], previous['task'])
        assert np.array_equal(local_indices[start:start+1600], previous['local_indices'])
    schedule_path = root/'inputs/duration4800_ce_schedule.npz'
    assert not schedule_path.exists()
    np.savez(schedule_path, task=task_order, local_indices=local_indices)
    controls = {'merged': root/'seed0/full_control_coverage_full_balanced_ce/run.json',
                'base': root/'seed0/base_initialization_ce_lr1e5/run.json'}
    runs = {k: json.loads(p.read_text()) for k, p in controls.items()}
    tasks = runs['base']['datasets']
    coverage_path = Path(old['coverage_audit'])
    assert sha256(coverage_path) == old['coverage_audit_sha256']
    coverage = json.loads(coverage_path.read_text())
    data_audit = {}
    for i, task in enumerate(tasks):
        old_rows = previous['local_indices'][previous['task'] == i].reshape(-1)
        new_rows = local_indices[task_order == i].reshape(-1)
        n = coverage['audits'][task]['train_size']
        old_counts, new_counts = [np.bincount(x, minlength=n) for x in [old_rows, new_rows]]
        assert np.array_equal(new_counts, 3*old_counts)
        data_audit[task] = {'unique_training_rows': int((new_counts > 0).sum()),
                            'previous_exposures': len(old_rows), 'new_exposures': len(new_rows),
                            'all_row_multiplicities_exactly_tripled': True,
                            'newly_accessed_training_rows': 0}
    worker = Path('train_duration_ce_control.py')
    source_hashes = copy.deepcopy(old['source_sha256'])
    source_hashes.update({str(p): sha256(p) for p in [worker, Path(__file__), old_path]})
    arms = {k: {'plan': str(root/f'inputs/duration4800_ce_{k}_plan.json'),
                'outdir': str(root/f'seed0/duration4800_ce_{k}'),
                'log': str(root/f'seed0/logs/duration4800_ce_{k}.log'),
                'gpu': i, 'control': str(p), 'control_sha256': sha256(p)}
            for i, (k, p) in enumerate(controls.items())}
    design_path = root/'inputs/duration4800_ce_design.json'
    design = {'fixed_unix': time.time(), 'source_sha256': source_hashes, 'worker': str(worker),
              'datasets': tasks, 'arms': arms, 'steps': 4800, 'batch_size': 128,
              'lr': 1e-5, 'eval_every': 160, 'average_steps': [2400, 2880, 3360, 3840, 4320, 4800],
              'schedule': str(schedule_path), 'schedule_sha256': sha256(schedule_path),
              'prefix_schedule': str(previous_schedule), 'prefix_schedule_sha256': sha256(previous_schedule),
              'data_audit': data_audit,
              'hypothesis': 'Determine how much ordinary CE can improve at three times the optimization budget on exactly the same finite training multiset, and whether merger initialization retains an advantage over the released base.',
              'intervention': 'From each original starting checkpoint, run the existing1600-step schedule three times consecutively with one continuous Adam state. All row multiplicities exactly triple; no new rows, teacher targets, augmentation, LR, dropout or BN changes.',
              'primary': 'Prespecified second-half six-state mean for both starts. Report4800-minus1600 within each start, merger-minus-base at4800, the interaction, all task effects and leave-one-task-out sensitivity. Retain final and selected states as secondary only.',
              'engineering_gate': 'The training/update loop is inherited from the previously completed base CE worker. Before the budget extension begins, each run must reproduce all six existing800/960/1120/1280/1440/1600 state dictionaries bitwise and all validation metric dictionaries exactly. Failures are recorded and stop the run. No new separate short scientific arm.',
              'budget': {'new_optimizer_updates_per_arm': 4800, 'exposures_per_arm': 614400,
                         'unique_fitting_rows_per_arm': sum(v['unique_training_rows'] for v in data_audit.values()),
                         'merger_preexisting_coefficient_updates': 1600,
                         'expert_bank_preexisting_updates': 48000},
              'motivation': {'decision_diagnostic': str(root/'analysis/decision_disagreements.json'),
                             'decision_diagnostic_sha256': sha256(root/'analysis/decision_disagreements.json'),
                             'joint_curves': str(root/'analysis/historical_joint_learning_curve_audit.json'),
                             'joint_curves_sha256': sha256(root/'analysis/historical_joint_learning_curve_audit.json'),
                             'boundary': 'Training disagreements and continuing historical Joint improvement motivate a conventional budget comparator, not a causal duration claim or a new method.'},
              'scope': 'Single-bank exposed-validation conventional control. Same data support and empirical row weights, larger optimization budget and later prescribed averaging window. Both initializations retained; no duration/rate sweep or independent confirmation. This is not an optimized short Joint recipe or end-to-end compute saving. No official test access.'}
    write_new(design_path, design)
    for key, arm in arms.items():
        run = runs[key]
        assert run['updates'] == 1600 and run['schedule_sha256'] == old['schedule_sha256']
        assert run['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        reference_path = root/f'inputs/duration4800_ce_{key}_initial_reference.json'
        write_new(reference_path, {'folded_val': run['initial_validation']['validation'],
                                   'source': arm['control'], 'source_sha256': arm['control_sha256']})
        prefix = {str(row['step']): {'path': row['path'], 'sha256': row['sha256']} for row in run['average_snapshots']}
        for spec in prefix.values():
            assert sha256(Path(spec['path'])) == spec['sha256']
        history = Path(arm['control']).with_name('history.json')
        checkpoint = Path(run['plan']['initial_checkpoint'])
        plan = {k: copy.deepcopy(old[k]) for k in ['source_fit', 'calibration', 'calibration_sha256',
                'training_pool', 'coverage_audit', 'coverage_audit_sha256', 'seed', 'batch_size', 'lr', 'conditions']}
        plan.update(fixed_unix=time.time(), design=str(design_path), design_sha256=sha256(design_path),
                    initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=sha256(checkpoint),
                    initial_reference=str(reference_path), initialization=key,
                    schedule=str(schedule_path), schedule_sha256=sha256(schedule_path),
                    steps=4800, eval_every=160, average_steps=design['average_steps'],
                    prefix_reference_snapshots=prefix, prefix_reference_history=str(history),
                    source_sha256=copy.deepcopy(source_hashes), scope=design['scope'])
        plan['source_sha256'].update({str(reference_path): sha256(reference_path),
                                     str(history): sha256(history), arm['control']: arm['control_sha256']})
        write_new(Path(arm['plan']), plan)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path), 'budget': design['budget'], 'arms': arms}, indent=2))


if __name__ == '__main__':
    main()
