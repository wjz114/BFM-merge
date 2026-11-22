"""Freeze one batch-organization pair and its deterministic native128 gate."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256
from task_mixing_support import build_micro_schedule


ROOT = Path('runs/paper_merge_v2')


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['initial', 'scientific'], required=True)
    args = parser.parse_args()
    design_path = ROOT/'inputs/task_mixing_design.json'
    if args.phase == 'scientific':
        design = json.loads(design_path.read_text())
        gate_path = ROOT/'analysis/task_mixing_ce_replay_verification.json'
        gate = json.loads(gate_path.read_text())
        assert gate['trainer_sha256'] == design['plan_template']['source_sha256']['train_task_mixing_ce.py']
        assert gate['all_native_metrics_histories_data_and_component_gradient_max_exact']
        for mode in ['same_task_micro4', 'mixed_task_micro4']:
            plan = dict(design['plan_template'])
            plan.update(design= str(design_path), design_sha256=sha256(design_path),
                        batch_organization=mode, **design['schedules'][mode],
                        engineering_gate=str(gate_path), engineering_gate_sha256=sha256(gate_path))
            path = ROOT/f'inputs/task_mixing_{mode}_plan.json'
            write(path, plan)
            print(path, sha256(path))
        return
    reference_path = ROOT/'seed0/bank_ce_deterministic_replay/run.json'
    reference = json.loads(reference_path.read_text())
    full_path = ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'
    full = json.loads(full_path.read_text())
    keys = ['source_fit', 'initial_checkpoint', 'initial_checkpoint_sha256', 'calibration',
            'calibration_sha256', 'schedule', 'schedule_sha256', 'steps', 'eval_every',
            'average_steps', 'batch_size', 'seed', 'lr', 'training_pool',
            'coverage_audit', 'coverage_audit_sha256']
    template = {k: full['plan'][k] for k in keys}
    template.update(fixed_unix=time.time(), conditions=['balanced_ce'])
    sources = dict(reference['plan']['source_sha256'])
    for name in ['train_task_mixing_ce.py', 'task_mixing_support.py',
                 'check_task_mixing_ce_replay.py', 'verify_task_mixing_ce_replay.py',
                 'prepare_task_mixing_control.py']:
        sources[name] = sha256(Path(name))
    for name, expected in sources.items():
        assert sha256(Path(name)) == expected
    template['source_sha256'] = sources
    schedules, audits = {}, {}
    for mode in ['native128_replay', 'same_task_micro4', 'mixed_task_micro4']:
        parent = reference if mode == 'native128_replay' else full
        with np.load(parent['plan']['schedule']) as original:
            micro = build_micro_schedule(original, mode)
            path = ROOT/f'inputs/task_mixing_{mode}_schedule.npz'
            with path.open('xb') as handle:
                np.savez(handle, **micro)
            schedules[mode] = {'micro_schedule': str(path), 'micro_schedule_sha256': sha256(path)}
            counts = {}
            for i, task in enumerate(full['datasets']):
                rows = micro['local_indices'][micro['task'] == i].reshape(-1)
                counts[task] = {'sample_exposures': len(rows), 'unique_rows': len(np.unique(rows))}
            audits[mode] = {'per_task': counts, 'total_exposures': sum(v['sample_exposures'] for v in counts.values()),
                            'total_unique_rows': sum(v['unique_rows'] for v in counts.values()),
                            'identical_task_row_multiset_in_every16_step_block': True,
                            'shape_tasks': list(micro['task'].shape), 'shape_rows': list(micro['local_indices'].shape)}
    assert audits['same_task_micro4'] == audits['mixed_task_micro4']
    assert audits['same_task_micro4']['total_exposures'] == 204800
    assert audits['same_task_micro4']['total_unique_rows'] == 117785
    design = {'fixed_unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'motivation': 'Cached original512 full-parameter mean gradients are common descent directions for all16 tasks at three states, while between-task mean-gradient variance is high. Test ordinary batch organization before invoking gradient surgery. These are not full-pool or Adam-step measurements.',
              'hypothesis': 'Averaging four tasks per update may reduce between-task gradient fluctuations and improve short CE repair. This is an ordinary optimization baseline, not a novel algorithm claim.',
              'arms': ['same_task_micro4', 'mixed_task_micro4'],
              'plan_template': template, 'schedules': schedules, 'schedule_audits': audits,
              'references': {str(p): sha256(p) for p in [reference_path, full_path, ROOT/'analysis/parameter_gradient_conflicts.json']},
              'organization': 'Each original16-step block contains one128-row batch per task. Both arms use four32-row microbatches per update. Same-task arm preserves original task order. Mixed arm step k uses source rows k%4+4*m for m=0..3, chunk k//4. Every four mixed steps visit all16 tasks; every16-step block preserves the entire task/example multiset.',
              'objective': 'Sum four per-microbatch mean class-weighted CE losses divided by4; zero gradients once and apply Adam once. Full-pool class weights unchanged. Model remains in eval mode. No teacher, gradient surgery, optimizer change, additional labels or runtime state.',
              'primary': 'Fixed mean of steps800/960/1120/1280/1440/1600, LR1e-5,1600updates,total128examples/update. Compare mixed versus same-task micro4, and retain native128CE as reference. Report all16 effects, task-deletion sensitivity, reference-calibration fit, costs and initial/export checks. No best-snapshot or task selection.',
              'scope': 'One exposed-validation pilot at seed0; not independent confirmation or equal wall time. Each arm has6400training forward/backward calls versus1600native calls. Chunking changes per-task batch size and optimizer moment updates; it does not isolate a unique cause beyond the specified batch-organization intervention. No q/LR sweep planned.'}
    write(design_path, design)
    replay = {k: reference['plan'][k] for k in keys}
    replay.update(fixed_unix=time.time(), source_sha256=sources, conditions=['balanced_ce'],
                  design=str(design_path), design_sha256=sha256(design_path),
                  batch_organization='native128_replay', **schedules['native128_replay'],
                  scope='Engineering32-step deterministic native128 replay, not scientific efficacy.')
    path = ROOT/'inputs/task_mixing_native128_replay_plan.json'
    write(path, replay)
    print('Frozen design', sha256(design_path), 'and native128 replay plan', sha256(path))
    print('Scientific paired schedules preserve204800exposures and117785unique task/row pairs.')


if __name__ == '__main__':
    main()
