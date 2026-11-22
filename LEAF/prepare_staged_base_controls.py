"""Freeze two conventional direct-base controls matching both adaptation stages."""
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
    old_plan_path = root/'inputs/base_initialization_ce_lr1e5_plan_v2.json'
    old = json.loads(old_plan_path.read_text())
    audit_path = root/'analysis/staged_base_budget_audit.json'
    audit = json.loads(audit_path.read_text())
    for name, expected in audit['inputs'].items():
        assert sha256(Path(name)) == expected, name
    for name, expected in old['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    assert audit['total_additional_updates'] == 3200 and audit['unique_examples'] == 118630
    existing_base = {rate: root/f'seed0/base_initialization_ce_{rate}/run.json' for rate in ['lr1e5', 'lr1e4']}
    base_runs = {rate: json.loads(p.read_text()) for rate, p in existing_base.items()}
    assert base_runs['lr1e5']['initial_validation'] == base_runs['lr1e4']['initial_validation']
    base_reference_path = root/'inputs/staged_base_initial_reference.json'
    base_reference = {'folded_val': base_runs['lr1e5']['initial_validation']['validation'],
                      'checkpoint': old['initial_checkpoint'], 'checkpoint_sha256': old['initial_checkpoint_sha256'],
                      'sources': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in existing_base.items()},
                      'scope': 'Previously observed base initial metrics, identical across both rates. No checkpoint selection.'}
    write_new(base_reference_path, base_reference)
    common = {k: copy.deepcopy(old[k]) for k in ['engineering_gate', 'engineering_gate_sha256', 'source_fit',
              'calibration', 'calibration_sha256', 'coverage_audit', 'coverage_audit_sha256',
              'seed', 'steps', 'batch_size', 'eval_every', 'average_steps', 'source_sha256', 'conditions']}
    common['source_sha256'].update({str(p): sha256(p) for p in [Path(__file__), Path('run_staged_base_control.py'),
                                                             audit_path, old_plan_path]})
    old_schedule = root/'inputs/full_model_control_schedule.npz'
    full_schedule = Path(old['schedule'])
    indices = np.load(old_schedule)
    source = json.loads((Path(old['source_fit'])/'run.json').read_text())
    original_metrics = np.load(Path(old['source_fit'])/'metrics.npz')
    assert np.array_equal(indices['task'], original_metrics['step_dataset'])
    for task_index in range(16):
        counts = np.bincount(indices['local_indices'][indices['task'] == task_index].reshape(-1), minlength=512)
        assert np.array_equal(counts, np.full(512, 25))
    budget_scope = ('Matches coefficient-fit plus repair additional updates (3200), labeled exposures (409600), '
                    'sample multiset (118630 unique rows), per-phase class weights and prescribed six-state means. '
                    'First-stage saved task order and per-example multiplicities match; original within-task batch '
                    'order was not logged and is not claimed identical. Stage2 uses the exact existing full-pool '
                    'stream. Expert training and wall-clock/FLOPs are not matched. Rates are two fixed pre-existing '
                    'full-weight rates; this is not an optimal short-Joint recipe or a new method.')
    common['scope'] = budget_scope
    arms, stage1_plans = {}, {}
    design_path = root/'inputs/staged_base_control_design.json'
    for rate, lr in [('lr1e5', 1e-5), ('lr1e4', 1e-4)]:
        stem = f'staged_base_{rate}'
        stage1 = {**copy.deepcopy(common), 'fixed_unix': time.time(), 'lr': lr,
                  'initial_checkpoint': old['initial_checkpoint'], 'initial_checkpoint_sha256': old['initial_checkpoint_sha256'],
                  'initial_reference': str(base_reference_path), 'training_pool': 'original512',
                  'schedule': str(old_schedule), 'schedule_sha256': sha256(old_schedule), 'stage': 1}
        stage1['source_sha256'][str(base_reference_path)] = sha256(base_reference_path)
        stage2 = {**copy.deepcopy(common), 'lr': lr, 'training_pool': 'full_training',
                  'schedule': str(full_schedule), 'schedule_sha256': sha256(full_schedule), 'stage': 2}
        arms[rate] = {'lr': lr, 'stage1_plan': str(root/f'inputs/{stem}_stage1_plan.json'),
                      'stage2_plan': str(root/f'inputs/{stem}_stage2_plan.json'),
                      'stage1_outdir': str(root/f'seed0/{stem}_stage1'),
                      'stage2_outdir': str(root/f'seed0/{stem}_stage2'),
                      'controller_outdir': str(root/f'seed0/{stem}_controller'),
                      'stage1_log': str(root/f'seed0/logs/{stem}_stage1.log'),
                      'stage2_log': str(root/f'seed0/logs/{stem}_stage2.log'),
                      'stage2_reference': str(root/f'inputs/{stem}_stage2_reference.json'),
                      'stage2_plan_template': stage2}
        stage1_plans[rate] = stage1
    control_paths = {'merged_lr1e5': root/'seed0/full_control_coverage_full_balanced_ce/run.json',
                     'merged_lr1e4': root/'seed0/full_control_coverage_lr1e4_balanced_ce/run.json',
                     **{'single_base_'+rate: p for rate, p in existing_base.items()}}
    design = {'fixed_unix': time.time(), 'hypothesis': 'Determine whether the existing merger-plus-repair advantage survives a direct-base comparator given the original coefficient-fitting update and label budget as well.',
              'worker': 'train_base_ce_control.py', 'source_sha256': common['source_sha256'],
              'arms': arms, 'datasets': source['datasets'], 'average_steps': old['average_steps'],
              'base_checkpoint': old['initial_checkpoint'], 'base_checkpoint_sha256': old['initial_checkpoint_sha256'],
              'base_reference': str(base_reference_path), 'base_reference_sha256': sha256(base_reference_path),
              'budget_audit': {'path': str(audit_path), 'sha256': sha256(audit_path)}, 'budget_scope': budget_scope,
              'transition_rule': 'Always export stage1 mean of steps800/960/1120/1280/1440/1600 as stage2 initialization; reset Adam; keep the same LR in both stages. No best-step/rate selection or outcome-dependent skip.',
              'stage2_materialization': 'Only source-stage checkpoint/run/reference paths and hashes, design hash and materialization time are added to the frozen template after successful stage1 checks. Training choices are already fixed.',
              'matched_controls': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in control_paths.items()},
              'primary': 'For both fixed rates, final stage2 fixed-mean full16 validation BAcc versus corresponding merger-plus-repair and single-stage direct-base controls; retain all16 paired task changes and task-deletion sensitivity. Report both stages, costs and calibration metrics.',
              'validation_scope': 'Exposed validation, single bank/order; not independent confirmation.',
              'engineering': 'No worker or optimizer changes. Reuse the existing verified worker and original512/full_training branches; require initial-reference replay, fixed-state/data hashes and ordinary checkpoint export checks in each stage. No new smoke run needed solely for orchestration.',
              'optimizer': 'Adam, weight_decay0, FP32, eval mode, all383 active tensors,1600updates/stage,bs128. No teachers or runtime additions.'}
    write_new(design_path, design)
    for rate, stage1 in stage1_plans.items():
        stage1.update(design=str(design_path), design_sha256=sha256(design_path))
        write_new(Path(arms[rate]['stage1_plan']), stage1)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path),
                      'stage1_plans': {rate: arms[rate]['stage1_plan'] for rate in arms}}, indent=2))


if __name__ == '__main__':
    main()
