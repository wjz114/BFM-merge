"""Separate reuse-time adaptation from the recorded cost of building the expert bank."""
import json
import math
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    fit_path = root/'seed0/original_fp32_tensor_balanced_ce/run.json'
    mean_path = root/'seed0/original_fp32_tensor_balanced_avg/run.json'
    joint_path = Path('runs/poc16/seed0/joint/run.json')
    fit, averaged, joint = [json.loads(p.read_text()) for p in [fit_path, mean_path, joint_path]]
    hashes = {str(p): sha256(p) for p in [fit_path, mean_path, joint_path, Path('d_downstream_qformer_ce.py')]}
    experts = {}
    for task in fit['datasets']:
        checkpoint = Path(fit['experts'][task])
        path = checkpoint.with_name('run.json')
        run = json.loads(path.read_text())
        hashes[str(path)] = sha256(path)
        hashes[str(checkpoint)] = sha256(checkpoint)
        assert run['datasets'] == [task]
        assert run['source_checkpoint_sha256'] == fit['base_checkpoint_sha256']
        assert run['actual_optimizer_steps'] == run['steps_per_dataset_actual'][task] == 3000
        n, batch, steps = run['split_sizes'][task]['train'], run['batch_size'], run['actual_optimizer_steps']
        batches_per_pass = math.ceil(n/batch)
        implied_exposures = (steps//batches_per_pass)*n+(steps % batches_per_pass)*batch
        experts[task] = {'run': str(path), 'checkpoint': str(checkpoint), 'actual_optimizer_steps': steps,
                         'selected_checkpoint_step': run['best_step'], 'batch_size': batch,
                         'training_pool_size': n, 'nominal_full_batch_exposures_upper_bound': steps*batch,
                         'exposures_implied_by_nondropping_loader': implied_exposures,
                         'precision': run['precision'], 'seconds': run['seconds']}
    assert joint['source_checkpoint_sha256'] == fit['base_checkpoint_sha256']
    assert joint['actual_optimizer_steps'] == 48000
    assert joint['datasets'] == fit['datasets']
    for task in fit['datasets']:
        assert joint['steps_per_dataset_actual'][task] == experts[task]['actual_optimizer_steps']
        assert joint['split_sizes'][task]['train'] == experts[task]['training_pool_size']
    repairs = {}
    for rate, folder in [('lr1e5', 'full_control_coverage_full_balanced_ce'),
                         ('lr1e4', 'full_control_coverage_lr1e4_balanced_ce')]:
        path = root/f'seed0/{folder}/run.json'
        run = json.loads(path.read_text())
        hashes[str(path)] = sha256(path)
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        repairs[rate] = {'path': str(path), 'updates': run['updates'],
                          'training_exposures': run['training_examples'], 'seconds': run['seconds'],
                          'recorded_reuse_stage_process_seconds': averaged['seconds']['total']+run['seconds']['total']}
    sum_steps = sum(e['actual_optimizer_steps'] for e in experts.values())
    sum_seconds = {k: sum(e['seconds'][k] for e in experts.values()) for k in joint['seconds']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'experts': experts, 'expert_bank_total_recorded_updates': sum_steps,
              'expert_bank_sum_process_seconds': sum_seconds,
              'expert_bank_nominal_label_exposures_upper_bound': sum(e['nominal_full_batch_exposures_upper_bound'] for e in experts.values()),
              'expert_bank_loader_implied_label_exposures': sum(e['exposures_implied_by_nondropping_loader'] for e in experts.values()),
              'coefficient_fit': {'updates': fit['alpha_updates_actual'],
                                  'training_exposures': fit['alpha_updates_actual']*fit['batch_size'],
                                  'process_seconds': fit['seconds'], 'fit_and_mean_seconds': averaged['seconds']},
              'repair': repairs,
              'joint': {'path': str(joint_path), 'updates': joint['actual_optimizer_steps'],
                        'steps_per_task': joint['steps_per_dataset_actual'], 'seconds': joint['seconds'],
                        'precision': joint['precision']},
              'recorded_updates_from_released_base': {'expert_bank_then_coefficients_then_repair': sum_steps+1600+1600,
                                                    'joint': joint['actual_optimizer_steps'], 'staged_direct_base': 3200},
              'distinctions': ['The staged control answers the marginal adaptation comparison when specialists already exist; it does not recreate specialist knowledge with equal historical compute.',
                              'Expert-bank construction and Joint each record48000optimizer updates. Bank construction plus coefficient fit and repair totals51200recorded updates. Equal update counts are not FLOPs, precision or wall-clock equivalence.',
                              'Summed process durations include setup/validation and count concurrent expert jobs separately; they are not end-to-end elapsed time or hardware-normalized compute.',
                              'Selected specialist checkpoints can precede step3000, but recorded training still ran3000steps each. Retain the full incurred cost.',
                              'Exact expert per-batch IDs/sizes were not logged. The loader-implied exposure count assumes the documented current non-dropping BatchStream semantics and saved train sizes; it is distinguished from the nominal full-batch upper bound.',
                              'Both pipelines start from the same released pretrained base; its earlier pretraining cost is outside this local accounting.'],
              'new_training_or_inference': False}
    path = root/'analysis/merge_pipeline_budget.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['expert_bank_total_recorded_updates', 'expert_bank_sum_process_seconds',
                                            'expert_bank_nominal_label_exposures_upper_bound',
                                            'expert_bank_loader_implied_label_exposures', 'recorded_updates_from_released_base']}, indent=2))


if __name__ == '__main__':
    main()
