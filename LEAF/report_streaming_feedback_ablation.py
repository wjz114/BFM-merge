"""Isolate calibrated-model feedback with a matched FP64 original-only memory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from streaming_original_memory_fp64 import RANK, update_memory
from run_streaming_strict_history import write_json


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert sha256(Path(run['plan'])) == run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == plan['updates_per_arm'] == 1620
    assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert run['history_used_by_merger'] and run['memory_policy'] == run['condition']
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    fixed, previous = set(base) - set(plan['active']), None
    training_seconds, memory_seconds = 0., 0.
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        assert sha256(directory / 'merged.ckpt') == stage['checkpoint_sha256']
        checkpoint = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
        assert checkpoint.keys() == base.keys()
        assert all(torch.equal(checkpoint[name], base[name]) for name in fixed)
        assert all(torch.isfinite(checkpoint[name]).all() for name in plan['active'])
        expert = Path(source['experts'][task])
        assert sha256(expert) == plan['expert_sha256'][task]
        incoming = torch.load(expert, map_location='cpu', weights_only=True)
        if number == 1:
            assert stage['updates'] == stage['training_examples'] == 0
            assert all(torch.equal(checkpoint[name], incoming[name]) for name in base)
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            assert sha256(directory / 'coefficients.pt') == stage['coefficients_sha256']
            c = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
            n = plan['per_arm_coefficients'][run['condition']]
            assert c['snapshots'].shape == (len(plan['average_rounds']), n)
            assert stage['coefficient_count'] == n
            assert torch.isfinite(c['snapshots']).all()
            assert torch.equal(c['snapshots'].double().mean(0).float(), c['mean'])
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(x['max_abs_logit_error'] <= 1e-5 and x['changed_predictions'] == 0
                       for x in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            training_seconds += stage['training_seconds']
            del c
        reconstructed = update_memory(history, base, previous, incoming, checkpoint,
                                      number - 1, plan['active'], run['condition'])
        for key, value in reconstructed.items():
            assert value == stage['memory_update'][key], (folder, number, key)
        if number > 1 and run['condition'] == 'complement_memory':
            assert reconstructed['max_stored_anchor_cosine_abs'] < 1e-6
        memory_seconds += stage['memory_update']['seconds']
        previous = checkpoint
        del incoming
    saved = torch.load(folder / 'history.pt', map_location='cpu', weights_only=True)
    assert saved['count'] == 16
    assert all(torch.equal(value, saved['sketch'][name]) for name, value in history.items())
    assert sum(value.numel() * value.element_size() for value in history.values()) == 206012416
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(item['validation_size'] for item in run['dataset_audits'].values()) == 60413
    costs = {'coefficient_count': plan['per_arm_coefficients'][run['condition']],
             'logical_history_bytes': 206012416, 'memory_update_seconds': memory_seconds,
             'training_and_export_seconds': training_seconds, 'elapsed_seconds': run['elapsed_seconds'],
             'peak_reserved_bytes': run['peak_reserved_bytes']}
    return run, costs



def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_feedback_ablation_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    real_root = Path('runs/streaming_complement_memory_v1')
    real_plan = json.loads((real_root / 'plan.json').read_text())
    real_report = json.loads((real_root / 'comparison.json').read_text())
    assert all(real_report['checks'].values())
    for key in ['seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference',
                'history_update_arithmetic', 'history_direction_scale']:
        assert plan[key] == real_plan[key], key
    for name, digest in real_plan['source_sha256'].items():
        assert plan['source_sha256'][name] == digest
    tasks, orders, sources, versions = plan['tasks'], {}, {}, set()
    for order in plan['orders']:
        folder = root / f'original_memory_{order}'
        random_run, random_cost = audit_run(folder, plan, base, source)
        assert random_run['condition'] == random_run['memory_policy'] == 'original_memory'
        sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        real_path = Path(plan['real_runs'][order]) / 'run.json'
        assert sha256(real_path) == real_report['sources'][str(real_path)]
        real = json.loads(real_path.read_text())
        sources[str(real_path)] = sha256(real_path)
        assert real['condition'] == real['memory_policy'] == 'joint_memory'
        runs = {'joint_memory': real, 'original_memory': random_run}
        costs = {'joint_memory': real_report['orders'][order]['audits_and_costs']['joint_memory'],
                 'original_memory': random_cost}
        for name, ref_folder in plan['reference_runs'][order].items():
            path = Path(ref_folder) / 'run.json'
            assert sha256(path) == real_report['sources'][str(path)]
            runs[name] = json.loads(path.read_text())
            costs[name] = real_report['orders'][order]['audits_and_costs'][name]
            sources[str(path)] = sha256(path)
        for run in runs.values():
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == real['dataset_audits']
            assert run['order'] == real['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in real['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in real['stages'][1:]]
            assert run['stages'][0]['validation'] == real['stages'][0]['validation']
            versions.add((run['torch_version'], run['cuda_version']))
        for stage in (2, 3):
            for filename in ['coefficients.pt', 'merged.ckpt']:
                a = torch.load(real_path.parent / f'stage_{stage:02d}' / filename, map_location='cpu', weights_only=True)
                b = torch.load(folder / f'stage_{stage:02d}' / filename, map_location='cpu', weights_only=True)
                assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        validation = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        controls = ['original_memory', *plan['reference_runs'][order]]
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in tasks])) for name, v in validation.items()},
            'contrasts': {f'joint_memory_minus_{name}': contrast(validation['joint_memory'], validation[name], tasks) for name in controls},
            'primary_gain_over_stages_pp': {str(s): 100 * (real['stages'][s - 1]['macro_BAcc'] - random_run['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']},
            'audits_and_costs': costs}
        print('checked', order, flush=True)
    assert len(versions) == 1
    keys = list(orders['forward']['contrasts'])
    result = {'sources': sources, 'orders': orders,
              'mean_gain_over_two_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys},
              'checks': {'both_new_runs_complete': True, 'strict_settings_recorded': True,
                         'actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'stages2_and3_before_feedback_intervention_exact_coefficients_and_native_weights': True,
                         'new_checkpoint_and_coefficient_hashes': True, 'new_fixed_coefficient_means_exact': True,
                         'new_native_exports_pass': True, 'new_original_memory_lineages_replay_exactly': True,
                         'frozen_real_and_strict_reference_audits_match': True},
              'new_scientific_updates': 3240, 'new_scientific_CE_example_passes': 51840,
              'extra_score_VJPs': 0,
              'scope': 'Two new original-only memory controls, selected after bank0 orientation and bank1 replication. Reused joint memory has the identical FP64 four-slot compression, floor, zero first memory and stage2 update. Only count>1 calibrated-model feedback is removed. Stages2 and3 coefficients and native weights match exactly before feedback changes subsequent calibration. Reused real runs and strict references retain frozen full audits. This isolates the feedback policy within this implementation, not novelty or universal benefit. Exposed development validation and growing original512 replay; no official test read. Task deletion is sensitivity, not confidence intervals. One worker per GPU now versus two in the reused joint runs; elapsed times are descriptive.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    names = ['incoming_rows', 'old_rows', 'history_sketch', 'random_sketch', 'joint_memory', 'original_memory']
    labels = ['Incoming\nrows', 'Bilateral\nrows', 'Raw\nhistory', 'Raw random\nhistory', 'Joint\nmemory', 'Original-only\nFP64']
    for j, order in enumerate(plan['orders']):
        axes[0].scatter(np.arange(6) + (j - .5) * .15, [100 * orders[order]['macro_BAcc'][n] for n in names], s=65, label=order)
    axes[0].set_xticks(np.arange(6), labels, fontsize=8)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    heat = np.asarray([[orders[o]['contrasts']['joint_memory_minus_original_memory']['per_task_pp'][t] for o in plan['orders']] for t in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Joint memory minus matched original-only history')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Calibrated-model feedback ablation — strict bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
