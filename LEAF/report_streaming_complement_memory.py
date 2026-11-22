"""Audit coupled-memory lineages and compare both policies with strict references."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_strict_history import write_json
from streaming_complement_memory import RANK, update_memory


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
    root = Path('runs/streaming_complement_memory_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    strict_root = Path('runs/streaming_strict_history_v1')
    strict_plan = json.loads((strict_root / 'plan.json').read_text())
    strict_report = json.loads((strict_root / 'comparison.json').read_text())
    assert all(strict_report['checks'].values())
    for key in ['seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference']:
        assert plan[key] == strict_plan[key], key
    for name, digest in strict_plan['source_sha256'].items():
        assert plan['source_sha256'][name] == digest
    tasks, orders, sources, versions = plan['tasks'], {}, {}, set()
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run, cost = audit_run(folder, plan, base, source)
            runs[condition], costs[condition] = run, cost
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            print('checked', order, condition, flush=True)
        for condition, folder in plan['reference_runs'][order].items():
            path = Path(folder) / 'run.json'
            assert sha256(path) == strict_report['sources'][str(path)]
            runs[condition] = json.loads(path.read_text())
            costs[condition] = strict_report['orders'][order]['audits_and_costs'][condition]
            sources[str(path)] = sha256(path)
        reference = runs['incoming_rows']
        for run in runs.values():
            assert run['status'] == 'completed'
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert run['order'] == reference['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
            versions.add((run['torch_version'], run['cuda_version']))
        first = root / f'joint_memory_{order}' / 'stage_02'
        second = root / f'complement_memory_{order}' / 'stage_02'
        for filename in ['coefficients.pt', 'merged.ckpt']:
            a = torch.load(first / filename, map_location='cpu', weights_only=True)
            b = torch.load(second / filename, map_location='cpu', weights_only=True)
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        validation = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        controls = ['joint_memory', *plan['reference_runs'][order]]
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in tasks])) for name, v in validation.items()},
            'contrasts': {f'complement_memory_minus_{name}': contrast(validation['complement_memory'], validation[name], tasks) for name in controls},
            'gain_over_stages_pp': {name: {str(stage): 100 * (runs['complement_memory']['stages'][stage - 1]['macro_BAcc'] - runs[name]['stages'][stage - 1]['macro_BAcc']) for stage in plan['validation_stages']} for name in controls},
            'memory_geometry_by_stage': {name: {str(s['stage']): s['memory_update'] for s in runs[name]['stages']} for name in plan['conditions']},
            'audits_and_costs': costs}
    assert len(versions) == 1
    keys = list(orders['forward']['contrasts'])
    result = {'sources': sources, 'orders': orders,
              'mean_gain_over_two_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys},
              'checks': {'all_four_complete': True, 'strict_settings_recorded': True,
                         'actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'stage2_before_policy_intervention_exact_coefficients_and_native_weights': True,
                         'checkpoint_and_coefficient_hashes': True, 'fixed_coefficient_means_exact': True,
                         'native_exports_pass': True, 'calibrated_state_memory_lineages_replay_exactly': True,
                         'stored_complement_memory_orthogonal_to_current_anchor': True,
                         'strict_reference_protocol_and_prior_audit_hashes_match': True},
              'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
              'extra_score_VJPs': 0,
              'scope': 'One bank, two correlated orders on exposed development validation; not independent seed replication. Task deletion is sensitivity, not confidence intervals. The eight strict references are reused with their completed audit and frozen hashes. Both new memory policies use the same capacity and shared calibration implementation. Memory depends on each evolving model: geometry after stage2 is not measured on identical inputs across conditions. Orthogonality and a toy span property do not prove functional or accuracy preservation. Replay grows with tasks. Shared-hardware timing is descriptive. No official test read.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.15, 1.5]})
    names = ['incoming_rows', 'old_rows', 'history_sketch', 'random_sketch', 'joint_memory', 'complement_memory']
    labels = ['Incoming\nrows', 'Bilateral\nrows', 'Raw\nhistory', 'Random\nhistory', 'Joint\nmemory', 'Complement\nmemory']
    x = np.arange(len(names))
    for j, order in enumerate(plan['orders']):
        axes[0].scatter(x + (j - .5) * .15, [100 * orders[order]['macro_BAcc'][name] for name in names], s=65, label=order)
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(name, order) for name in ['joint_memory', 'old_rows', 'history_sketch'] for order in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][f'complement_memory_minus_{name}']['per_task_pp'][t] for name, o in columns] for t in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(np.arange(6), ['Joint\nfwd', 'Joint\nrev', 'Bilateral\nfwd', 'Bilateral\nrev', 'Raw history\nfwd', 'Raw history\nrev'], fontsize=9)
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Complement memory minus control')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Current model plus bounded memory — strict computation, bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
