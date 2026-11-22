"""Audit all eight strict comparisons and report every method, order and task."""
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
from streaming_history_sketch_merge import RANK, update_sketch


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert sha256(Path(run['plan'])) == run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == plan['updates_per_arm'] == 1620
    assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert run['history_used_by_merger'] == plan['history_used_by_condition'][run['condition']]
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    fixed = set(base) - set(plan['active'])
    training_seconds, sketch_seconds = 0., 0.
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
            coefficients = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
            n = plan['per_arm_coefficients'][run['condition']]
            assert coefficients['snapshots'].shape == (len(plan['average_rounds']), n)
            assert stage['coefficient_count'] == n
            assert torch.isfinite(coefficients['snapshots']).all()
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(x['max_abs_logit_error'] <= 1e-5 and x['changed_predictions'] == 0
                       for x in stage['export_checks'].values())
            training_seconds += stage['training_seconds']
            del coefficients
        reconstructed = update_sketch(history, incoming, base, plan['active'])
        for key, value in reconstructed.items():
            assert value == stage['history_sketch_update'][key], (folder, number, key)
        sketch_seconds += stage['history_sketch_update']['seconds']
        del incoming, checkpoint
    saved = torch.load(folder / 'history.pt', map_location='cpu', weights_only=True)
    assert saved['count'] == 16
    assert all(torch.equal(value, saved['sketch'][name]) for name, value in history.items())
    assert sum(value.numel() * value.element_size() for value in history.values()) == 206012416
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(item['validation_size'] for item in run['dataset_audits'].values()) == 60413
    costs = {'coefficient_count': plan['per_arm_coefficients'][run['condition']],
             'history_used_by_merger': run['history_used_by_merger'],
             'logical_history_bytes': 206012416 if run['history_used_by_merger'] else 0,
             'raw_history_audit_bytes': 206012416, 'history_update_seconds': sketch_seconds,
             'training_and_export_seconds': training_seconds, 'elapsed_seconds': run['elapsed_seconds'],
             'peak_reserved_bytes': run['peak_reserved_bytes']}
    return run, costs, history


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_strict_history_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    tasks, orders, sources = plan['tasks'], {}, {}
    pairs = [('history_sketch', name) for name in ('incoming_rows', 'old_rows', 'random_sketch')]
    pairs.append(('old_rows', 'incoming_rows'))
    versions = set()
    for order in plan['orders']:
        runs, costs, reference_history = {}, {}, None
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run, cost, history = audit_run(folder, plan, base, source)
            assert run['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            versions.add((run['torch_version'], run['cuda_version']))
            if reference_history is None:
                reference_history = history
            else:
                assert all(torch.equal(value, reference_history[name]) for name, value in history.items())
            runs[condition], costs[condition] = run, cost
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            print('checked', order, condition, flush=True)
        reference = runs['incoming_rows']
        historical = {}
        for condition, folder in plan['historical_runs'][order].items():
            path = Path(folder) / 'run.json'
            historical[condition] = json.loads(path.read_text())
            assert historical[condition]['status'] == 'completed'
            sources[str(path)] = sha256(path)
        for run in [*runs.values(), *historical.values()]:
            assert run['dataset_audits'] == reference['dataset_audits']
            assert run['order'] == reference['order']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
        for run in runs.values():
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
        validation = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([m[t]['BAcc'] for t in tasks])) for name, m in validation.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(validation[a], validation[b], tasks) for a, b in pairs},
            'history_gain_over_stages_pp': {name: {str(stage): 100 * (runs['history_sketch']['stages'][stage - 1]['macro_BAcc'] - runs[name]['stages'][stage - 1]['macro_BAcc']) for stage in plan['validation_stages']} for name in ('incoming_rows', 'old_rows', 'random_sketch')},
            'strict_minus_earlier_default_pp': {name: contrast(validation[name], old['stages'][-1]['validation'], tasks) for name, old in historical.items()},
            'first_expert_validation_equal_to_earlier_default': {name: old['stages'][0]['validation'] == reference['stages'][0]['validation'] for name, old in historical.items()},
            'audits_and_costs': costs}
    assert len(versions) == 1
    keys = list(orders['forward']['contrasts'])
    result = {'sources': sources, 'orders': orders,
              'mean_gain_over_two_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys},
              'checks': {'all_eight_complete': True, 'strict_settings_recorded': True,
                         'actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'checkpoint_and_coefficient_hashes': True, 'fixed_coefficient_means_exact': True,
                         'native_exports_pass': True, 'raw_history_lineages_replay_exactly': True,
                         'raw_history_matches_across_all_four_conditions': True},
              'new_scientific_updates': 12960, 'new_scientific_CE_example_passes': 207360,
              'extra_score_VJPs': 0,
              'scope': 'Previously selected candidate, one bank, two correlated orders on exposed development validation. Task deletion is sensitivity, not confidence intervals. Strict settings are recorded; this is not a replicated full-stream reproducibility estimate. Historical default results are context rather than fresh paired controls. Random directions match capacity and residual norms within each evolving model, not necessarily norms across evolving arms. Structural baselines also construct unused history for auditing; their logical history requirement is zero. Replay grows with tasks. Four workers share hardware, so times are descriptive. No official test read.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    names = plan['conditions']
    labels = ['Incoming\nrows', 'Bilateral\nrows', 'Real\nhistory', 'Random\nhistory']
    x = np.arange(len(names))
    for j, order in enumerate(plan['orders']):
        axes[0].scatter(x + (j - .5) * .12, [100 * orders[order]['macro_BAcc'][name] for name in names], s=65, label=order)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(base_name, order) for base_name in ('incoming_rows', 'old_rows', 'random_sketch') for order in plan['orders']]
    heat = np.asarray([[orders[order]['contrasts'][f'history_sketch_minus_{base_name}']['per_task_pp'][task] for base_name, order in columns] for task in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(np.arange(6), ['Incoming\nfwd', 'Incoming\nrev', 'Bilateral\nfwd', 'Bilateral\nrev', 'Random\nfwd', 'Random\nrev'], fontsize=9)
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Real history minus each control')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Strict computation: bounded-history merger and structural controls — bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
