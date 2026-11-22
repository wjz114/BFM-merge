"""Audit the fixed history-selection comparison and report every task/order."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_history_sketch_merge import write_json
from streaming_history_sketch_merge import RANK
from streaming_response_sketch import update_sketch


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == len(plan['tasks']) == 16
    assert sha256(Path(run['plan'])) == run['plan_sha256']
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == plan['updates_per_arm'] == 1620
    assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    fixed = set(base) - set(plan['active'])
    calls, training_seconds, sketch_seconds, zero_rows = [], 0., 0., []
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        stage_dir = folder / f'stage_{number:02d}'
        assert sha256(stage_dir / 'merged.ckpt') == stage['checkpoint_sha256']
        checkpoint = torch.load(stage_dir / 'merged.ckpt', map_location='cpu', weights_only=True)
        assert checkpoint.keys() == base.keys()
        assert all(torch.equal(checkpoint[name], base[name]) for name in fixed)
        assert all(torch.isfinite(checkpoint[name]).all() for name in plan['active'])
        expert_path = Path(source['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        incoming = torch.load(expert_path, map_location='cpu', weights_only=True)
        statistics = None
        if number == 1:
            assert all(torch.equal(checkpoint[name], incoming[name]) for name in base)
            assert stage['updates'] == stage['training_examples'] == 0
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            for filename, key in [('coefficients.pt', 'coefficients_sha256'),
                                   ('responses.pt', 'response_statistics_sha256')]:
                assert sha256(stage_dir / filename) == stage[key]
            coefficients = torch.load(stage_dir / 'coefficients.pt', map_location='cpu', weights_only=True)
            assert coefficients['snapshots'].shape == (len(plan['average_rounds']), 694566)
            assert torch.isfinite(coefficients['snapshots']).all()
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert stage['coefficient_count'] == coefficients['mean'].numel() == 694566
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                       for v in stage['export_checks'].values())
            statistics = torch.load(stage_dir / 'responses.pt', map_location='cpu', weights_only=True)
            assert statistics['hook_calls'] == stage['hook_calls']
            assert set(statistics['covariance']) == set(plan['active'])
            for name, covariance in statistics['covariance'].items():
                rows = base[name].shape[0] if base[name].ndim else 1
                assert covariance.shape == (rows, 3, 3) and covariance.dtype == torch.float64
                assert torch.isfinite(covariance).all()
                assert torch.equal(covariance, covariance.transpose(-1, -2))
                assert (covariance.diagonal(dim1=-2, dim2=-1) >= 0).all()
                assert 0 <= statistics['hook_calls'][name] <= stage['updates']
            assert sum(x.numel() * x.element_size() for x in statistics['covariance'].values()) == 12488400
            assert statistics['appended_dictionary_bytes'] == stage['appended_dictionary_bytes'] == 309018624
            assert statistics['response_covariance_bytes'] == stage['response_covariance_bytes'] == 12488400
            check = stage['hook_gradient_check']
            assert check['max_abs_response_vs_gate_gradient'] < 1e-6 and check['relative_l2_error'] < 2e-5
            calls.append({'stage': number, 'min': min(statistics['hook_calls'].values()),
                          'max': max(statistics['hook_calls'].values())})
            training_seconds += stage['training_seconds']
            del coefficients
        # Replay the complete memory lineage using its actual saved probes.
        reconstructed = update_sketch(history, incoming, base, plan['active'], run['condition'], statistics)
        for key, value in reconstructed.items():
            actual = stage['history_sketch_update'][key]
            if isinstance(value, float):
                assert np.isclose(value, actual, rtol=1e-7, atol=1e-20), (number, key, value, actual)
            else:
                assert value == actual
        zero_rows.append(reconstructed.get('zero_response_rows'))
        sketch_seconds += stage['history_sketch_update']['seconds']
        del incoming, checkpoint, statistics
    saved = torch.load(folder / 'history.pt', map_location='cpu', weights_only=True)
    assert saved['count'] == 16
    assert all(torch.equal(value, saved['sketch'][name]) for name, value in history.items())
    assert sum(value.numel() * value.element_size() for value in history.values()) == 206012416
    assert sum(stage['input_split']['extra_forward_examples'] for stage in run['stages'][1:]) == 1920
    assert sum(item['validation_size'] for item in run['dataset_audits'].values()) == 60413
    if run['condition'] == 'energy_history':
        order = 'forward' if run['order'] == plan['tasks'] else 'reverse'
        original = torch.load(Path(plan['baseline_runs'][order]['coefficient_history']) / 'history.pt',
                              map_location='cpu', weights_only=True)
        assert all(torch.equal(value, original['sketch'][name]) for name, value in history.items())
    return run, {'raw_history_replay_exact': True, 'hook_calls_by_stage': calls,
                 'zero_response_rows_by_stage': zero_rows, 'training_and_export_seconds': training_seconds,
                 'history_update_seconds': sketch_seconds, 'elapsed_seconds': run['elapsed_seconds'],
                 'peak_reserved_bytes': run['peak_reserved_bytes']}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_response_sketch_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    tasks, sources, orders = plan['tasks'], {}, {}
    for order in plan['orders']:
        runs, audits = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run, audit = audit_run(folder, plan, base, source)
            assert run['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            runs[condition], audits[condition] = run, audit
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            print(f'checked {order} {condition}', flush=True)
        for name, folder in plan['baseline_runs'][order].items():
            path = Path(folder) / 'run.json'
            runs[name] = json.loads(path.read_text())
            assert runs[name]['status'] == 'completed' and len(runs[name]['stages']) == 16
            sources[str(path)] = sha256(path)
        reference = runs['energy_history']
        for run in runs.values():
            assert run['dataset_audits'] == reference['dataset_audits']
            assert run['order'] == reference['order']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
        validation = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        pairs = [('response_history', 'energy_history'), ('response_history', 'incoming_rows'),
                 ('response_history', 'coefficient_history'), ('energy_history', 'coefficient_history'),
                 ('response_history', 'bilateral')]
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([metrics[t]['BAcc'] for t in tasks])) for name, metrics in validation.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(validation[a], validation[b], tasks) for a, b in pairs},
            'primary_gain_over_stages_pp': {str(stage): 100 * (runs['response_history']['stages'][stage - 1]['macro_BAcc'] - reference['stages'][stage - 1]['macro_BAcc']) for stage in plan['validation_stages']},
            'audits_and_costs': audits}
    keys = list(orders['forward']['contrasts'])
    result = {'sources': sources, 'orders': orders,
              'mean_gain_over_two_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys},
              'checks': {'actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'checkpoint_and_statistic_hashes': True, 'fixed_coefficient_means_exact': True,
                         'native_exports_pass': True, 'saved_response_covariances_finite_symmetric': True,
                         'raw_history_lineages_replay_exactly': True, 'energy_history_matches_original_history_exactly': True},
              'new_scientific_updates': 4 * 1620, 'new_scientific_training_examples': 4 * 25920,
              'scope': 'One expert bank and two correlated arrival orders on exposed development validation. Task deletion is sensitivity, not confidence intervals. The history-selection rule changes; both fresh arms collect the same kind of trajectory loss responses, whose values depend on their evolving models. They need not have identical covariance or history tensors. This is not predictive Fisher or an output-preservation guarantee. Replay grows with task count. Additional dictionary/covariance computation and training storage are not free; shared-hardware times are descriptive. No official test read by these runs.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.5, 1]})
    names = ['incoming_rows', 'bilateral', 'coefficient_history', 'energy_history', 'response_history']
    labels = ['Incoming\nrows', 'Bilateral\ncoeff.', 'Original\nhistory', 'Fresh energy\nhistory', 'Response\nhistory']
    x = np.arange(len(names))
    for j, order in enumerate(plan['orders']):
        axes[0].bar(x + (j - .5) * .36, [100 * orders[order]['macro_BAcc'][name] for name in names], .36, label=order)
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    heat = np.asarray([[orders[order]['contrasts']['response_history_minus_energy_history']['per_task_pp'][task] for order in plan['orders']] for task in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks([0, 1], plan['orders'])
    axes[1].set_yticks(np.arange(16), [task.replace('MI_', '').replace('EMO_', '') for task in tasks], fontsize=9)
    axes[1].set_title('Response minus energy history')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Streaming history selection from existing calibration gradients — bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
