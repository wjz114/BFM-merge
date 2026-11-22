"""Audit and compare fixed native-CE propagation rules and prior baselines."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_subspace_merge import write_json


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_correction_retention_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    active, tasks = plan['active'], plan['tasks']
    reports, sources = {}, {}
    for order in ('forward', 'reverse'):
        expected_order = tasks if order == 'forward' else list(reversed(tasks))
        runs, costs = {}, {}
        stage2 = {}
        for condition in ('recursive', 'retain_correction'):
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['order'] == expected_order and run['condition'] == condition
            assert run['updates'] == plan['updates_per_arm'] == 1620
            assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
            assert run['official_test_read'] is False
            for stage in run['stages']:
                stage_dir = folder / f"stage_{stage['stage']:02d}"
                assert sha256(stage_dir / 'merged.ckpt') == stage['checkpoint_sha256']
                state = torch.load(stage_dir / 'merged.ckpt', map_location='cpu', weights_only=True)
                assert state.keys() == base.keys()
                assert all(torch.equal(state[n], base[n]) for n in base if n not in active)
                if stage['stage'] > 1:
                    assert sha256(stage_dir / 'snapshots.pt') == stage['snapshots_sha256']
                    saved = torch.load(stage_dir / 'snapshots.pt', map_location='cpu', weights_only=True)
                    assert saved['rounds'] == plan['average_rounds']
                    snapshots = saved['snapshots']
                    assert len(snapshots) == 6 and all(set(s) == set(active) for s in snapshots)
                    for name in active:
                        averaged = torch.stack([s[name].double() for s in snapshots]).mean(0).float()
                        assert torch.equal(averaged, state[name]), (order, condition, stage['stage'], name)
                    del saved, snapshots
                    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                               for v in stage['export_checks'].values())
                if stage['stage'] == 2:
                    stage2[condition] = state
            if condition == 'retain_correction':
                history = torch.load(folder / 'original_sum.pt', map_location='cpu', weights_only=True)
                assert history['count'] == 16 and set(history['original_sum']) == set(active)
                expected = {n: torch.zeros_like(base[n], dtype=torch.float64) for n in active}
                for task in expected_order:
                    path = Path(source['experts'][task])
                    assert sha256(path) == plan['expert_sha256'][task]
                    original = torch.load(path, map_location='cpu', weights_only=True)
                    for name in active:
                        expected[name].add_(original[name].double())
                assert all(torch.equal(expected[n], history['original_sum'][n]) for n in active)
                history_bytes = sum(v.numel() * v.element_size() for v in history['original_sum'].values())
                assert history_bytes == plan['persistent_original_sum_bytes']
                del history, expected, original
            else:
                history_bytes = 0
                assert not (folder / 'original_sum.pt').exists()
            assert run['persistent_original_sum_bytes'] == history_bytes
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            runs[condition] = run
            costs[condition] = {'elapsed_seconds': run['elapsed_seconds'],
                                'training_seconds': sum(s.get('training_seconds', 0) for s in run['stages']),
                                'peak_reserved_bytes': run['peak_reserved_bytes'],
                                'persistent_original_sum_bytes': history_bytes,
                                'trainable_parameters': run['stages'][-1]['trainable_parameters']}
            print(f'checked {order} {condition}', flush=True)
        for condition, dirname in [('incoming_rows', 'streaming_subspace_v1'),
                                   ('history_sketch', 'streaming_history_sketch_v1')]:
            path = Path('runs') / dirname / f'{condition}_{order}' / 'run.json'
            run = json.loads(path.read_text())
            assert run['status'] == 'completed' and run['order'] == expected_order
            sources[str(path)] = sha256(path)
            runs[condition] = run
        baseline = runs['recursive']
        for run in runs.values():
            assert run['dataset_audits'] == baseline['dataset_audits']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in baseline['stages']]
            assert run['stages'][0]['validation'] == baseline['stages'][0]['validation']
        validation = {c: r['stages'][-1]['validation'] for c, r in runs.items()}
        comparisons = [('retain_correction', 'recursive'), ('recursive', 'incoming_rows'),
                       ('retain_correction', 'incoming_rows'), ('retain_correction', 'history_sketch')]
        trajectory = {}
        for stage in plan['validation_stages']:
            seen = expected_order[:stage]
            trajectory[str(stage)] = contrast(runs['retain_correction']['stages'][stage - 1]['validation'],
                                              baseline['stages'][stage - 1]['validation'], seen) if stage > 1 else {'gain_pp': 0.0}
        reports[order] = {
            'macro_BAcc': {c: float(np.mean([v[t]['BAcc'] for t in tasks])) for c, v in validation.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(validation[a], validation[b], tasks) for a, b in comparisons},
            'primary_gain_over_stages': trajectory, 'cost': costs,
            'stage2_parameter_repeatability_max_abs_error': max(
                float((stage2['retain_correction'][n] - stage2['recursive'][n]).abs().max()) for n in active)}
    result = {'sources': sources, 'orders': reports,
              'mean_primary_gain_over_two_orders_pp': float(np.mean([
                  r['contrasts']['retain_correction_minus_recursive']['gain_pp'] for r in reports.values()])),
              'checks': {'checkpoint_hashes': True, 'all_fixed_parameter_means_exact': True,
                         'native_reload_probes_pass': True, 'original_sum_exact_from_original_experts': True,
                         'paired_data_and_training_schedules': True},
              'new_scientific_updates': 6480, 'new_scientific_training_examples': 103680,
              'scope': 'One bank and two correlated fixed orders on exposed development validation. '
                       'Primary contrast changes only propagation, with identical native CE rate, steps, '
                       'batches and averaging. Coefficient baselines have the same CE schedule but different '
                       'parameterization, learning rate and auxiliary-probe costs. Floating-point execution '
                       'is not claimed bitwise reproducible. Snapshot storage is an audit artifact, not '
                       'persistent algorithmic state. Task-deletion ranges are sensitivity, not CIs. '
                       'Native CE and mean/residual separation are not claimed novel. No official-test data.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.3]})
    labels = ['incoming_rows', 'history_sketch', 'recursive', 'retain_correction']
    for j, order in enumerate(('forward', 'reverse')):
        offset = (j - .5) * .36
        values = [100 * reports[order]['macro_BAcc'][k] for k in labels]
        axes[0].bar(np.arange(4) + offset, values, width=.36, label=order)
        gains = reports[order]['contrasts']['retain_correction_minus_recursive']['per_task_pp']
        axes[1].barh(np.arange(16) + offset, [gains[t] for t in tasks], height=.36, label=order)
    axes[0].set_xticks(np.arange(4), ['Incoming\nrows', 'History\nsketch', 'Native CE\nrecursive', 'Native CE\nretain correction'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', lw=.8)
    axes[1].set_xlabel('Retain correction minus recursive native CE (pp)')
    axes[1].legend()
    fig.suptitle('Calibration-correction propagation — bank0, fixed native CE budget')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({order: {'macro_BAcc': r['macro_BAcc'], 'primary_gain_pp':
          r['contrasts']['retain_correction_minus_recursive']['gain_pp']} for order, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
