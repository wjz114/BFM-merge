"""Audit transported versus restarted moment calibration against strict references."""
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
from aligned_support import array_hash


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(Path(run['plan'])) == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    assert run['condition'] in plan['conditions']
    assert run['history_used_by_merger'] and run['memory_policy'] == 'additive_original_expert_fisher_moments'
    assert run['fisher_payload_forward_examples'] == 2048 and run['fisher_payload_backward_probes'] == 128
    assert run['new_fisher_estimation_examples'] == run['new_fisher_estimation_backward_probes'] == 0
    cache = json.loads(Path(plan['fisher_cache']).read_text())
    fixed = set(base) - set(plan['active'])
    expected_moments = {n: torch.zeros((3, *base[n].shape), dtype=torch.float64) for n in plan['active']}
    previous_mean = None
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        payload = cache['experts'][task]
        assert stage['fisher_arrival'] == {'task': task, 'path': payload['path'], 'sha256': payload['sha256'],
            'forward_examples': payload['forward_examples'], 'backward_probes': payload['backward_probes']}
        assert sha256(Path(payload['path'])) == payload['sha256']
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task] == payload['expert_sha256']
        for key, value in payload['calibration_audit'].items():
            assert run['dataset_audits'][task][key] == value
        original = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        fisher = torch.load(payload['path'], map_location='cpu', weights_only=True)
        assert set(fisher) == set(plan['active'])
        for name in plan['active']:
            assert torch.isfinite(fisher[name]).all() and (fisher[name] >= 0).all()
            delta = (original[name] - base[name]).double()
            precision = fisher[name].double()
            expected_moments[name][0].add_(precision)
            expected_moments[name][1].add_(precision * delta)
            expected_moments[name][2].add_(delta)
        del fisher
        directory = folder / f'stage_{number:02d}'
        checkpoint_path = directory / 'merged.ckpt'
        assert checkpoint_path.is_symlink() and checkpoint_path.resolve() == Path(stage['checkpoint_storage_path'])
        assert sha256(checkpoint_path) == stage['checkpoint_sha256']
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[name], base[name]) for name in fixed)
        assert all(torch.isfinite(state[name]).all() for name in plan['active'])
        assert stage['moment_bytes'] == 618037248
        expected_retained = 1390664 if run['condition'] == 'moment_transport' and number > 1 else 0
        assert stage['retained_coefficient_bytes'] == expected_retained
        if number == 1:
            assert all(torch.equal(state[name], original[name]) for name in state)
            assert stage['updates'] == stage['training_examples'] == 0
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            coefficient_path = directory / 'coefficients.pt'
            assert coefficient_path.is_symlink()
            assert sha256(coefficient_path) == stage['coefficients_sha256']
            coefficients = torch.load(coefficient_path, map_location='cpu', weights_only=True)
            assert coefficients['snapshots'].shape == (6, 347666)
            assert coefficients['initial'].shape == coefficients['mean'].shape == (347666,)
            assert stage['coefficient_count'] == 347666
            assert all(torch.isfinite(v).all() for v in coefficients.values())
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            carry = run['condition'] == 'moment_transport' and number > 2
            assert stage['initialization_from_previous_coefficients'] == carry
            if carry:
                assert torch.equal(coefficients['initial'], previous_mean)
            else:
                default = torch.zeros_like(coefficients['initial'])
                default[:len(plan['active'])] = 1.
                assert torch.equal(coefficients['initial'], default)
            assert array_hash(coefficients['initial'].numpy()) == stage['initial_coefficient_hash']
            previous_mean = coefficients['mean']
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0 for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
        del original, state
    moment_path = folder / 'moments.pt'
    assert moment_path.is_symlink() and moment_path.resolve() == Path(run['stages'][-1]['checkpoint_storage_path']).parent.parent / 'moments.pt'
    assert sha256(moment_path) == run['moments_sha256']
    stored = torch.load(moment_path, map_location='cpu', weights_only=True)
    assert stored['count'] == 16 and stored['moments'].keys() == expected_moments.keys()
    assert all(torch.equal(stored['moments'][n], expected_moments[n]) for n in expected_moments)
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(x['validation_size'] for x in run['dataset_audits'].values()) == 60413
    costs = {'coefficient_count': 347666, 'moment_bytes': 618037248,
             'algorithmic_retained_coefficient_bytes': run['stages'][-1]['retained_coefficient_bytes'],
             'implementation_previous_coefficient_buffer_bytes': 1390664,
             'buffer_note': 'Both runners retain the previous vector for the common interface/audit; only transport uses it to initialize calibration.',
             'moment_update_seconds': sum(s['moment_update_seconds'] for s in run['stages']),
             'training_and_export_seconds': sum(s['training_seconds'] for s in run['stages'][1:]),
             'calibration_including_setup_seconds': sum(s['calibration_call_seconds'] for s in run['stages'][1:]),
             'elapsed_seconds': run['elapsed_seconds'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
             'required_Fisher_payload_examples': 2048, 'required_Fisher_payload_VJPs': 128,
             'actual_new_Fisher_estimation_examples': 0, 'actual_new_Fisher_estimation_VJPs': 0}
    return run, costs


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_moment_transport_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    previous = json.loads(Path(plan['reference_report']).read_text())
    assert all(previous['checks'].values())
    source = json.loads(Path(plan['source_fit']).read_text())
    assert source['seed'] == plan['seed'] == 0
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    rows = sum(base[n].shape[0] if base[n].ndim else 1 for n in plan['active'])
    assert 2 * len(plan['active']) + 2 * rows == 347666
    assert 3 * sum(base[n].numel() for n in plan['active']) * 8 == 618037248
    for task in plan['tasks']:
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
    orders, sources, versions = {}, {}, set()
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            runs[condition], costs[condition] = audit_run(folder, plan, base, source)
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        for condition, folder in plan['reference_runs'][order].items():
            path = Path(folder) / 'run.json'
            assert sha256(path) == previous['sources'][str(path)]
            sources[str(path)] = sha256(path)
            runs[condition] = json.loads(path.read_text())
            costs[condition] = previous['orders'][order]['audits_and_costs'][condition]
        first, second = [root / f'{c}_{order}' for c in plan['conditions']]
        a = torch.load(first / 'stage_02' / 'merged.ckpt', map_location='cpu', weights_only=True)
        b = torch.load(second / 'stage_02' / 'merged.ckpt', map_location='cpu', weights_only=True)
        assert all(torch.equal(a[n], b[n]) for n in a)
        a = torch.load(first / 'stage_02' / 'coefficients.pt', map_location='cpu', weights_only=True)
        b = torch.load(second / 'stage_02' / 'coefficients.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(a[n], b[n]) for n in a)
        del a, b
        reference = runs['old_rows']
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                    'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference'):
            assert plan[key] == reference_plan[key], key
        for run in runs.values():
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert run['order'] == reference['order'] == (plan['tasks'] if order == 'forward' else list(reversed(plan['tasks'])))
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
            versions.add((run['torch_version'], run['cuda_version']))
        pairs = [('moment_transport', 'old_rows'), ('moment_transport', 'moment_restart'),
                 ('moment_restart', 'old_rows'), ('moment_transport', 'raw_columns')]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc'])
                for s in plan['validation_stages']} for a, b in pairs}, 'audits_and_costs': costs}
        print('checked', order, flush=True)
    assert len(versions) == 1
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'orders': orders,
        'mean_gain_over_orders_pp': {key: float(np.mean([o['contrasts'][key]['gain_pp'] for o in orders.values()])) for key in orders['forward']['contrasts']},
        'checks': {'four_new_complete_runs': True, 'actual_data_schedules_auxiliary_ids_match': True,
                   'strict_settings_match': True, 'first_expert_unchanged': True,
                   'checkpoint_and_coefficient_hashes_pass': True, 'fixed_coefficient_means_exact': True,
                   'coefficient_transfer_and_restart_initialization_exact': True,
                   'stage2_models_and_all_coefficients_equal_between_arms': True,
                   'original_expert_Fisher_moments_replay_exact': True, 'Fisher_payload_provenance_matches': True,
                   'native_exports_pass': True, 'coefficient_and_state_counts_match': True,
                   'storage_links_resolve': True, 'frozen_reference_audits_match': True},
        'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
        'new_Fisher_estimation_examples': 0, 'new_Fisher_estimation_VJPs': 0,
        'reused_Fisher_cache_examples_once': 2048, 'reused_Fisher_cache_VJPs_once': 128,
        'scope': 'One exposed expert bank and two correlated orders. Both moment arms share the same347666 coefficient family and618037248byte additive moments. Transport additionally uses the previous1390664byte coefficient vector. This tests coefficient-state transfer, not additional capacity. Comparison to bilateral also changes expert basis and initialization; Fisher necessity is not isolated. Calibration CE updates/data match, but the moment algorithms additionally require cached2048examples/128VJP Fisher payloads. Those old estimates used a non-strict backend and are fixed inputs; current training is strict. Task-deletion ranges are sensitivity, not confidence intervals. Shared-hardware times are descriptive. No novel superior method is established by engineering.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'raw_columns', 'moment_restart', 'moment_transport')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .10,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('moment_transport_minus_old_rows', 'moment_transport_minus_moment_restart') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs bilateral', 'R vs bilateral', 'F vs restart', 'R vs restart'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Calibration coefficient transport over original-expert Fisher moments — bank0')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
