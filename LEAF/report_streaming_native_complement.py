"""Compare simultaneous coefficient/native optimization under matched replay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_strict_history import write_json
from streaming_projected_sketch_v2 import orthonormal_rows


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(Path(run['plan'])) == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    condition = run['condition']
    assert condition in plan['conditions']
    assert not run['history_used_by_merger'] and run['memory_policy'] == 'no_auxiliary_history_between_stages'
    assert run['extra_direction_construction_forward_examples'] == run['extra_direction_construction_backward_probes'] == 0
    fixed = set(base) - set(plan['active'])
    previous = None
    independent_maximum_projection = 0.
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
        original = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        directory = folder / f'stage_{number:02d}'
        checkpoint_path = directory / 'merged.ckpt'
        assert checkpoint_path.is_symlink() and checkpoint_path.resolve() == Path(stage['checkpoint_storage_path'])
        assert sha256(checkpoint_path) == stage['checkpoint_sha256']
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[name], base[name]) for name in fixed)
        assert all(torch.isfinite(state[name]).all() for name in plan['active'])
        assert stage['persistent_auxiliary_history_bytes'] == 0
        if number == 1:
            assert all(torch.equal(state[name], original[name]) for name in state)
            assert stage['updates'] == stage['training_examples'] == 0
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            coefficient_path = directory / 'coefficients.pt'
            assert coefficient_path.is_symlink()
            assert sha256(coefficient_path) == stage['coefficients_sha256']
            parameters = torch.load(coefficient_path, map_location='cpu', weights_only=True)
            assert parameters['snapshots'].shape == (6, 347666)
            assert parameters['initial'].shape == parameters['mean'].shape == (347666,)
            assert stage['coefficient_count'] == 347666
            assert all(torch.isfinite(parameters[k]).all() for k in ('initial', 'snapshots', 'mean'))
            assert torch.equal(parameters['snapshots'].double().mean(0).float(), parameters['mean'])
            default = torch.zeros_like(parameters['initial'])
            n = len(plan['active'])
            default[:n] = (number - 1) / number
            default[n:2 * n] = 1 / number
            assert torch.equal(parameters['initial'], default)
            assert array_hash(parameters['initial'].numpy()) == stage['initial_coefficient_hash']
            assert len(parameters['native_snapshots']) == 6
            assert set(parameters['native_mean']) == set(plan['active'])
            assert all(set(s) == set(plan['active']) for s in parameters['native_snapshots'])
            assert stage['native_residual_count'] == 25751552 and stage['native_residual_bytes'] == 103006208
            assert stage['native_residual_audit']['squared_norm'] > 0
            represented = 0
            for name in plan['active']:
                snapshots = torch.stack([s[name] for s in parameters['native_snapshots']])
                assert snapshots.shape == (6, *base[name].shape) and snapshots.dtype == torch.float32
                assert torch.isfinite(snapshots).all()
                mean = snapshots.double().mean(0).float()
                assert torch.equal(mean, parameters['native_mean'][name])
                if condition == 'complement_hybrid':
                    old, new = previous[name] - base[name], original[name] - base[name]
                    rows = old.shape[0] if old.ndim else 1
                    q = orthonormal_rows(torch.stack([old.reshape(rows, -1), new.reshape(rows, -1)], dim=1)).double()
                    residual = mean.double().reshape(rows, -1)
                    coordinates = (q * residual[:, None, :]).sum(-1)
                    error = float((coordinates.abs() / residual.norm(dim=1, keepdim=True).clamp_min(1e-30)).max())
                    independent_maximum_projection = max(independent_maximum_projection, error)
                    assert error < 2e-6
                    represented += int((q.square().sum(-1) > .5).sum())
                del snapshots, mean
            if condition == 'native_only':
                assert all(torch.equal(s, default) for s in parameters['snapshots'])
                assert stage['trainable_parameters'] == 25751552
                assert stage['optimizer_state_bytes'] == 206013948
                assert stage['update_kind'] == 'native_only'
            else:
                assert stage['trainable_parameters'] == 26099218
                assert stage['optimizer_state_bytes'] == 208795280
                assert stage['update_kind'] == 'joint_coefficient_and_native'
            if condition == 'complement_hybrid':
                assert stage['basis_audit']['padded_basis_bytes'] == 206012416
                assert stage['basis_audit']['represented_span_dimensions'] == represented
                assert stage['basis_audit']['maximum_orthogonality_error'] < 2e-5
                assert stage['native_residual_audit']['max_relative_row_projection'] < 2e-6
                counts = stage['basis_audit']['per_row_effective_rank_counts']
                assert sum(counts.values()) == 173450
                assert sum(int(k) * v for k, v in counts.items()) == represented
            else:
                assert stage['basis_audit']['padded_basis_bytes'] == 0
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0 for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            del parameters
        previous = state
        del original
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(x['validation_size'] for x in run['dataset_audits'].values()) == 60413
    final = run['stages'][-1]
    costs = {'trainable_parameters': final['trainable_parameters'],
             'coefficient_parameters_learned': 0 if condition == 'native_only' else 347666,
             'native_residual_parameters': 25751552, 'native_residual_bytes': 103006208,
             'optimizer_state_bytes': final['optimizer_state_bytes'],
             'padded_basis_bytes': final['basis_audit']['padded_basis_bytes'],
             'persistent_auxiliary_history_bytes': 0,
             'independently_recomputed_max_relative_row_projection': independent_maximum_projection,
             'basis_and_residual_by_stage': {str(s['stage']): {'basis': s['basis_audit'], 'residual': s['native_residual_audit']} for s in run['stages'][1:]},
             'training_and_export_seconds': sum(s['training_seconds'] for s in run['stages'][1:]),
             'calibration_including_setup_seconds': sum(s['calibration_call_seconds'] for s in run['stages'][1:]),
             'elapsed_seconds': run['elapsed_seconds'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
             'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0}
    return run, costs


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_native_complement_v1')
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
        reference = runs['old_rows']
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                    'validation_stages', 'calibration', 'calibration_reference'):
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
        pairs = [('complement_hybrid', 'old_rows'), ('complement_hybrid', 'unprojected_hybrid'),
                 ('complement_hybrid', 'native_only'), ('unprojected_hybrid', 'old_rows'),
                 ('native_only', 'old_rows'), ('unprojected_hybrid', 'native_only'),
                 ('complement_hybrid', 'raw_columns')]
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
        'checks': {'six_new_complete_runs': True, 'actual_data_schedules_auxiliary_ids_match': True,
                   'strict_settings_match': True, 'first_expert_unchanged': True,
                   'checkpoint_and_coefficient_hashes_pass': True, 'fixed_coefficient_and_native_means_exact': True,
                   'bilateral_default_initial_coefficients_exact': True,
                   'native_only_coefficients_fixed': True, 'native_complement_recomputed_from_saved_states': True,
                   'native_exports_pass': True, 'coefficient_and_state_counts_match': True,
                   'storage_links_resolve': True, 'frozen_reference_audits_match': True},
        'new_scientific_updates': 9720, 'new_scientific_CE_example_passes': 155520,
        'joint_coefficient_and_native_update_calls': 6480, 'native_only_update_calls': 3240,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'scope': 'One exposed bank, two correlated orders, three new conditions. Hybrids jointly optimize347666bilateral coefficients atAdam0.01 and25751552shared native residual parameters atAdam1e-5. Native-only freezes coefficients at the same initial values. Constrained hybrid projects native gradients and residuals off a fixed numerical old/new row basis; unprojected hybrid has no projection. All begin at the same bilateral initialization and use the same original512CE schedule. Both hybrid families can span full native parameter space in exact arithmetic, so this tests coordinates and update rules, not increased capacity from orthogonality. Native residual/moments and projection bases are additional training costs. No inference routing or growing inference bank. Means use fixed rounds7..12 separately for coefficients and native residuals inFP64. Extra EEG direction-construction passes0. Task-deletion ranges are sensitivity, not confidence intervals. No general preservation guarantee or established novelty.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'native_only', 'unprojected_hybrid', 'complement_hybrid')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .10,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('complement_hybrid_minus_old_rows', 'complement_hybrid_minus_unprojected_hybrid') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs bilateral', 'R vs bilateral', 'F vs raw hybrid', 'R vs raw hybrid'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Simultaneous bilateral and native calibration — bank0')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
