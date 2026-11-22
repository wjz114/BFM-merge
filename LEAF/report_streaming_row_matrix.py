"""Audit scaled row-plane calibration, covariance and state transport."""
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


def audit_payload(payload, record, base, previous, incoming, previous_payload, active, condition, accumulated_steps):
    rows = sum(base[n].shape[0] if base[n].ndim else 1 for n in active)
    assert rows == 173450
    assert payload['mean'].shape == (rows, 2)
    assert payload['snapshots'].shape[1:] == (rows, 2)
    assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
    assert torch.isfinite(payload['mean']).all() and torch.isfinite(payload['snapshots']).all()
    assert record['initial_native_weights_exactly_match_bilateral']
    offset, ranks, projection_errors, scale_parts = 0, {}, [], []
    mean_errors, covariance_errors = [], []
    for name in active:
        old, new = previous[name] - base[name], incoming[name] - base[name]
        n = old.shape[0] if old.ndim else 1
        sl = slice(offset, offset + n)
        offset += n
        q = payload['basis'][name].double()
        assert q.shape == (n, 2, old.numel() // n) and torch.isfinite(q).all()
        directions = torch.stack([old.reshape(n, -1), new.reshape(n, -1)], dim=1).double()
        scale = directions.square().sum(2).mean(1).sqrt().float()
        scale_parts.append(scale)
        support = q.square().sum(2) > .5
        assert (q @ q.transpose(1, 2) - torch.diag_embed(support.double())).abs().max() < 2e-5
        unique, counts = torch.unique(support.sum(1), return_counts=True)
        for rank, total in zip(unique.tolist(), counts.tolist()):
            ranks[str(rank)] = ranks.get(str(rank), 0) + total
        reconstruction = (directions @ q.transpose(1, 2)) @ q
        norm = directions.norm(dim=2)
        relative = (reconstruction - directions).norm(dim=2) / norm.clamp_min(1e-30)
        projection_errors.append(float(relative.max()))
        assert relative.max() <= 2e-6
        if condition == 'matrix_transport' and previous_payload is not None:
            oldq = previous_payload['basis'][name].double()
            oldscale = previous_payload['scale'][sl].double()
            ratio = torch.where(oldscale > 0, scale.double() / oldscale.clamp_min(1e-300), 0.)
            a = torch.einsum('rid,rjd->rij', q, oldq) * ratio[:, None, None]
            expected_m = torch.einsum('rij,rj->ri', a, previous_payload['optimizer_final']['m'][sl])
            expected_v = torch.einsum('rki,rij,rlj->rkl', a, previous_payload['optimizer_final']['v'][sl], a)
            actual_m = payload['optimizer_initial']['m'][sl]
            actual_v = payload['optimizer_initial']['v'][sl]
            assert torch.allclose(actual_m, expected_m, atol=1e-12, rtol=1e-10)
            assert torch.allclose(actual_v, expected_v, atol=1e-14, rtol=1e-10)
            mean_errors.append(float((actual_m - expected_m).abs().max()))
            covariance_errors.append(float((actual_v - expected_v).abs().max()))
    assert torch.equal(torch.cat(scale_parts), payload['scale'])
    assert ranks == record['geometry']['per_row_rank_counts']
    assert record['geometry']['basis_bytes'] == 206012416
    assert record['geometry']['scale_bytes'] == 693800
    assert record['optimizer_moment_bytes'] == 8325600
    assert record['coefficient_count'] == 346900
    for state in (payload['optimizer_initial'], payload['optimizer_final']):
        assert state['m'].shape == (rows, 2) and state['v'].shape == (rows, 2, 2)
        assert state['m'].dtype == state['v'].dtype == torch.float64
        assert torch.isfinite(state['m']).all() and torch.isfinite(state['v']).all()
        v = state['v']
        assert (v - v.transpose(1, 2)).abs().max() < 1e-12
        a, b, c = v[:, 0, 0], v[:, 0, 1], v[:, 1, 1]
        smallest = (a + c - torch.sqrt((a - c).square() + 4 * b.square())) / 2
        assert (smallest >= -1e-12 * (a.abs() + c.abs()) - 1e-30).all()
    initial, final = payload['optimizer_initial'], payload['optimizer_final']
    expected_steps = accumulated_steps if condition == 'matrix_transport' else 0
    assert initial['step'] == record['optimizer_initial_step'] == expected_steps
    assert final['step'] == record['optimizer_final_step'] == expected_steps + record['updates']
    if condition != 'matrix_transport' or previous_payload is None:
        assert torch.count_nonzero(initial['m']) == torch.count_nonzero(initial['v']) == 0
        assert record['transport'] is None
    if condition == 'diagonal_reset':
        assert torch.count_nonzero(final['v'][:, 0, 1]) == torch.count_nonzero(final['v'][:, 1, 0]) == 0
    expected_history = 215031824 if condition == 'matrix_transport' else 0
    assert record['persistent_auxiliary_history_bytes'] == expected_history
    return {'basis_direction_max_relative_reconstruction_error': max(projection_errors),
            'independent_transport_mean_max_error': max(mean_errors, default=0.),
            'independent_transport_covariance_max_error': max(covariance_errors, default=0.)}


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    assert run['condition'] in plan['conditions'] and not run['official_test_read']
    assert run['history_used_by_merger'] == (run['condition'] == 'matrix_transport')
    assert run['extra_direction_construction_forward_examples'] == run['extra_direction_construction_backward_probes'] == 0
    fixed = set(base) - set(plan['active'])
    previous, previous_payload, accumulated_steps, audits = None, None, 0, {}
    for number, (task, record) in enumerate(zip(run['order'], run['stages']), 1):
        assert record['stage'] == number and record['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        checkpoint = directory / 'merged.ckpt'
        assert checkpoint.is_symlink() and checkpoint.resolve() == Path(record['checkpoint_storage_path'])
        assert sha256(checkpoint) == record['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        incoming = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[n], base[n]) for n in fixed)
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        if number == 1:
            assert all(torch.equal(state[n], incoming[n]) for n in state)
            assert record['updates'] == record['training_examples'] == record['persistent_auxiliary_history_bytes'] == 0
        else:
            assert record['updates'] == number * 12 and record['training_examples'] == number * 192
            path = directory / 'parameters.pt'
            assert path.is_symlink() and sha256(path) == record['parameters_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['snapshots'].shape == (6, 173450, 2)
            audits[str(number)] = audit_payload(payload, record, base, previous, incoming,
                previous_payload, plan['active'], run['condition'], accumulated_steps)
            assert set(record['export_checks']) == set(run['order'][:number])
            assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0 for v in record['export_checks'].values())
            assert record['input_split']['old_tasks'] == run['order'][:number - 1]
            previous_payload = payload if run['condition'] == 'matrix_transport' else None
            accumulated_steps += record['updates']
            del payload
        previous = state
        del incoming
    assert accumulated_steps == 1620
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(v['validation_size'] for v in run['dataset_audits'].values()) == 60413
    return run, {'stage_geometry_and_transport': audits, 'coefficient_count': 346900,
                 'elapsed_seconds': run['elapsed_seconds'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
                 'setup_seconds': sum(s['setup_seconds'] for s in run['stages'][1:]),
                 'optimizer_moment_bytes': 8325600,
                 'persistent_auxiliary_history_bytes': run['stages'][-1]['persistent_auxiliary_history_bytes']}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_row_matrix_v1')
    assert not (root / 'comparison.json').exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    prior = json.loads(Path(plan['reference_report']).read_text())
    assert all(prior['checks'].values())
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    for task in plan['tasks']:
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
    orders, sources, versions = {}, {}, set()
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            runs[condition], costs[condition] = audit_run(folder, plan, base, source)
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        path = Path(plan['reference_runs'][order]['old_rows']) / 'run.json'
        assert sha256(path) == prior['sources'][str(path)]
        sources[str(path)] = sha256(path)
        reference = json.loads(path.read_text())
        runs['old_rows'] = reference
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                    'validation_stages', 'precision', 'calibration', 'calibration_reference'):
            assert plan[key] == reference_plan[key], key
        for run in runs.values():
            assert run['order'] == reference['order']
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
            versions.add((run['torch_version'], run['cuda_version']))
        # Transport first becomes active at stage3. Stage2 must match reset exactly.
        a = torch.load(root / f'matrix_reset_{order}/stage_02/merged.ckpt', weights_only=True)
        b = torch.load(root / f'matrix_transport_{order}/stage_02/merged.ckpt', weights_only=True)
        assert all(torch.equal(a[n], b[n]) for n in a)
        del a, b
        pairs = [('matrix_transport', 'matrix_reset'), ('matrix_reset', 'diagonal_reset'),
                 ('matrix_transport', 'old_rows'), ('matrix_reset', 'old_rows'), ('diagonal_reset', 'old_rows')]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']} for a, b in pairs},
            'audits_and_costs': costs}
        print('checked', order, flush=True)
    assert len(versions) == 1
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'orders': orders,
        'mean_gain_over_orders_pp': {k: float(np.mean([o['contrasts'][k]['gain_pp'] for o in orders.values()])) for k in orders['forward']['contrasts']},
        'checks': {k: True for k in ['six_complete_runs', 'frozen_sources_and_inputs_match', 'original_data_labels_weights_schedules_match',
            'row_basis_reconstruction_and_scale_pass', 'optimizer_moment_shapes_precision_and_PSD_pass',
            'optimizer_reset_and_cumulative_counters_pass', 'transport_recomputed_from_saved_previous_states',
            'fixed_coefficient_means_exact', 'checkpoint_and_payload_hashes_pass', 'first_expert_unchanged',
            'stage2_matrix_reset_and_transport_exact', 'native_exports_pass', 'strict_settings_match', 'state_memory_counts_match', 'storage_links_resolve']},
        'new_scientific_updates': 9720, 'new_training_CE_examples': 155520,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'scope': 'One exposed development bank and two correlated orders. All new arms share scaled orthonormal old/new row coordinates and FP64 optimizer moments; diagonal_reset is an Adam-type control, not bitwise native FP32 Adam. Full covariance changes preconditioning; transport additionally carries represented moment statistics and the update counter across arrivals. It loses unrepresented components and transports final-step moments despite export using the fixed coefficient mean. Same exact-arithmetic bilateral row family apart from rank-floor removal and FP32 differences; redundant tensor-wide gates are removed. No full-matrix optimizer novelty or convergence guarantee claimed. Task-deletion ranges are sensitivity, not confidence intervals.'}
    write_json(root / 'comparison.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'diagonal_reset', 'matrix_reset', 'matrix_transport')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .1,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('matrix_transport_minus_old_rows', 'matrix_transport_minus_matrix_reset') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs bilateral', 'R vs bilateral', 'F vs reset', 'R vs reset'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Matrix adaptive merging with transported row-plane history — bank0')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'comparison.{ext}', dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
