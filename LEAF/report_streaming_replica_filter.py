"""Audit replica-filtered innovations against same-source soup and shifted controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from check_streaming_replica_filter import independent_incoming
from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_replica_filter import load_replicas
from run_streaming_strict_history import write_json


def audit_export(base, previous, incoming, state, mean, active):
    n = len(active)
    row_count = sum(base[k].shape[0] if base[k].ndim else 1 for k in active)
    offset, errors = 2 * n, []
    for j, name in enumerate(active):
        rows = base[name].shape[0] if base[name].ndim else 1
        shape = (-1,) + (1,) * (base[name].ndim - 1) if base[name].ndim else ()
        old, new = previous[name] - base[name], incoming[name] - base[name]
        # Independently expand the two tensor and two row terms.
        expected = base[name] + mean[j] * old + mean[n + j] * new
        expected = expected + mean[offset:offset + rows].reshape(shape) * new
        expected = expected + mean[offset + row_count:offset + row_count + rows].reshape(shape) * old
        error = float((expected - state[name]).abs().max())
        assert error < 2e-6, (name, error)
        errors.append(error)
        offset += rows
    assert offset + row_count == mean.numel() == 347666
    return max(errors)


def audit_run(folder, plan, base):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    assert run['condition'] in plan['conditions'] and not run['official_test_read']
    assert not run['history_used_by_merger'] and run['persistent_auxiliary_history_bytes'] == 0
    assert run['source_expert_count'] == 48
    assert run['extra_direction_construction_forward_examples'] == run['extra_direction_construction_backward_probes'] == 0
    previous, audits = None, {}
    fixed = set(base) - set(plan['active'])
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        for filename, digest in [('merged.ckpt', stage['checkpoint_sha256']), ('incoming.ckpt', stage['incoming_sha256'])]:
            path = directory / filename
            assert path.is_symlink() and sha256(path) == digest
        assert (directory / 'merged.ckpt').resolve() == Path(stage['checkpoint_storage_path'])
        state = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
        incoming = torch.load(directory / 'incoming.ckpt', map_location='cpu', weights_only=True)
        for model in (state, incoming):
            assert model.keys() == base.keys()
            assert all(torch.equal(model[k], base[k]) for k in fixed)
            assert all(torch.isfinite(model[k]).all() for k in plan['active'])
        replicas = load_replicas(plan, task)
        oracle = independent_incoming(base, previous, replicas, plan['active'], run['condition'], plan['tasks'].index(task))
        error = max(float((incoming[k] - oracle[k]).abs().max()) for k in plan['active'])
        assert error < 2e-7
        audits[str(number)] = {'independent_incoming_max_abs_error': error}
        if number == 1:
            assert stage['updates'] == stage['training_examples'] == 0
            assert all(torch.equal(state[k], incoming[k]) for k in state)
            assert not stage['filter_geometry']
        else:
            assert stage['updates'] == number * 12 and stage['training_examples'] == number * 192
            path = directory / 'coefficients.pt'
            assert path.is_symlink() and sha256(path) == stage['coefficients_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['snapshots'].shape == (6, 347666) and payload['mean'].shape == (347666,)
            assert all(torch.isfinite(v).all() for v in payload.values())
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            assert stage['coefficient_count'] == 347666
            audits[str(number)]['independent_export_max_abs_error'] = audit_export(base, previous, incoming, state, payload['mean'], plan['active'])
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0 for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            if run['condition'] == 'source_soup':
                assert not stage['filter_geometry']
            else:
                assert set(stage['filter_geometry']) == set(plan['active'])
                for row in stage['filter_geometry'].values():
                    assert row['filtered_squared_norm'] <= row['innovation_squared_norm'] * (1 + 1e-12)
                    assert row['filtered_squared_norm'] >= row['innovation_squared_norm'] / 9 * (1 - 1e-12)
            del payload
        previous = state
        del incoming, replicas, oracle
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(v['validation_size'] for v in run['dataset_audits'].values()) == 60413
    return run, {'stage_audits': audits, 'elapsed_seconds': run['elapsed_seconds'],
                 'construction_seconds': sum(s['construction_seconds'] for s in run['stages']),
                 'peak_reserved_bytes': run['peak_reserved_bytes'], 'coefficient_count': 347666,
                 'persistent_auxiliary_history_bytes': 0, 'source_training_cost': plan['source_training_cost']}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_replica_filter_v1')
    assert not (root / 'comparison.json').exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    prior = json.loads(Path(plan['reference_report']).read_text())
    assert all(prior['checks'].values())
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    orders, sources, versions = {}, {}, set()
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            runs[condition], costs[condition] = audit_run(folder, plan, base)
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            print('audited', condition, order, flush=True)
        path = Path(plan['reference_runs'][order]['old_rows']) / 'run.json'
        assert sha256(path) == prior['sources'][str(path)]
        sources[str(path)] = sha256(path)
        reference = json.loads(path.read_text())
        runs['old_rows'] = reference
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                    'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference'):
            assert plan[key] == reference_plan[key], key
        for run in runs.values():
            assert run['order'] == reference['order']
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            versions.add((run['torch_version'], run['cuda_version']))
        # Three new arms share the first source soup; the one-bank reference does not.
        first = torch.load(root / f'source_soup_{order}/stage_01/merged.ckpt', map_location='cpu', weights_only=True)
        for condition in plan['conditions']:
            other = torch.load(root / f'{condition}_{order}/stage_01/merged.ckpt', map_location='cpu', weights_only=True)
            assert all(torch.equal(first[k], other[k]) for k in first)
            assert runs[condition]['stages'][0]['validation'] == runs['source_soup']['stages'][0]['validation']
        del first, other
        pairs = [('innovation_filter', 'source_soup'), ('innovation_filter', 'shifted_filter'),
                 ('shifted_filter', 'source_soup')] + [(c, 'old_rows') for c in plan['conditions']]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        old, new = reference['order'][:12], reference['order'][12:]
        orders[order] = {'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']} for a, b in pairs},
            'last_block_retention': {name: {
                'old12_final_BAcc': float(np.mean([final[name][t]['BAcc'] for t in old])),
                'old12_stage12_to16_change_pp': float(np.mean([100 * (final[name][t]['BAcc'] - run['stages'][11]['validation'][t]['BAcc']) for t in old])),
                'new4_final_BAcc': float(np.mean([final[name][t]['BAcc'] for t in new]))} for name, run in runs.items()},
            'audits_and_costs': costs}
    assert len(versions) == 1
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'orders': orders,
        'mean_gain_over_orders_pp': {k: float(np.mean([o['contrasts'][k]['gain_pp'] for o in orders.values()])) for k in orders['forward']['contrasts']},
        'checks': {k: True for k in ['six_complete_runs', 'frozen_sources_and_inputs_match', 'original_data_labels_weights_schedules_match',
            'incoming_independently_reconstructed_all_stages', 'export_independently_reconstructed_all_stages',
            'fixed_coefficient_means_exact', 'checkpoint_and_payload_hashes_pass', 'first_source_soup_identical_across_new_arms',
            'native_exports_pass', 'strict_settings_match', 'filter_norm_bounds_pass', 'source_budget_accounted', 'storage_links_resolve']},
        'new_scientific_updates': 9720, 'new_training_CE_examples': 155520,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'scope': 'Three same-task replicas arrive together; no future expert payload enters the operator. All new arms use48experts and the same first source soup, versus16experts for the original bilateral reference. This comparison changes source training cost; superiority of the new operator requires comparison to same-source soup and shifted orientation controls. One exposed calibration bank and two correlated orders. Trace normalization discards absolute scatter magnitude; seed variation is not calibrated uncertainty. No new inference state, routing or model architecture. Task-deletion ranges are sensitivity, not confidence intervals.'}
    write_json(root / 'comparison.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'source_soup', 'shifted_filter', 'innovation_filter')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .1,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('innovation_filter_minus_source_soup', 'innovation_filter_minus_shifted_filter') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs soup', 'R vs soup', 'F vs shifted', 'R vs shifted'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Replica-filtered sequential merging — three source banks')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'comparison.{ext}', dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
