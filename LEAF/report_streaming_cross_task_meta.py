"""Audit static cross-task meta merges against exposure-matched controls."""
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
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import write_json


def expected_schedules(plan, tasks, stage):
    permutations = {t: np.random.default_rng([plan['seed'], stage,
        plan['tasks'].index(t), 217]).permutation(512) for t in tasks}
    rng = np.random.default_rng([plan['seed'], stage, 931])
    primary, paired = [], []
    bs = plan['batch_size']
    for rnd in range(plan['rounds']):
        order = rng.permutation(len(tasks)).tolist()
        for position, k in enumerate(order):
            q, s = tasks[k], tasks[order[(position + 1) % len(tasks)]]
            qids, sids = [permutations[t][rnd * bs:(rnd + 1) * bs].tolist() for t in (q, s)]
            primary.append([plan['tasks'].index(q), *qids])
            paired.append([plan['tasks'].index(s), *sids, plan['tasks'].index(q), *qids])
    return [array_hash(np.asarray(x, dtype=np.int64)) for x in (primary, paired)]


def audit_run(folder, plan, source, base):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 51840
    assert run['condition'] in plan['conditions'] and not run['official_test_read']
    assert not run['history_used_by_merger'] and run['persistent_auxiliary_history_bytes'] == 0
    assert run['source_expert_count'] == 16
    assert run['attention_backend'] == 'math SDP; MHA fastpath disabled; all three arms'
    previous, audits = None, {}
    fixed = set(base) - set(plan['active'])
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        path = directory / 'merged.ckpt'
        assert path.is_symlink() and sha256(path) == stage['checkpoint_sha256']
        assert path.resolve() == Path(stage['checkpoint_storage_path'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        source_path = Path(source['experts'][task])
        assert sha256(source_path) == stage['incoming_sha256'] == plan['expert_sha256'][task]
        assert stage['incoming_path'] == str(source_path)
        incoming = torch.load(source_path, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[k], base[k]) for k in fixed)
        assert all(torch.isfinite(state[k]).all() for k in plan['active'])
        if number == 1:
            assert stage['updates'] == stage['training_examples'] == 0
            assert all(torch.equal(state[k], incoming[k]) for k in state)
        else:
            updates = number * 12
            assert stage['updates'] == updates and stage['training_examples'] == updates * 32
            assert stage['training_forward_batches'] == 2 * updates
            assert stage['support_gradient_calls'] == stage['outer_gradient_calls'] == updates
            assert stage['outer_calls_with_second_order'] == (updates if run['condition'] == 'cross_task_meta' else 0)
            primary, paired = expected_schedules(plan, run['order'][:number], number)
            assert stage['schedule_hash'] == primary and stage['pair_schedule_hash'] == paired
            path = directory / 'coefficients.pt'
            assert path.is_symlink() and sha256(path) == stage['coefficients_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['snapshots'].shape == (6, 347666) and payload['mean'].shape == (347666,)
            assert all(torch.isfinite(v).all() for v in payload.values())
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            assert stage['coefficient_count'] == 347666
            audits[str(number)] = {'independent_export_max_abs_error': audit_export(
                base, previous, incoming, state, payload['mean'], plan['active'])}
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                       for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            assert len(stage['round_diagnostics']) == 12
            for diagnostic in stage['round_diagnostics']:
                assert all(np.isfinite(np.asarray(v)).all() for v in diagnostic.values())
                assert all(0 <= x <= plan['virtual_step_rate'] + 1e-7
                           for x in diagnostic['virtual_step_block_rms'])
            del payload
        previous = state
        del incoming
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(v['validation_size'] for v in run['dataset_audits'].values()) == 60413
    return run, {'stage_audits': audits, 'elapsed_seconds': run['elapsed_seconds'],
        'training_seconds': sum(s.get('training_seconds', 0) for s in run['stages']),
        'peak_reserved_bytes': run['peak_reserved_bytes'], 'training_CE_examples': run['training_examples'],
        'training_forward_batches': sum(s.get('training_forward_batches', 0) for s in run['stages']),
        'support_gradient_calls': sum(s.get('support_gradient_calls', 0) for s in run['stages']),
        'outer_gradient_calls': sum(s.get('outer_gradient_calls', 0) for s in run['stages']),
        'outer_calls_with_second_order': sum(s.get('outer_calls_with_second_order', 0) for s in run['stages']),
        'coefficient_count': 347666, 'persistent_auxiliary_history_bytes': 0}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_cross_task_meta_v1')
    assert not (root / 'comparison.json').exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    prior = json.loads(Path(plan['reference_report']).read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    orders, sources, versions = {}, {}, set()
    pairs = [('cross_task_meta', 'paired_ce'), ('cross_task_meta', 'detached_lookahead'),
             ('detached_lookahead', 'paired_ce')] + [(c, 'old_rows') for c in plan['conditions']]
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            runs[condition], costs[condition] = audit_run(folder, plan, source, base)
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
        for name, run in runs.items():
            assert run['order'] == reference['order']
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            versions.add((run['torch_version'], run['cuda_version']))
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        old, new = reference['order'][:12], reference['order'][12:]
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s-1]['macro_BAcc'] - runs[b]['stages'][s-1]['macro_BAcc'])
                                   for s in plan['validation_stages']} for a, b in pairs},
            'last_block_retention': {name: {
                'old12_final_BAcc': float(np.mean([final[name][t]['BAcc'] for t in old])),
                'old12_stage12_to16_change_pp': float(np.mean([100 * (final[name][t]['BAcc'] - run['stages'][11]['validation'][t]['BAcc']) for t in old])),
                'new4_final_BAcc': float(np.mean([final[name][t]['BAcc'] for t in new]))} for name, run in runs.items()},
            'audits_and_costs': costs}
    assert len(versions) == 1
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'orders': orders,
        'mean_gain_over_orders_pp': {k: float(np.mean([o['contrasts'][k]['gain_pp'] for o in orders.values()]))
                                   for k in orders['forward']['contrasts']},
        'checks': {k: True for k in ['six_complete_runs', 'frozen_sources_and_inputs_match',
            'original_data_labels_weights_primary_schedules_match', 'pair_schedules_independently_reconstructed',
            'export_independently_reconstructed_all_stages', 'fixed_coefficient_means_exact',
            'checkpoint_and_payload_hashes_pass', 'first_checkpoint_equals_original_expert',
            'native_exports_pass', 'strict_settings_match', 'same_attention_backend_all_new_arms',
            'step_rms_bounds_pass', 'source_and_training_cost_accounted', 'storage_links_resolve']},
        'new_scientific_updates': 9720, 'new_training_CE_examples': 311040,
        'outer_gradient_calls_with_second_order': 3240,
        'scope': 'Exploratory adaptation of established MLDG-style meta-learning to sequential bilateral merging. No new-method novelty claim. Primary controls match source experts, calibration, examples, outer updates and attention backend. Full meta has extra second-order derivative cost; equal FLOPs are not claimed. Historical single-batch bilateral has half the training exposures and its original attention backend. One exposed calibration bank and two correlated orders; no official test read. Deletion ranges are sensitivity, not confidence intervals.'}
    write_json(root / 'comparison.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'paired_ce', 'detached_lookahead', 'cross_task_meta')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .1, [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']],
                        label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('cross_task_meta_minus_paired_ce', 'cross_task_meta_minus_detached_lookahead') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs paired CE', 'R vs paired CE', 'F vs detached', 'R vs detached'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Cross-task update-aware coefficient learning — one source bank')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'comparison.{ext}', dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
        'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
