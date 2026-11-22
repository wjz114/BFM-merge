"""Audit unchanged replica filtering with a second calibration bank."""
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


from report_streaming_replica_filter import audit_run


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_replica_filter_replication_v1/seed2')
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
    selection_path = Path(plan['selection_report'])
    selection = json.loads(selection_path.read_text())
    assert all(selection['checks'].values())
    selected_plan = json.loads(Path(plan['selection_plan']).read_text())
    for key in ('tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                'optimizer', 'precision', 'replica_seeds', 'replica_experts', 'replica_expert_sha256',
                'conditions', 'source_training_cost'):
        assert plan[key] == selected_plan[key], key
    for name in ('streaming_replica_filter.py', 'run_streaming_replica_filter.py',
                 'run_streaming_strict_history.py', 'streaming_bilateral_merge.py',
                 'streaming_residual_merge.py', 'streaming_residual_merge_v2.py'):
        assert plan['source_sha256'][name] == selected_plan['source_sha256'][name], name
    result['selection_report'] = str(selection_path)
    result['selection_report_sha256'] = sha256(selection_path)
    result['combined_mean_gain_over_banks_and_orders_pp'] = {
        k: (v + selection['mean_gain_over_orders_pp'][k]) / 2
        for k, v in result['mean_gain_over_orders_pp'].items()}
    result['checks']['same_operator_sources_and_all_three_source_banks'] = True
    result['scope'] += ' This replication changes calibration samples and training RNG to seed2; the three source expert banks stay fixed. It is not an independent replication of source training or evaluation subjects. Both development banks are exposed.'
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
    fig.suptitle('Replica filtering replication — calibration seed2, same three source banks')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'comparison.{ext}', dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
