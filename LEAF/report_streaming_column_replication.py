"""Audit unchanged column operators on bank2 and separate their bank0 selection results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from report_streaming_column_residual import audit_run
from run_streaming_strict_history import write_json
from streaming_column_residual import eligible


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_column_replication_v1/seed2')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    previous = json.loads(Path(plan['reference_report']).read_text())
    selected = json.loads(Path(plan['selection_report']).read_text())
    assert all(previous['checks'].values()) and all(selected['checks'].values())
    for path, digest in selected['sources'].items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    assert source['seed'] == plan['seed'] == 2
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    assert [name for name in plan['active'] if eligible(name, base[name])] == plan['column_layers']
    assert len(plan['column_layers']) == 124
    assert 2 * sum(base[name].shape[1] for name in plan['column_layers']) == 112640
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
            costs[condition] = previous['banks']['2']['orders'][order]['audits_and_costs'][condition]
        reference = runs['old_rows']
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size',
                    'learning_rate', 'validation_stages', 'optimizer', 'precision',
                    'calibration', 'calibration_reference'):
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
        pairs = [('projected_columns', 'old_rows'), ('projected_columns', 'raw_columns'),
                 ('raw_columns', 'old_rows'), ('projected_columns', 'full_history')]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc'])
                for s in plan['validation_stages']} for a, b in pairs},
            'audits_and_costs': costs}
        print('checked bank2', order, flush=True)
    assert len(versions) == 1
    bank2_mean = {key: float(np.mean([o['contrasts'][key]['gain_pp'] for o in orders.values()]))
                  for key in orders['forward']['contrasts']}
    banks = {'0': {'role': 'initial exploration used to select follow-up', 'orders': selected['orders'],
                   'mean_gain_over_orders_pp': selected['mean_gain_over_orders_pp']},
             '2': {'role': 'unchanged follow-up on an already exposed development bank', 'orders': orders,
                   'mean_gain_over_orders_pp': bank2_mean}}
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'banks': banks,
        'selection_report': plan['selection_report'], 'selection_report_sha256': sha256(Path(plan['selection_report'])),
        'followup_bank_mean_gain_pp': bank2_mean,
        'all_four_case_mean_gain_pp': {k: float(np.mean([banks[b]['mean_gain_over_orders_pp'][k] for b in banks])) for k in bank2_mean},
        'checks': {'four_new_complete_runs': True, 'unchanged_operator_source_hashes': True,
                   'actual_data_schedules_auxiliary_ids_match': True, 'strict_settings_match': True,
                   'first_expert_unchanged': True, 'checkpoint_and_coefficient_hashes_pass': True,
                   'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
                   'matrix_and_coefficient_counts_match': True, 'no_persistent_auxiliary_expert_history': True,
                   'storage_links_resolve': True, 'frozen_reference_and_selection_reports_match': True},
        'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
        'extra_score_VJPs': 0,
        'scope': 'Bank2 is deliberately selected for earlier candidate failures and already development-exposed. Bank0 selected this follow-up. Orders and banks share task datasets, so neither four independent replications nor held-out confirmation is claimed. Raw and projected have identical attainable model families at fixed stage in exact arithmetic; their contrast tests parameterization and optimization. Task-deletion ranges are sensitivity, not confidence intervals. No novelty or superiority is assumed. Shared-hardware timings are descriptive.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    cases = [(b, o) for b in banks for o in plan['orders']]
    for i, name in enumerate(('raw_columns', 'projected_columns')):
        key = f'{name}_minus_old_rows'
        axes[0].scatter(np.arange(4) + (i - .5) * .12,
            [banks[b]['orders'][o]['contrasts'][key]['gain_pp'] for b, o in cases],
            label=name.replace('_', ' '), s=75)
    axes[0].axhline(0, color='grey', linewidth=1)
    axes[0].set_xticks(range(4), [f'Bank{b} {o[0].upper()}' for b, o in cases])
    axes[0].set_ylabel('Gain versus bilateral rows (percentage points)')
    axes[0].legend()
    heat = np.asarray([[banks[b]['orders'][o]['contrasts']['projected_columns_minus_raw_columns']['per_task_pp'][t]
                        for b, o in cases] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), [f'Bank{b} {o[0].upper()}' for b, o in cases])
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    axes[1].set_title('Projected minus raw columns')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Shared input-column corrections: bank0 exploration and bank2 follow-up')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'followup_bank_mean_gain_pp': bank2_mean,
                     'all_four_case_mean_gain_pp': result['all_four_case_mean_gain_pp']}, indent=2))


if __name__ == '__main__':
    main()
