"""Audit unchanged Fisher residual operators on bank2, separately from bank0 selection."""
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


from report_streaming_fisher_residual import audit_run


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_fisher_residual_replication_v1/seed2')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    previous = json.loads(Path(plan['reference_report']).read_text())
    assert all(previous['checks'].values())
    selected = json.loads(Path(plan['selection_report']).read_text())
    assert all(selected['checks'].values())
    for path, digest in selected['sources'].items():
        assert sha256(Path(path)) == digest, path
    selected_plan = json.loads(Path(plan['selection_plan']).read_text())
    selected_root = Path(plan['selection_report']).parent
    for name in ('streaming_fisher_residual.py', 'streaming_fisher_residual_training.py', 'run_streaming_fisher_residual.py', 'fisher_moment_merge.py'):
        assert sha256(Path(name)) == plan['source_sha256'][name] == selected_plan['source_sha256'][name], name
    selected_orders = {}
    source = json.loads(Path(plan['source_fit']).read_text())
    assert source['seed'] == plan['seed'] == 2
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
            costs[condition] = previous['banks']['2']['orders'][order]['audits_and_costs'][condition]
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
        pairs = [('fisher_residual', 'old_rows'), ('fisher_residual', 'shifted_fisher_residual'),
                 ('shifted_fisher_residual', 'old_rows'), ('fisher_residual', 'raw_columns'),
                 ('shifted_fisher_residual', 'fisher_residual'), ('shifted_fisher_residual', 'raw_columns')]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc'])
                for s in plan['validation_stages']} for a, b in pairs}, 'audits_and_costs': costs}
        selected_runs = {name: json.loads((selected_root / f'{name}_{order}' / 'run.json').read_text())
                         for name in plan['conditions']}
        selected_runs.update({name: json.loads((Path(folder) / 'run.json').read_text())
                              for name, folder in selected_plan['reference_runs'][order].items()})
        old_final = {name: run['stages'][-1]['validation'] for name, run in selected_runs.items()}
        selected_orders[order] = {
            'macro_BAcc': selected['orders'][order]['macro_BAcc'],
            'contrasts': {f'{a}_minus_{b}': contrast(old_final[a], old_final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (selected_runs[a]['stages'][s - 1]['macro_BAcc'] - selected_runs[b]['stages'][s - 1]['macro_BAcc'])
                for s in plan['validation_stages']} for a, b in pairs},
            'audits_and_costs': selected['orders'][order]['audits_and_costs']}
        for key, value in selected['orders'][order]['contrasts'].items():
            assert selected_orders[order]['contrasts'][key] == value
        print('checked bank2', order, flush=True)
    assert len(versions) == 1
    means = lambda orders: {key: float(np.mean([o['contrasts'][key]['gain_pp'] for o in orders.values()]))
                             for key in orders['forward']['contrasts']}
    banks = {'0': {'role': 'exploration used for post-hoc selection of the shifted control',
                   'orders': selected_orders, 'mean_gain_over_orders_pp': means(selected_orders)},
             '2': {'role': 'unchanged follow-up on an already development-exposed expert bank',
                   'orders': orders, 'mean_gain_over_orders_pp': means(orders)}}
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'banks': banks,
        'selection_report': plan['selection_report'], 'selection_report_sha256': sha256(Path(plan['selection_report'])),
        'followup_bank_mean_gain_pp': means(orders),
        'all_four_case_mean_gain_pp': {k: float(np.mean([banks[b]['mean_gain_over_orders_pp'][k] for b in banks])) for k in means(orders)},
        'checks': {'four_new_complete_runs': True, 'actual_data_schedules_auxiliary_ids_match': True,
                   'strict_settings_match': True, 'first_expert_unchanged': True,
                   'checkpoint_and_coefficient_hashes_pass': True, 'fixed_coefficient_means_exact': True,
                   'bilateral_default_and_zero_residual_initial_coefficients_exact': True,
                   'residual_projection_and_scale_records_pass': True,
                   'original_expert_Fisher_moments_replay_exact': True, 'Fisher_payload_provenance_matches': True,
                   'native_exports_pass': True, 'coefficient_and_state_counts_match': True,
                   'storage_links_resolve': True, 'frozen_reference_and_selection_audits_match': True,
                   'unchanged_operator_and_training_source_hashes': True},
        'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
        'new_Fisher_estimation_examples_once_for_bank2': 2048,
        'new_Fisher_estimation_VJPs_once_for_bank2': 128,
        'scope': 'Shifted assignment was a control selected post hoc from bank0. Bank2 is deliberately selected because prior small improvements failed there, and is already development-exposed. Both operators, recipes and old data schedules are unchanged. New bank2 Fisher inputs use the same estimator with the bank2 seed and training calibration. Estimation is configure-only, not strict; merging is strict. No task-independent inference bank. Coefficient/state/Fisher costs exceed bilateral. Orders and banks share task datasets; four cases are not independent replications or held-out confirmation. Task-deletion ranges are sensitivity, not confidence intervals. No novelty or superiority is presumed.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    cases = [(b, o) for b in banks for o in plan['orders']]
    for i, name in enumerate(('old_rows', 'raw_columns', 'fisher_residual', 'shifted_fisher_residual')):
        axes[0].scatter(np.arange(4) + (i - 1.5) * .10,
            [100 * banks[b]['orders'][o]['macro_BAcc'][name] for b, o in cases],
            label=name.replace('_', ' '), s=65)
    axes[0].axvline(1.5, color='gray', linestyle='--')
    axes[0].set_xticks(range(4), ['B0 F', 'B0 R', 'B2 F', 'B2 R'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend(fontsize=8)
    keys = ['shifted_fisher_residual_minus_old_rows', 'shifted_fisher_residual_minus_fisher_residual']
    columns = [(b, o, k) for k in keys for b, o in cases]
    heat = np.asarray([[banks[b]['orders'][o]['contrasts'][k]['per_task_pp'][t]
                        for b, o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(8), [f'B{b} {o[0].upper()}\nvs {"bilat" if k == keys[0] else "true F"}'
                                 for b, o, k in columns], fontsize=7)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Fisher residual: bank0 selection and unchanged bank2 follow-up')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'followup_bank_mean_gain_pp': result['followup_bank_mean_gain_pp'],
                      'all_four_case_mean_gain_pp': result['all_four_case_mean_gain_pp']}, indent=2))


if __name__ == '__main__':
    main()
