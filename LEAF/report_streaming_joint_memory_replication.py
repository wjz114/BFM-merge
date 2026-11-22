"""Audit bank1 replication and retain the selected bank0 comparison separately."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from report_streaming_complement_memory import audit_run as audit_joint
from report_streaming_strict_history import audit_run as audit_bilateral
from run_streaming_strict_history import write_json


def summarize(runs, costs, plan):
    tasks, orders = plan['tasks'], {}
    for order in plan['orders']:
        pair = runs[order]
        reference = pair['old_rows']
        for run in pair.values():
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == reference['dataset_audits']
            assert run['order'] == reference['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
        validation = {name: run['stages'][-1]['validation'] for name, run in pair.items()}
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in tasks])) for name, v in validation.items()},
            'joint_memory_minus_old_rows': contrast(validation['joint_memory'], validation['old_rows'], tasks),
            'gain_over_stages_pp': {str(s): 100 * (pair['joint_memory']['stages'][s - 1]['macro_BAcc'] - reference['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']},
            'audits_and_costs': costs[order]}
    return {'orders': orders, 'mean_gain_over_two_orders_pp': float(np.mean([v['joint_memory_minus_old_rows']['gain_pp'] for v in orders.values()]))}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_joint_memory_replication_v1/seed1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    assert source['seed'] == plan['seed'] == 1
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    bank0_report = json.loads(Path('runs/streaming_complement_memory_v1/comparison.json').read_text())
    assert all(bank0_report['checks'].values())
    bank0_plan = json.loads(Path('runs/streaming_complement_memory_v1/plan.json').read_text())
    for key in ['tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                'validation_stages', 'optimizer', 'precision', 'history_representation',
                'history_update_arithmetic', 'history_direction_scale']:
        assert plan[key] == bank0_plan[key], key
    for name, digest in bank0_plan['source_sha256'].items():
        assert plan['source_sha256'][name] == digest
    banks, sources, versions = {}, {}, set()
    for seed in (0, 1):
        runs, costs = {}, {}
        for order in plan['orders']:
            runs[order], costs[order] = {}, {}
            for condition in plan['conditions']:
                folder = Path(plan['bank0_runs'][order][condition]) if seed == 0 else root / f'{condition}_{order}'
                path = folder / 'run.json'
                if seed == 0:
                    assert sha256(path) == bank0_report['sources'][str(path)]
                    run = json.loads(path.read_text())
                    cost = bank0_report['orders'][order]['audits_and_costs'][condition]
                elif condition == 'old_rows':
                    run, cost, history = audit_bilateral(folder, plan, base, source)
                    del history
                else:
                    run, cost = audit_joint(folder, plan, base, source)
                assert run['condition'] == condition
                run_plan = json.loads(Path(run['plan']).read_text())
                assert run_plan['seed'] == seed
                runs[order][condition], costs[order][condition] = run, cost
                sources[str(path)] = sha256(path)
                versions.add((run['torch_version'], run['cuda_version']))
                print('checked', seed, order, condition, flush=True)
        banks[str(seed)] = summarize(runs, costs, plan)
    assert len(versions) == 1
    gains = [banks[str(seed)]['mean_gain_over_two_orders_pp'] for seed in (0, 1)]
    result = {'sources': sources, 'banks': banks,
              'mean_gain_over_two_banks_pp': float(np.mean(gains)),
              'sample_sd_of_bank_means_pp': float(np.std(gains, ddof=1)),
              'checks': {'four_new_complete_runs': True, 'strict_settings_recorded': True,
                         'within_bank_actual_data_schedules_auxiliary_ids_match': True,
                         'first_expert_unchanged': True, 'new_checkpoint_and_coefficient_hashes': True,
                         'new_fixed_coefficient_means_exact': True, 'new_native_exports_pass': True,
                         'new_raw_and_joint_memory_lineages_replay_exactly': True,
                         'bank0_frozen_audit_matches': True, 'same_algorithm_and_training_hyperparameters': True},
              'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
              'extra_score_VJPs': 0,
              'scope': 'One newly tested expert-training seed and two correlated orders, compared to the selected bank0. The development validation sets were exposed in earlier research, including other bank1 methods; this is not a pristine holdout or official test. Bank means and their two-value sample SD are descriptive, not a reliable population uncertainty estimate. Task deletion is sensitivity, not confidence intervals. Replay grows with tasks. Bilateral runs construct an unused raw history for their frozen audit, with zero logical history requirement. Joint memory consumes two model-sized auxiliary slots. Four workers share hardware; elapsed times are descriptive.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.3]})
    columns = [(str(seed), order) for seed in (0, 1) for order in plan['orders']]
    for j, name in enumerate(plan['conditions']):
        axes[0].scatter(np.arange(4) + (j - .5) * .15, [100 * banks[s]['orders'][o]['macro_BAcc'][name] for s, o in columns], s=65, label=name.replace('_', ' '))
    labels = ['Bank0\nforward', 'Bank0\nreverse', 'Bank1\nforward', 'Bank1\nreverse']
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    tasks = plan['tasks']
    heat = np.asarray([[banks[s]['orders'][o]['joint_memory_minus_old_rows']['per_task_pp'][t] for s, o in columns] for t in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(np.arange(4), labels, fontsize=9)
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Joint memory minus bilateral rows')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Selected bank0 and new bank1 — unchanged joint-memory operator')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'banks': {s: {'macros': {o: v['macro_BAcc'] for o, v in b['orders'].items()}, 'gain_pp': b['mean_gain_over_two_orders_pp']} for s, b in banks.items()},
                      'mean_gain_over_banks_pp': result['mean_gain_over_two_banks_pp']}, indent=2))


if __name__ == '__main__':
    main()
