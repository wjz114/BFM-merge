"""Audit the joint-memory orientation control and its selected real counterpart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from report_streaming_complement_memory import audit_run
from run_streaming_strict_history import write_json


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_joint_memory_orientation_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    real_root = Path('runs/streaming_complement_memory_v1')
    real_plan = json.loads((real_root / 'plan.json').read_text())
    real_report = json.loads((real_root / 'comparison.json').read_text())
    assert all(real_report['checks'].values())
    for key in ['seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference',
                'history_representation', 'history_update_arithmetic', 'history_direction_scale']:
        assert plan[key] == real_plan[key], key
    for name, digest in real_plan['source_sha256'].items():
        assert plan['source_sha256'][name] == digest
    tasks, orders, sources, versions = plan['tasks'], {}, {}, set()
    for order in plan['orders']:
        folder = root / f'random_memory_{order}'
        random_run, random_cost = audit_run(folder, plan, base, source)
        assert random_run['condition'] == random_run['memory_policy'] == 'joint_memory'
        assert random_run['direction_policy'] == 'random' and random_run['merger_condition'] == 'random_sketch'
        sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        real_path = Path(plan['real_runs'][order]) / 'run.json'
        assert sha256(real_path) == real_report['sources'][str(real_path)]
        real = json.loads(real_path.read_text())
        sources[str(real_path)] = sha256(real_path)
        assert real['condition'] == real['memory_policy'] == 'joint_memory'
        runs = {'joint_memory': real, 'random_memory': random_run}
        costs = {'joint_memory': real_report['orders'][order]['audits_and_costs']['joint_memory'],
                 'random_memory': random_cost}
        for name, ref_folder in plan['reference_runs'][order].items():
            path = Path(ref_folder) / 'run.json'
            assert sha256(path) == real_report['sources'][str(path)]
            runs[name] = json.loads(path.read_text())
            costs[name] = real_report['orders'][order]['audits_and_costs'][name]
            sources[str(path)] = sha256(path)
        for run in runs.values():
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert run['dataset_audits'] == real['dataset_audits']
            assert run['order'] == real['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in real['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in real['stages'][1:]]
            assert run['stages'][0]['validation'] == real['stages'][0]['validation']
            versions.add((run['torch_version'], run['cuda_version']))
        for filename in ['coefficients.pt', 'merged.ckpt']:
            a = torch.load(real_path.parent / 'stage_02' / filename, map_location='cpu', weights_only=True)
            b = torch.load(folder / 'stage_02' / filename, map_location='cpu', weights_only=True)
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        validation = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        controls = ['random_memory', *plan['reference_runs'][order]]
        orders[order] = {
            'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in tasks])) for name, v in validation.items()},
            'contrasts': {f'joint_memory_minus_{name}': contrast(validation['joint_memory'], validation[name], tasks) for name in controls},
            'primary_gain_over_stages_pp': {str(s): 100 * (real['stages'][s - 1]['macro_BAcc'] - random_run['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']},
            'audits_and_costs': costs}
        print('checked', order, flush=True)
    assert len(versions) == 1
    keys = list(orders['forward']['contrasts'])
    result = {'sources': sources, 'orders': orders,
              'mean_gain_over_two_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys},
              'checks': {'both_new_runs_complete': True, 'strict_settings_recorded': True,
                         'actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'stage2_before_direction_intervention_exact_coefficients_and_native_weights': True,
                         'new_checkpoint_and_coefficient_hashes': True, 'new_fixed_coefficient_means_exact': True,
                         'new_native_exports_pass': True, 'new_joint_memory_lineages_replay_exactly': True,
                         'frozen_real_and_strict_reference_audits_match': True},
              'new_scientific_updates': 3240, 'new_scientific_CE_example_passes': 51840,
              'extra_score_VJPs': 0,
              'scope': 'Joint memory was selected from the earlier control after results were seen. These are two new orientation controls, not new independent expert banks or preregistered confirmation of the earlier candidate. Reused real runs and strict references retain their full frozen audits. Random residual slots match norms and capacity within their own evolving model; models and memory dictionaries can diverge across policies. The common memory update includes calibrated-model feedback. Exposed development validation and growing original512 replay; no official test read. Task deletion is sensitivity, not confidence intervals. One worker per GPU now versus two in the reused runs; elapsed times are descriptive, not a controlled speed comparison.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    names = ['incoming_rows', 'old_rows', 'history_sketch', 'random_sketch', 'joint_memory', 'random_memory']
    labels = ['Incoming\nrows', 'Bilateral\nrows', 'Raw\nhistory', 'Raw random\nhistory', 'Joint\nmemory', 'Joint random\nmemory']
    for j, order in enumerate(plan['orders']):
        axes[0].scatter(np.arange(6) + (j - .5) * .15, [100 * orders[order]['macro_BAcc'][n] for n in names], s=65, label=order)
    axes[0].set_xticks(np.arange(6), labels, fontsize=8)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    heat = np.asarray([[orders[o]['contrasts']['joint_memory_minus_random_memory']['per_task_pp'][t] for o in plan['orders']] for t in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Real joint memory minus matched random directions')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Orientation control for selected joint-memory policy — strict bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
