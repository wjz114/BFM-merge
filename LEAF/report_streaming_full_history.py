"""Audit the full causal history candidate on bank0 and bank2."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from streaming_full_history import initialize_history, update_memory
from run_streaming_strict_history import write_json


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert sha256(Path(run['plan'])) == run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == plan['updates_per_arm'] == 1620
    assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert run['history_used_by_merger'] and run['memory_policy'] == run['condition']
    memory = initialize_history(base, plan['active'])
    fixed, previous = set(base) - set(plan['active']), None
    training_seconds, memory_seconds, calibration_seconds = 0., 0., 0.
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        assert sha256(directory / 'merged.ckpt') == stage['checkpoint_sha256']
        checkpoint = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
        assert checkpoint.keys() == base.keys()
        assert all(torch.equal(checkpoint[name], base[name]) for name in fixed)
        assert all(torch.isfinite(checkpoint[name]).all() for name in plan['active'])
        expert = Path(source['experts'][task])
        assert sha256(expert) == plan['expert_sha256'][task]
        incoming = torch.load(expert, map_location='cpu', weights_only=True)
        if number == 1:
            assert stage['updates'] == stage['training_examples'] == 0
            assert all(torch.equal(checkpoint[name], incoming[name]) for name in base)
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            assert sha256(directory / 'coefficients.pt') == stage['coefficients_sha256']
            c = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
            n = plan['bilateral_coefficients'] + (number - 1) * plan['row_count']
            assert stage['history_slot_count'] == number - 1
            assert c['snapshots'].shape == (len(plan['average_rounds']), n)
            assert stage['coefficient_count'] == n
            assert torch.isfinite(c['snapshots']).all()
            assert torch.equal(c['snapshots'].double().mean(0).float(), c['mean'])
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(x['max_abs_logit_error'] <= 1e-5 and x['changed_predictions'] == 0
                       for x in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            training_seconds += stage['training_seconds']
            calibration_seconds += stage['calibration_call_seconds']
            del c
        reconstructed = update_memory(memory, base, previous, incoming, checkpoint,
                                      number - 1, plan['active'], run['condition'])
        for key, value in reconstructed.items():
            assert value == stage['memory_update'][key], (folder, number, key)
        assert reconstructed['state_bytes'] == number * plan['history_bytes_per_task']
        memory_seconds += stage['memory_update']['seconds']
        previous = checkpoint
        del incoming
    saved = torch.load(folder / 'history.pt', map_location='cpu', weights_only=True)
    assert saved['count'] == 16
    assert saved['history'].keys() == memory.keys()
    assert all(torch.equal(value, saved['history'][name]) for name, value in memory.items())
    history_bytes = sum(value.numel() * value.element_size() for value in memory.values())
    assert history_bytes == plan['history_bytes'] == 1648099328
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(item['validation_size'] for item in run['dataset_audits'].values()) == 60413
    costs = {'final_stage_coefficient_count': plan['bilateral_coefficients'] + 15 * plan['row_count'],
             'coefficient_counts_by_stage': [stage.get('coefficient_count', 0) for stage in run['stages']],
             'logical_history_bytes': history_bytes,
             'history_bytes_by_stage': [stage['memory_update']['state_bytes'] for stage in run['stages']],
             'memory_update_seconds': memory_seconds,
             'training_and_export_seconds': training_seconds,
             'calibration_including_setup_seconds': calibration_seconds,
             'elapsed_seconds': run['elapsed_seconds'],
             'peak_reserved_bytes': run['peak_reserved_bytes']}
    return run, costs



def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_full_history_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    previous_path = Path('runs/streaming_anchored_covariance_v1/comparison.json')
    previous_report = json.loads(previous_path.read_text())
    assert all(previous_report['checks'].values())
    original_report = json.loads(Path('runs/streaming_feedback_ablation_v1/comparison.json').read_text())
    assert all(original_report['checks'].values())
    banks, sources, versions = {}, {}, set()
    for seed in (0, 2):
        bank_root = root / f'seed{seed}'
        plan = json.loads((bank_root / 'plan.json').read_text())
        for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
            assert sha256(Path(path)) == digest, path
        source = json.loads(Path(plan['source_fit']).read_text())
        assert source['seed'] == plan['seed'] == seed
        base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
        tasks, orders = plan['tasks'], {}
        for order in plan['orders']:
            folder = bank_root / f'full_history_{order}'
            run, cost = audit_run(folder, plan, base, source)
            assert run['condition'] == run['memory_policy'] == 'full_history'
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            runs, costs = {'full_history': run}, {'full_history': cost}
            for condition, ref_folder in plan['reference_runs'][order].items():
                path = Path(ref_folder) / 'run.json'
                if condition == 'original_memory':
                    assert seed == 0 and sha256(path) == original_report['sources'][str(path)]
                    cost_reference = original_report['orders'][order]['audits_and_costs'][condition]
                else:
                    assert sha256(path) == previous_report['sources'][str(path)]
                    cost_reference = previous_report['banks'][str(seed)]['orders'][order]['audits_and_costs'][condition]
                runs[condition] = json.loads(path.read_text())
                costs[condition] = cost_reference
                sources[str(path)] = sha256(path)
            real = runs['joint_memory']
            real_plan = json.loads(Path(real['plan']).read_text())
            for key in ['seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                        'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference']:
                assert plan[key] == real_plan[key], key
            for name, digest in real_plan['source_sha256'].items():
                assert plan['source_sha256'][name] == digest
            for item in runs.values():
                assert item['status'] == 'completed' and len(item['stages']) == 16
                assert item['strict_deterministic_algorithms'] and item['cudnn_deterministic']
                assert item['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
                assert item['dataset_audits'] == real['dataset_audits']
                assert item['order'] == real['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
                assert [s.get('schedule_hash') for s in item['stages']] == [s.get('schedule_hash') for s in real['stages']]
                assert [s['input_split']['selected_ids_hash'] for s in item['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in real['stages'][1:]]
                assert item['stages'][0]['validation'] == real['stages'][0]['validation']
                versions.add((item['torch_version'], item['cuda_version']))
            reference_folder = Path(plan['reference_runs'][order]['joint_memory'])
            a = torch.load(reference_folder / 'stage_02' / 'merged.ckpt', map_location='cpu', weights_only=True)
            b = torch.load(folder / 'stage_02' / 'merged.ckpt', map_location='cpu', weights_only=True)
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            ca = torch.load(reference_folder / 'stage_02' / 'coefficients.pt', map_location='cpu', weights_only=True)
            cb = torch.load(folder / 'stage_02' / 'coefficients.pt', map_location='cpu', weights_only=True)
            for key in ('snapshots', 'mean'):
                assert torch.equal(ca[key][..., :347666], cb[key][..., :347666])
                assert torch.count_nonzero(cb[key][..., 347666:]) == 0
            for number, stage in enumerate(run['stages'], 1):
                checkpoint = folder / f'stage_{number:02d}' / 'merged.ckpt'
                assert checkpoint.is_symlink() and checkpoint.resolve() == Path(stage['checkpoint_storage_path'])
            validation = {name: item['stages'][-1]['validation'] for name, item in runs.items()}
            orders[order] = {
                'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in tasks])) for name, v in validation.items()},
                'contrasts': {f'full_history_minus_{name}': contrast(validation['full_history'], validation[name], tasks) for name in plan['reference_runs'][order]},
                'gain_over_stages_pp': {name: {str(s): 100 * (run['stages'][s - 1]['macro_BAcc'] - runs[name]['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']} for name in plan['reference_runs'][order]},
                'audits_and_costs': costs}
            print('checked', seed, order, flush=True)
        keys = list(orders['forward']['contrasts'])
        banks[str(seed)] = {'orders': orders, 'mean_gain_over_orders_pp': {key: float(np.mean([r['contrasts'][key]['gain_pp'] for r in orders.values()])) for key in keys}}
    assert len(versions) == 1
    result = {'sources': sources, 'banks': banks,
              'mean_gain_over_two_banks_pp': {key: float(np.mean([b['mean_gain_over_orders_pp'][key] for b in banks.values()])) for key in keys},
              'checks': {'four_new_complete_runs': True, 'strict_settings_recorded': True,
                         'within_bank_actual_data_schedules_auxiliary_ids_match': True, 'first_expert_unchanged': True,
                         'stage2_exact_bilateral_coefficients_zero_history_gates_and_native_weights': True,
                         'new_checkpoint_and_coefficient_hashes': True, 'new_fixed_coefficient_means_exact': True,
                         'new_native_exports_pass': True, 'new_original_only_full_history_lineages_replay_exactly': True,
                         'frozen_reference_audits_match': True, 'native_storage_links_resolve_to_audited_paths': True, 'growing_history_and_coefficient_costs_checked': True},
              'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680, 'extra_score_VJPs': 0,
              'scope': 'Larger-state causal reference retaining every arrived original expert delta. No sketch rank truncation beyond the inherited numerical floor, no feedback observations, and no future expert access during training. State grows to1648099328bytes and final-stage coefficients to2949416, so this is not a capacity-matched two-slot comparison. Data/update/optimizer/averaging protocol is held fixed. Two exposed development banks and correlated orders, no independent holdout. No novelty or accuracy-upper-bound claim. Single ordinary inference model. Shared-hardware times are descriptive; setup-inclusive calibration times are separately recorded.'}

    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.3]})
    columns = [(str(seed), order) for seed in (0, 2) for order in ('forward', 'reverse')]
    for j, name in enumerate(['old_rows', 'anchored_covariance', 'full_history']):
        axes[0].scatter(np.arange(4) + (j - 1) * .15, [100 * banks[s]['orders'][o]['macro_BAcc'][name] for s, o in columns], s=65, label=name.replace('_', ' '))
    labels = ['Bank0\nforward', 'Bank0\nreverse', 'Bank2\nforward', 'Bank2\nreverse']
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    heat = np.asarray([[banks[s]['orders'][o]['contrasts']['full_history_minus_old_rows']['per_task_pp'][t] for s, o in columns] for t in tasks])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(np.arange(4), labels, fontsize=9)
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Full causal history minus bilateral rows')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Full causal history — larger-state reference on two exposed banks')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'banks': {s: {'macros': {o: v['macro_BAcc'] for o, v in b['orders'].items()}, 'gain_pp': b['mean_gain_over_orders_pp']} for s, b in banks.items()}, 'mean_gain_over_banks_pp': result['mean_gain_over_two_banks_pp']}, indent=2))


if __name__ == '__main__':
    main()
