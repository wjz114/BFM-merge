"""Audit coupled conditional factors against factorized and bilateral controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256, prepare_text, normalized
from load_config import build_model_config, load_yaml
from report_streaming_bilateral_merge import contrast
from run_streaming_strict_history import write_json


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(Path(run['plan'])) == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    assert run['condition'] in plan['conditions']
    assert run['history_used_by_merger'] and run['memory_policy'] == 'original_expert_conditional_factors'
    assert run['extra_direction_construction_forward_examples'] == run['extra_direction_construction_backward_probes'] == 0
    config = build_model_config(load_yaml(Path('configs/LEAF_mpnet.yaml')))
    history_path = folder / 'history.pt'
    assert history_path.is_symlink()
    assert sha256(history_path) == run['history_sha256']
    stored = torch.load(history_path, map_location='cpu', weights_only=True)
    assert stored['count'] == len(stored['history']) == 16
    history = stored['history']
    fixed = set(base) - set(plan['active'])
    for number, (task, stage, row) in enumerate(zip(run['order'], run['stages'], history), 1):
        assert stage['stage'] == number and stage['arriving_task'] == row['task'] == task
        assert stage['history_arrival'] == {'task': task, 'expert_sha256': plan['expert_sha256'][task],
                                             'seen_experts': number}
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
        original = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        assert set(row) == {'task', 'columns', 'scale', 'instruction'}
        assert row['columns'].shape == (256, 2048) and row['columns'].dtype == torch.float32
        assert row['scale'].shape == (256,) and row['scale'].dtype == torch.float64
        assert row['instruction'].shape == (769,) and row['instruction'].dtype == torch.float64
        assert all(torch.isfinite(row[k]).all() for k in ('columns', 'scale', 'instruction'))
        # Rebuild the saved original payload independently of the online helper.
        expected_columns = torch.cat([original[f'qformer.layers.{i}.cross_attn.in_proj_weight'][256:]
                                      for i in range(4)], dim=0).T.contiguous()
        assert torch.equal(expected_columns, row['columns'])
        labels, text, instruction, _ = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        assert row['instruction'][-1] == 1.
        assert torch.allclose(row['instruction'][:-1], normalized(instruction, 'instruction').double(), atol=2e-8, rtol=2e-7)
        expected_scale = 1 + (original['qformer.film.weight'][:256].double() @ row['instruction'][:-1] + original['qformer.film.bias'][:256].double())
        assert torch.equal(expected_scale, row['scale'])
        directory = folder / f'stage_{number:02d}'
        checkpoint_path = directory / 'merged.ckpt'
        assert checkpoint_path.is_symlink() and checkpoint_path.resolve() == Path(stage['checkpoint_storage_path'])
        assert sha256(checkpoint_path) == stage['checkpoint_sha256']
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[name], base[name]) for name in fixed)
        assert all(torch.isfinite(state[name]).all() for name in plan['active'])
        assert stage['history_bytes'] == number * 2105352
        assert stage['retained_coefficient_bytes'] == 0
        if number == 1:
            assert all(torch.equal(state[name], original[name]) for name in state)
            assert stage['updates'] == stage['training_examples'] == 0
        else:
            assert stage['updates'] == number * plan['rounds']
            assert stage['training_examples'] == stage['updates'] * plan['batch_size']
            coefficient_path = directory / 'coefficients.pt'
            assert coefficient_path.is_symlink()
            assert sha256(coefficient_path) == stage['coefficients_sha256']
            coefficients = torch.load(coefficient_path, map_location='cpu', weights_only=True)
            assert coefficients['snapshots'].shape == (6, 347922)
            assert coefficients['initial'].shape == coefficients['mean'].shape == (347922,)
            assert stage['coefficient_count'] == 347922
            assert all(torch.isfinite(v).all() for v in coefficients.values())
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            default = torch.zeros_like(coefficients['initial'])
            n = len(plan['active'])
            default[:n] = (number - 1) / number
            default[n:2 * n] = 1 / number
            assert torch.equal(coefficients['initial'], default)
            assert array_hash(coefficients['initial'].numpy()) == stage['initial_coefficient_hash']
            audit = stage['factor_audit']
            assert audit['seen_experts'] == number and audit['state_bytes'] == number * 2105352
            assert audit['relative_optimality_gap'] < 1e-10
            assert audit['max_affine_scale_reconstruction_error'] < 1e-9
            assert audit['max_relative_column_norm_error'] < 1e-12
            x = torch.stack([r['instruction'] for r in history[:number]])
            singular = torch.linalg.svdvals(x)
            assert audit['instruction_rank'] == int((singular > 1e-10 * singular.max()).sum())
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0 for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
        del original, state
    assert history_path.resolve() == Path(run['stages'][-1]['checkpoint_storage_path']).parent.parent / 'history.pt'
    assert sum(v.numel() * v.element_size() for row in history for v in row.values() if isinstance(v, torch.Tensor)) == 33685632
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(x['validation_size'] for x in run['dataset_audits'].values()) == 60413
    costs = {'coefficient_count': 347922, 'extra_shared_channel_coefficients': 256,
             'final_history_bytes': 33685632, 'history_bytes_per_expert': 2105352,
             'algorithmic_retained_coefficient_bytes': 0,
             'factor_fit_by_stage': {str(s['stage']): s['factor_audit'] for s in run['stages'][1:]},
             'history_update_seconds': sum(s['history_update_seconds'] for s in run['stages']),
             'training_and_export_seconds': sum(s['training_seconds'] for s in run['stages'][1:]),
             'calibration_including_setup_seconds': sum(s['calibration_call_seconds'] for s in run['stages'][1:]),
             'elapsed_seconds': run['elapsed_seconds'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
             'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0}
    return run, costs


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_conditional_factor_v1')
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
        pairs = [('coupled_factors', 'old_rows'), ('coupled_factors', 'factorized_control'),
                 ('factorized_control', 'old_rows'), ('coupled_factors', 'raw_columns')]
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
        'checks': {'four_new_complete_runs': True, 'actual_data_schedules_auxiliary_ids_match': True,
                   'strict_settings_match': True, 'first_expert_unchanged': True,
                   'checkpoint_and_coefficient_hashes_pass': True, 'fixed_coefficient_means_exact': True,
                   'bilateral_default_and_zero_extra_initial_coefficients_exact': True,
                   'conditional_factor_objective_records_pass': True,
                   'original_expert_columns_and_responses_replay_exact': True,
                   'normalized_instruction_provenance_matches': True,
                   'native_exports_pass': True, 'coefficient_and_state_counts_match': True,
                   'storage_links_resolve': True, 'frozen_reference_audits_match': True},
        'new_scientific_updates': 6480, 'new_scientific_CE_example_passes': 103680,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'scope': 'One exposed bank, two correlated arrival orders. Both arms preserve the bilateral initial native model and add256 shared channel gates, total347922. Coupled targets retain each original expert projection/modulation association; control replaces expert projection columns with their mean before forming targets. Both use the same conditional rank-one factorization, reference-norm/sign gauge and minimum-change affine instruction fit. Original projection columns and own-instruction responses grow with task count during training (33685632bytes at16); inference uses one ordinary shared LEAF checkpoint. The construction fits conditional linear token-to-K/V maps, not beta/position effects or complete model functions. No additional EEG passes for direction construction. Task-deletion ranges are sensitivity, not confidence intervals; no superiority or novelty is inferred from engineering alone.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'raw_columns', 'factorized_control', 'coupled_factors')):
        axes[0].scatter(np.arange(2) + (i - 1.5) * .10,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('coupled_factors_minus_old_rows', 'coupled_factors_minus_factorized_control') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs bilateral', 'R vs bilateral', 'F vs control', 'R vs control'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Coupled conditional projection factors — bank0')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
