"""Audit paired-block selection and its same-information random control."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from report_streaming_bilateral_merge import contrast
from run_streaming_strict_history import write_json


def audit_selection(payload, record, task, stage, plan, original, fresh, dataset_audit):
    g = payload['gradients'].double()
    assert g.shape == (2, 32, 347666) and torch.isfinite(g).all()
    assert array_hash(payload['gradients'].numpy()) == record['gradient_hash']
    assert np.array_equal(payload['original_indices'].numpy(), original[task])
    b = payload['fresh_indices'].numpy()
    assert set(b.tolist()) == set(fresh[task].tolist())
    scale = g.square().mean((0, 1)).sqrt() + plan['selection_epsilon']
    assert torch.equal(scale, payload['scale'])
    differences = (g[0] - g[1]) / (2 * scale)
    gram = differences @ differences.T
    assert torch.equal(gram, payload['gram'])
    rng = np.random.default_rng([plan['seed'], plan['tasks'].index(task), 41629])
    signs = torch.from_numpy(rng.choice([-1., 1.], size=32))
    assert torch.equal(signs, payload['initial_signs'])
    initial_objective = float(signs @ gram @ signs)
    assert initial_objective == record['initial_objective']
    if record['condition'] == 'gradient_match':
        trace = record['descent']
        assert len(trace['flips']) <= 64
        for j, k in enumerate(trace['flips']):
            alternatives = signs.repeat(32, 1)
            alternatives[torch.arange(32), torch.arange(32)] *= -1
            objectives = torch.einsum('bi,ij,bj->b', alternatives, gram, alternatives)
            assert int(objectives.argmin()) == k
            signs[k] *= -1
            objective = float(signs @ gram @ signs)
            assert abs(objective - trace['objectives'][j + 1]) < 1e-7
            assert objective < trace['objectives'][j]
        if len(trace['flips']) < 64:
            changes = 4 * (gram.diag() - signs * (gram @ signs))
            assert float(changes.min()) >= -trace['stopping_tolerance']
    else:
        assert not record['descent']['flips']
    assert torch.equal(signs, payload['selected_signs'])
    selected_g = torch.stack([g[0 if s > 0 else 1, k] for k, s in enumerate(signs)])
    direct_objective = float((((selected_g.mean(0) - g.mean((0, 1))) / scale).square().sum()))
    objective = float(signs @ gram @ signs)
    assert abs(direct_objective * 32 ** 2 - objective) < 1e-9 * max(objective, 1.)
    assert objective == record['selected_objective'] <= initial_objective + 1e-8
    default = torch.zeros(347666)
    n = len(plan['active'])
    default[:n] = (stage - 1) / stage
    default[n:2 * n] = 1 / stage
    assert torch.equal(default, payload['initial_coefficients'])
    assert array_hash(default.numpy()) == record['initial_coefficients_hash']
    use_b = np.repeat(signs.numpy() < 0, 16)
    selected = np.where(use_b, b, original[task])
    assert len(np.unique(selected)) == 512
    assert np.array_equal(selected, payload['selected_indices'].numpy())
    assert array_hash(selected) == record['selected_indices_hash']
    with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
        rows, labels = row_map(h, task, 'calibration')
        labels = labels.astype(np.int64)
        assert np.array_equal(labels[b], labels[original[task]])
        assert np.array_equal(labels[selected], labels[original[task]])
        common = set(original[task].tolist()) & set(b.tolist())
        assert all(b[k] == row for k, row in enumerate(original[task]) if row in common)
        x = selected_eeg(h, [rows[int(i)] for i in selected])
    assert array_hash(x) == record['selected_eeg_hash'] == dataset_audit['calibration_eeg_hash']
    assert array_hash(labels[selected]) == record['selected_labels_hash'] == dataset_audit['calibration_labels_hash']
    assert record['pool_unique_rows'] == len(np.union1d(original[task], b))
    assert record['shared_rows_at_identical_positions'] == len(common)
    assert len(common) == (108 if task == 'ADHD_AliMotie' else 0)
    assert record['selection_examples'] == 1024 and record['selection_backwards'] == 64
    assert record['optimizer_updates'] == 0 and record['retained_unique_rows'] == 512
    return {'gradient_objective_ratio': objective / initial_objective if initial_objective else 1.,
            'fresh_blocks': record['fresh_blocks'], 'elapsed_seconds': record['elapsed_seconds']}


def audit_run(folder, plan, base, source):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 1620 and run['training_examples'] == 25920
    assert run['selection_examples'] == 15360 and run['selection_backwards'] == 960
    assert not run['history_used_by_merger'] and run['persistent_auxiliary_history_bytes'] == 0
    assert not run['official_test_read']
    original, fresh = np.load(plan['calibration']), np.load(plan['fresh_calibration'])
    fixed = set(base) - set(plan['active'])
    audits = {}
    for number, (task, stage) in enumerate(zip(run['order'], run['stages']), 1):
        assert stage['stage'] == number and stage['arriving_task'] == task
        directory = folder / f'stage_{number:02d}'
        checkpoint = directory / 'merged.ckpt'
        assert checkpoint.is_symlink() and checkpoint.resolve() == Path(stage['checkpoint_storage_path'])
        assert sha256(checkpoint) == stage['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[n], base[n]) for n in fixed)
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        if number == 1:
            expert = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
            assert all(torch.equal(state[n], expert[n]) for n in state)
            assert stage['updates'] == stage['selection_examples'] == stage['selection_backwards'] == 0
            assert task not in run['selection_records']
            del expert
        else:
            assert stage['updates'] == number * 12 and stage['training_examples'] == number * 192
            for name, key in [('coefficients.pt', 'coefficients_sha256'), ('selection.pt', 'selection_sha256')]:
                assert (directory / name).is_symlink() and sha256(directory / name) == stage[key]
            coefficients = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
            assert coefficients['snapshots'].shape == (6, 347666)
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert all(torch.isfinite(v).all() for v in coefficients.values())
            assert stage['coefficient_count'] == 347666
            assert set(stage['export_checks']) == set(run['order'][:number])
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0 for c in stage['export_checks'].values())
            assert stage['input_split']['old_tasks'] == run['order'][:number - 1]
            payload = torch.load(directory / 'selection.pt', map_location='cpu', weights_only=True)
            assert stage['selection'] == run['selection_records'][task]
            audits[task] = audit_selection(payload, stage['selection'], task, number, plan,
                                           original, fresh, run['dataset_audits'][task])
            del payload, coefficients
        del state
    assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
    assert sum(v['validation_size'] for v in run['dataset_audits'].values()) == 60413
    return run, {'selection': audits, 'selection_seconds': sum(a['elapsed_seconds'] for a in audits.values()),
                 'elapsed_seconds': run['elapsed_seconds'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
                 'coefficient_count': 347666, 'selection_examples': 15360, 'selection_backwards': 960}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_gradient_selection_v1')
    assert not (root / 'comparison.json').exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    prior = json.loads(Path(plan['reference_report']).read_text())
    assert all(prior['checks'].values())
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    orders, sources, versions = {}, {}, set()
    for task in plan['tasks']:
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
    for order in plan['orders']:
        runs, costs = {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            runs[condition], costs[condition] = audit_run(folder, plan, base, source)
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        reference_path = Path(plan['reference_runs'][order]['old_rows']) / 'run.json'
        assert sha256(reference_path) == prior['sources'][str(reference_path)]
        reference = json.loads(reference_path.read_text())
        sources[str(reference_path)] = sha256(reference_path)
        runs['old_rows'] = reference
        reference_plan = json.loads(Path(reference['plan']).read_text())
        for key in ('seed', 'tasks', 'active', 'rounds', 'average_rounds', 'batch_size', 'learning_rate',
                    'validation_stages', 'optimizer', 'precision', 'calibration', 'calibration_reference'):
            assert plan[key] == reference_plan[key], key
        for run in runs.values():
            assert run['order'] == reference['order']
            assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
            assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in reference['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in reference['stages'][1:]]
            assert run['stages'][0]['validation'] == reference['stages'][0]['validation']
            for task in plan['tasks']:
                a, b = run['dataset_audits'][task], reference['dataset_audits'][task]
                assert {k: v for k, v in a.items() if k != 'calibration_eeg_hash'} == {k: v for k, v in b.items() if k != 'calibration_eeg_hash'}
            versions.add((run['torch_version'], run['cuda_version']))
        # At the second arrival, before the first differing training batch, both
        # arms must score exactly the same current model and input pool.
        assert runs['gradient_match']['stages'][1]['selection']['gradient_hash'] == runs['random_blocks']['stages'][1]['selection']['gradient_hash']
        pairs = [('gradient_match', 'random_blocks'), ('gradient_match', 'old_rows'), ('random_blocks', 'old_rows')]
        final = {name: run['stages'][-1]['validation'] for name, run in runs.items()}
        orders[order] = {'macro_BAcc': {name: float(np.mean([v[t]['BAcc'] for t in plan['tasks']])) for name, v in final.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(final[a], final[b], plan['tasks']) for a, b in pairs},
            'gain_over_stages_pp': {f'{a}_minus_{b}': {str(s): 100 * (runs[a]['stages'][s - 1]['macro_BAcc'] - runs[b]['stages'][s - 1]['macro_BAcc']) for s in plan['validation_stages']} for a, b in pairs},
            'audits_and_costs': costs}
        print('checked', order, flush=True)
    assert len(versions) == 1
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'orders': orders,
        'mean_gain_over_orders_pp': {k: float(np.mean([o['contrasts'][k]['gain_pp'] for o in orders.values()])) for k in orders['forward']['contrasts']},
        'checks': {k: True for k in ['four_complete_runs', 'frozen_sources_and_inputs_match', 'same_training_labels_weights_and_schedules',
            'same_candidate_pools_and_scoring_budget', 'independent_gradient_objective_and_greedy_choices_pass',
            'actual_selected_EEG_reloaded_and_hashed', 'selected_rows_unique', 'fixed_coefficient_means_exact',
            'bilateral_initial_coefficients_exact', 'checkpoint_and_payload_hashes_pass', 'first_expert_unchanged',
            'native_exports_pass', 'strict_settings_match', 'second_arrival_scoring_exact_across_arms', 'storage_links_resolve']},
        'new_scientific_updates': 6480, 'new_training_CE_examples': 103680,
        'new_selection_examples': 61440, 'new_selection_backwards': 3840,
        'scope': 'One exposed development bank, two correlated task orders. New arms have matched1024-slot candidate pools and scoring costs,512 retained unique rows/task, and unchanged347666-coefficient bilateral training. Original512 reference has lower data access and scoring cost; comparisons to it do not isolate selection intelligence. ADHD candidate multiset has108 repeated rows, aligned to prevent repeats in the retained512. Task-deletion ranges are sensitivity, not confidence intervals. Gradient objective reduction alone does not establish better generalization, novelty or research-goal completion.'}
    write_json(root / 'comparison.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    for i, name in enumerate(('old_rows', 'random_blocks', 'gradient_match')):
        axes[0].scatter(np.arange(2) + (i - 1) * .12,
            [100 * orders[o]['macro_BAcc'][name] for o in plan['orders']], label=name.replace('_', ' '), s=75)
    axes[0].set_xticks([0, 1], ['Forward', 'Reverse'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    columns = [(o, k) for k in ('gradient_match_minus_random_blocks', 'gradient_match_minus_old_rows') for o in plan['orders']]
    heat = np.asarray([[orders[o]['contrasts'][k]['per_task_pp'][t] for o, k in columns] for t in plan['tasks']])
    limit = max(float(np.abs(heat).max()), .01)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(4), ['F vs random', 'R vs random', 'F vs original512', 'R vs original512'], fontsize=8)
    axes[1].set_yticks(range(16), [t.replace('MI_', '').replace('EMO_', '') for t in plan['tasks']], fontsize=8)
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Arrival-time merger-gradient calibration selection — bank0')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'comparison.{ext}', dpi=180)
    print(json.dumps({'macro_BAcc': {o: v['macro_BAcc'] for o, v in orders.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
