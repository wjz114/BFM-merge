"""Audit the fixed streaming projected spaces and compare strong references."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from row_projected_adam import GroupedBasis
from run_streaming_subspace_merge import write_json
from streaming_correction_retention import initialize


def audit_run(folder, plan, source, base):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(Path(run['plan']))
    assert run['updates'] == plan['updates_per_arm'] == 1620
    assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
    assert run['official_test_read'] is False
    active, condition = plan['active'], run['condition']
    previous, maximum_orthogonality, maximum_coordinate_error = None, 0., 0.
    norm_errors, scales, ranks_by_stage = [], [], {}
    for stage in run['stages']:
        number = stage['stage']
        stage_dir = folder / f'stage_{number:02d}'
        assert sha256(stage_dir / 'merged.ckpt') == stage['checkpoint_sha256']
        state = torch.load(stage_dir / 'merged.ckpt', map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[n], base[n]) for n in base if n not in active)
        assert all(torch.isfinite(state[n]).all() for n in active)
        expert_path = Path(source['experts'][stage['arriving_task']])
        assert sha256(expert_path) == plan['expert_sha256'][stage['arriving_task']]
        incoming = torch.load(expert_path, map_location='cpu', weights_only=True)
        if number == 1:
            assert all(torch.equal(state[n], incoming[n]) for n in base)
        else:
            assert sha256(stage_dir / 'snapshots.pt') == stage['snapshots_sha256']
            assert sha256(stage_dir / 'bases.pt') == stage['bases_sha256']
            saved = torch.load(stage_dir / 'snapshots.pt', map_location='cpu', weights_only=True)
            assert saved['rounds'] == plan['average_rounds'] and len(saved['snapshots']) == 6
            snapshots = saved['snapshots']
            assert all(set(s) == set(active) for s in snapshots)
            bases = torch.load(stage_dir / 'bases.pt', map_location='cpu', weights_only=True)
            assert list(bases) == active
            initial = initialize(previous, incoming, active, number - 1, 'recursive')
            coordinates = [torch.load(stage_dir / f'projection_coordinates_{rnd * number}.pt',
                                      map_location='cpu', weights_only=True) for rnd in plan['average_rounds']]
            assert all(list(c) == active for c in coordinates)
            rank_counts, basis_bytes = {}, 0
            for name in active:
                mean = torch.stack([s[name].double() for s in snapshots]).mean(0).float()
                assert torch.equal(mean, state[name]), (folder, number, name)
                record = bases[name]
                q = record['q']
                assert record['permutation'] is None and tuple(record['shape']) == tuple(state[name].shape)
                assert q.shape[1] == (2 if condition == 'bilateral' else 4)
                mask = q.square().sum(-1) > .5
                error = float((q @ q.transpose(1, 2) - torch.diag_embed(mask.float())).abs().max())
                maximum_orthogonality = max(maximum_orthogonality, error)
                assert error <= 2e-5
                rank, counts = torch.unique(mask.sum(-1), return_counts=True)
                for a, b in zip(rank.tolist(), counts.tolist()):
                    rank_counts[str(a)] = rank_counts.get(str(a), 0) + b
                if number == 2 and condition != 'bilateral':
                    assert torch.count_nonzero(q[:, 2:]) == 0
                if number <= 3:
                    assert int(mask.sum(-1).max()) <= number
                beta = torch.stack([c[name].double() for c in coordinates]).mean(0)
                # Independent FP64 reconstruction avoids requiring identical
                # CPU/GPU accumulation order for rounded native checkpoints.
                expected = initial[name].double() + GroupedBasis(q.double(), record['shape']).decode(beta)
                coordinate_error = float((expected - state[name].double()).abs().max())
                maximum_coordinate_error = max(maximum_coordinate_error, coordinate_error)
                assert coordinate_error <= 1e-6
                basis_bytes += q.numel() * q.element_size()
            assert basis_bytes == plan['padded_basis_bytes'][condition]
            assert rank_counts == stage['basis_audit']['per_row_effective_rank_counts']
            ranks_by_stage[str(number)] = rank_counts
            dynamics = [json.loads(line) for line in (stage_dir / 'projection_dynamics.jsonl').read_text().splitlines()]
            assert len(dynamics) == stage['updates'] == 12 * number
            assert [d['step'] for d in dynamics] == list(range(1, len(dynamics) + 1))
            for row in dynamics:
                proposed, projected, scale = (row[k] for k in
                    ('adam_proposal_norm', 'projected_proposal_norm', 'restoration_scale'))
                assert all(math.isfinite(x) and x >= 0 for x in (proposed, projected, scale))
                assert row['zero_projection'] == (projected == 0)
                expected_scale = proposed / projected if projected else 0.
                assert math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12)
                scales.append(scale)
                ratio = row['actual_to_proposal_norm_ratio']
                if ratio is not None:
                    assert math.isfinite(ratio)
                    norm_errors.append(abs(ratio - 1))
            assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                       for v in stage['export_checks'].values())
            assert stage['coordinate_average_audit']['max_abs_weight_mean_vs_coordinate_mean'] <= 1e-6
            del saved, snapshots, bases, initial, coordinates, expected
        previous = state
    assert run['persistent_sketch_bytes'] == plan['persistent_sketch_bytes'][condition]
    if condition == 'bilateral':
        assert not (folder / 'history.pt').exists()
    return run, {'orthogonality_max_abs_error': maximum_orthogonality,
                 'native_mean_vs_FP64_coordinate_mean_max_abs_error': maximum_coordinate_error,
                 'max_actual_step_norm_relative_error': max(norm_errors),
                 'mean_restoration_scale': float(np.mean(scales)),
                 'per_row_effective_ranks_by_stage': ranks_by_stage}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_projected_sketch_v2')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    tasks, reports, sources = plan['tasks'], {}, {}
    for order in ('forward', 'reverse'):
        expected_order = tasks if order == 'forward' else list(reversed(tasks))
        runs, audits, costs = {}, {}, {}
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run, audit = audit_run(folder, plan, source, base)
            assert run['order'] == expected_order and run['condition'] == condition
            runs[condition], audits[condition] = run, audit
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
            costs[condition] = {k: run[k] for k in ('elapsed_seconds', 'peak_reserved_bytes', 'persistent_sketch_bytes')}
            costs[condition].update(training_seconds=sum(s.get('training_seconds', 0) for s in run['stages']),
                                    padded_basis_bytes=plan['padded_basis_bytes'][condition],
                                    native_adam_moment_bytes=plan['native_adam_moment_bytes'])
            print(f'checked {order} {condition}', flush=True)
        histories = [torch.load(root / f'{c}_{order}' / 'history.pt', map_location='cpu', weights_only=True)
                     for c in ('history_sketch', 'random_sketch')]
        assert histories[0]['count'] == histories[1]['count'] == 16
        assert set(histories[0]['sketch']) == set(histories[1]['sketch']) == set(plan['active'])
        assert all(torch.equal(v, histories[1]['sketch'][n]) for n, v in histories[0]['sketch'].items())
        assert sum(v.numel() * v.element_size() for v in histories[0]['sketch'].values()) == 206012416
        del histories
        for name, pattern in plan['secondary_comparisons'].items():
            path = Path(pattern.format(order=order)) / 'run.json'
            run = json.loads(path.read_text())
            assert run['status'] == 'completed' and run['order'] == expected_order
            assert run['plan_sha256'] == sha256(Path(run['plan']))
            runs[name] = run
            sources[str(path)] = sha256(path)
        baseline = runs['bilateral']
        for run in runs.values():
            assert run['dataset_audits'] == baseline['dataset_audits']
            assert run['updates'] == 1620 and run['training_examples'] == 25920
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in baseline['stages']]
            assert run['stages'][0]['validation'] == baseline['stages'][0]['validation']
        validation = {k: v['stages'][-1]['validation'] for k, v in runs.items()}
        pairs = [('history_sketch', 'bilateral'), ('history_sketch', 'random_sketch'),
                 ('history_sketch', 'coefficient_history'), ('bilateral', 'coefficient_bilateral'),
                 ('history_sketch', 'incoming_rows'), ('bilateral', 'incoming_rows'),
                 ('history_sketch', 'native_full_ce'), ('bilateral', 'native_full_ce')]
        reports[order] = {
            'macro_BAcc': {k: float(np.mean([v[t]['BAcc'] for t in tasks])) for k, v in validation.items()},
            'contrasts': {f'{a}_minus_{b}': contrast(validation[a], validation[b], tasks) for a, b in pairs},
            'audits': audits, 'costs': costs,
            'primary_gains_over_stages': {str(stage): {
                c: contrast(runs['history_sketch']['stages'][stage - 1]['validation'],
                            runs[c]['stages'][stage - 1]['validation'], expected_order[:stage])
                for c in ('bilateral', 'random_sketch')} for stage in (4, 8, 12, 16)}}
    keys = list(reports['forward']['contrasts'])
    result = {'sources': sources, 'orders': reports,
              'mean_gain_over_two_orders_pp': {k: float(np.mean([r['contrasts'][k]['gain_pp'] for r in reports.values()])) for k in keys},
              'checks': {'all_native_fixedmeans_exact': True, 'all_checkpoint_and_basis_hashes': True,
                         'saved_basis_orthogonality_and_early_ranks': True, 'coordinate_mean_consistency': True,
                         'paired_actual_data_and_CE_batches': True, 'native_export_probes': True,
                         'real_random_final_history_identical': True},
              'new_scientific_updates': 9720, 'new_scientific_training_examples': 155520,
              'scope': 'Bank0 and two correlated fixed orders on exposed development validation. Primary '
                       'comparisons share native optimizer, replay, rounds and fixedmean. Random historical '
                       'support/rank matches the meaningful real support for its own current model; models '
                       'evolve differently across arms. Coefficient references differ in optimizer coordinates, '
                       'learning rate and auxiliary-probe costs. No equal-compute claim. Norm restoration '
                       'matches each step to its own native proposal, not to proposals from other arms. '
                       'Task-deletion ranges are sensitivity, not confidence intervals. Stage means cover '
                       'only seen tasks. Stored bases/snapshots are audit artifacts; physical memory and '
                       'shared-hardware wall times are not a controlled speed benchmark. No official-test '
                       'data or novelty claim from generic projection or historical sketching.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.4, 1]})
    labels = ['incoming_rows', 'coefficient_bilateral', 'coefficient_history', 'native_full_ce',
              'bilateral', 'history_sketch', 'random_sketch']
    for j, order in enumerate(('forward', 'reverse')):
        axes[0].bar(np.arange(len(labels)) + (j - .5) * .36,
                    [100 * reports[order]['macro_BAcc'][k] for k in labels], width=.36, label=order)
    axes[0].set_xticks(np.arange(len(labels)), ['Incoming\ncoeff.', 'Bilateral\ncoeff.', 'History\ncoeff.',
                                              'Full\nAdam', 'Bilateral\nproj.', 'History\nproj.', 'Random\nproj.'], fontsize=9)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    heat = np.asarray([[reports[o]['contrasts'][f'history_sketch_minus_{c}']['per_task_pp'][t]
                        for o in ('forward', 'reverse') for c in ('bilateral', 'random_sketch')] for t in tasks])
    limit = max(float(np.abs(heat).max()), 1e-8)
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(np.arange(4), ['Fwd\nH−B', 'Fwd\nH−R', 'Rev\nH−B', 'Rev\nH−R'])
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=8)
    axes[1].set_title('Projected history minus bilateral / random')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Streaming native projected merging — fixed replay budget, bank0')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'macro_BAcc': {o: r['macro_BAcc'] for o, r in reports.items()},
                      'mean_gain_over_orders_pp': result['mean_gain_over_two_orders_pp']}, indent=2))


if __name__ == '__main__':
    main()
