"""Audit the four-task crossfit pilot without treating it as a full benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_crossfit_training import CONDITIONS


def expected_schedule(plan, tasks, count, assignments, condition):
    seen = tasks[:count]
    permutations = {t: np.random.default_rng([plan['seed'], count, plan['tasks'].index(t), 217])
                    .permutation(512) for t in seen}
    rng = np.random.default_rng([plan['seed'], count, 931])
    schedule, routing, batches = [], [], 0
    for rnd in range(plan['rounds']):
        for k in rng.permutation(count):
            task = seen[int(k)]
            ids = permutations[task][rnd * plan['batch_size']:(rnd + 1) * plan['batch_size']]
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            worlds = assignments[task][ids].copy()
            if condition == 'out_of_fold':
                worlds = 1 - worlds
            elif condition in ('source_soup', 'single_full'):
                worlds[:] = 0
            for world in (0, 1):
                selected = [int(i) for i, w in zip(ids, worlds) if w == world]
                batches += bool(selected)
                routing.extend([[plan['tasks'].index(task), i, world] for i in selected])
    return array_hash(np.asarray(schedule, dtype=np.int64)), array_hash(np.asarray(routing, dtype=np.int64)), batches


def audit(folder, plan, plan_path, source, base, assignments):
    run = json.loads((folder / 'run.json').read_text())
    condition, tasks = run['condition'], run['order']
    assert run['status'] == 'completed' and len(run['stages']) == 4
    assert run['plan_sha256'] == sha256(plan_path) and condition in CONDITIONS
    assert tasks in (plan['pilot_tasks'], list(reversed(plan['pilot_tasks'])))
    assert run['updates'] == 108 and run['training_examples'] == 1728
    assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
    assert not run['official_test_read'] and run['strict_deterministic_algorithms']
    assert run['cudnn_deterministic'] and run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    proxy_mode = condition in ('out_of_fold', 'in_fold')
    assert run['persistent_training_proxy_count'] == (2 if proxy_mode else 0)
    assert run['source_experts_used_per_task'] == (1 if condition == 'single_full' else 3)
    fixed = set(base) - set(plan['active'])
    previous, old_proxies, errors = None, [], []
    for count, stage in enumerate(run['stages'], 1):
        task = tasks[count - 1]
        assert stage['stage'] == count and stage['arriving_task'] == task
        paths = [source['experts'][task]]
        if condition != 'single_full':
            paths += plan['half_experts'][task]
        assert paths == stage['source_paths']
        assert stage['source_sha256'] == [plan['expert_sha256'][p] for p in paths]
        models = []
        for p in paths:
            assert sha256(Path(p)) == plan['expert_sha256'][p]
            models.append(torch.load(p, map_location='cpu', weights_only=True))
        incoming = models[0]
        if condition == 'source_soup':
            incoming = {n: ((models[0][n].double() + models[1][n].double()
                             + models[2][n].double()) / 3).float() if n in plan['active']
                        else v.clone() for n, v in models[0].items()}
        directory = folder / f'stage_{count:02d}'
        path = directory / 'merged.ckpt'
        assert path.is_symlink() and sha256(path) == stage['checkpoint_sha256']
        assert str(path.resolve()) == stage['checkpoint_storage_path']
        native = torch.load(path, map_location='cpu', weights_only=True)
        proxies = []
        assert len(stage['proxies']) == (2 if proxy_mode else 0)
        for k, record in enumerate(stage['proxies']):
            path = directory / f'proxy{k}.ckpt'
            assert record['world'] == k and str(path.resolve()) == record['path']
            assert path.is_symlink() and sha256(path) == record['sha256']
            proxies.append(torch.load(path, map_location='cpu', weights_only=True))
        for state in [native, *proxies, *models]:
            assert state.keys() == base.keys()
            assert all(torch.equal(state[n], base[n]) for n in fixed)
            assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        if count == 1:
            assert stage['updates'] == stage['training_examples'] == 0
            assert all(torch.equal(native[n], incoming[n]) for n in base)
            for k, state in enumerate(proxies):
                assert all(torch.equal(state[n], models[k + 1][n]) for n in base)
        else:
            path = directory / 'coefficients.pt'
            assert path.is_symlink() and sha256(path) == stage['coefficients_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['snapshots'].shape == (6, 347666)
            assert torch.equal(payload['mean'], payload['snapshots'].double().mean(0).float())
            errors.append(audit_export(base, previous, incoming, native, payload['mean'], plan['active']))
            for k, state in enumerate(proxies):
                errors.append(audit_export(base, old_proxies[k], models[k + 1], state,
                                           payload['mean'], plan['active']))
            schedule, routing, batches = expected_schedule(plan, tasks, count, assignments, condition)
            assert stage['schedule_hash'] == schedule and stage['routing_hash'] == routing
            assert stage['updates'] == count * 12 and stage['training_examples'] == count * 192
            assert stage['training_forward_batches'] == stage['backward_calls'] == batches
            assert len(stage['export_checks']) == (3 if proxy_mode else 1)
            assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                       for checks in stage['export_checks'].values() for c in checks.values())
        if count in plan['validation_stages']:
            assert set(stage['validation']) == set(tasks[:count])
            mean = float(np.mean([m['BAcc'] for m in stage['validation'].values()]))
            assert stage['macro_BAcc'] == mean
        previous, old_proxies = native, proxies
    return run, max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    path = args.root / 'plan.json'
    plan = json.loads(path.read_text())
    assert not (args.root / 'comparison.json').exists()
    for p, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == digest, p
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    with np.load(plan['folds']) as folds:
        assignments = {t: folds[f'{t}__calibration_fold'].copy() for t in plan['pilot_tasks']}
    runs, errors, results = {}, {}, {}
    for condition in CONDITIONS:
        results[condition] = {}
        for order in ('forward', 'reverse'):
            name = f'{condition}_{order}'
            run, error = audit(args.root / name, plan, path, source, base, assignments)
            runs[name], errors[name] = run, error
            results[condition][order] = run['stages'][-1]['macro_BAcc']
        results[condition]['mean'] = float(np.mean(list(results[condition].values())))
    audits = [r['dataset_audits'] for r in runs.values()]
    assert all(a == audits[0] for a in audits)
    contrasts = {}
    for left, right in [('out_of_fold', 'in_fold'), ('out_of_fold', 'source_soup'),
                         ('out_of_fold', 'single_full'), ('in_fold', 'single_full'),
                         ('source_soup', 'single_full')]:
        differences = {o: 100 * (results[left][o] - results[right][o]) for o in ('forward', 'reverse')}
        differences['mean_pp'] = float(np.mean(list(differences.values())))
        differences['task_deltas_pp'] = {o: {t: 100 * (runs[f'{left}_{o}']['stages'][-1]['validation'][t]['BAcc']
            - runs[f'{right}_{o}']['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']}
            for o in ('forward', 'reverse')}
        contrasts[f'{left}_minus_{right}'] = differences
    result = {'plan_sha256': sha256(path), 'results': results, 'contrasts': contrasts,
        'audit': {'all_eight_runs_passed': True, 'matched_calibration': True,
            'independent_schedule_and_fold_routing': True, 'fixed_late_average': True,
            'all_native_states_independently_reconstructed': True, 'max_errors': errors},
        'scope': 'Four prespecified development tasks, two arrival orders, one calibration/fold/source seed. Not full16, independent-subject or independent-seed evidence. No novelty or superiority claim from this pilot alone.',
        'source_budget': plan['source_budget'], 'updates': 864, 'training_CE_examples': 13824,
        'official_test_read': False, 'overall_goal_achieved': False}
    with (args.root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({'results': results, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
