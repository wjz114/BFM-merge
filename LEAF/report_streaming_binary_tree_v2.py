"""Audit every binary merge and compare against both sequential controls."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_binary_tree import events, live_intervals, interval_key


def audit_run(folder, plan, source, base, reference):
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and len(run['stages']) == 16
    assert run['plan_sha256'] == sha256(folder.parent / 'plan.json')
    assert run['updates'] == 2028 and run['training_examples'] == 32448
    assert not run['official_test_read'] and run['inference_checkpoint_count'] == 1
    assert run['order'] == reference['order']
    fixed = set(base) - set(plan['active'])
    tree = run['condition'] == 'binary_tree'
    cache, previous, maximum, stage_schedules = {}, None, 0., []
    for stage, record in enumerate(run['stages'], 1):
        task = run['order'][stage - 1]
        directory = folder / f'stage_{stage:02d}'
        source_path = Path(source['experts'][task])
        assert sha256(source_path) == plan['expert_sha256'][task] == record['source_sha256']
        incoming = torch.load(source_path, map_location='cpu', weights_only=True)
        if tree:
            cache[(stage, stage)] = incoming
        assert record['stage'] == stage and record['arriving_task'] == task
        assert sha256(directory / 'merged.ckpt') == record['checkpoint_sha256']
        assert (directory / 'merged.ckpt').resolve() == Path(record['checkpoint_storage_path'])
        final = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
        if stage == 1:
            assert not record['events'] and record['updates'] == 0
            assert all(torch.equal(final[k], incoming[k]) for k in final)
        initial, schedules, event_updates = None, [], 0
        assert len(record['events']) == len(events(stage))
        for j, ((a, b), event) in enumerate(zip(events(stage), record['events'])):
            assert event['left_interval'] == list(a) and event['right_interval'] == list(b)
            tasks = run['order'][a[0] - 1:b[1]]
            assert event['tasks'] == tasks
            eventdir = directory / f'event_{j:02d}'
            assert sha256(eventdir / 'coefficients.pt') == event['coefficients_sha256']
            assert sha256(eventdir / 'merged.ckpt') == event['checkpoint_sha256']
            native = torch.load(eventdir / 'merged.ckpt', map_location='cpu', weights_only=True)
            payload = torch.load(eventdir / 'coefficients.pt', map_location='cpu', weights_only=True)
            assert payload['snapshots'].shape == (6, 347666)
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            assert all(torch.isfinite(v).all() for v in payload.values())
            if tree:
                left, right = cache[a], cache[b]
                na, nb = a[1] - a[0] + 1, b[1] - b[0] + 1
                assert event['basis_intervals'] == [list(a), list(b)]
            else:
                left, right, na, nb = previous, incoming, stage - 1, 1
                assert event['basis_intervals'] == [[1, stage - 1], [stage, stage]]
            assert event['left_count'] == na and event['right_count'] == nb
            if tree or initial is None:
                expected_initial = torch.zeros_like(payload['initial'])
                n = len(plan['active'])
                expected_initial[:n] = na / (na + nb)
                expected_initial[n:2 * n] = nb / (na + nb)
            else:
                expected_initial = initial
            assert torch.equal(expected_initial, payload['initial'])
            assert native.keys() == base.keys()
            assert all(torch.equal(native[k], base[k]) for k in fixed)
            assert all(torch.isfinite(native[k]).all() for k in plan['active'])
            maximum = max(maximum, audit_export(base, left, right, native, payload['mean'], plan['active']))
            permutations = {t: np.random.default_rng([plan['seed'], stage,
                plan['tasks'].index(t), 217]).permutation(512) for t in tasks}
            rng, schedule = np.random.default_rng([plan['seed'], stage, 931]), []
            for rnd in range(12):
                for k in rng.permutation(len(tasks)):
                    t = tasks[int(k)]
                    schedule.append([plan['tasks'].index(t), *permutations[t][rnd * 16:(rnd + 1) * 16].tolist()])
            digest = array_hash(np.asarray(schedule, dtype=np.int64))
            assert digest == event['schedule_hash']
            assert event['updates'] == len(tasks) * 12 and event['training_examples'] == len(tasks) * 192
            assert set(event['export_checks']) == set(tasks)
            assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                       for v in event['export_checks'].values())
            schedules.append(digest)
            event_updates += event['updates']
            if tree:
                cache[(a[0], b[1])] = native
            else:
                initial = payload['mean']
            del left, right
        if stage > 1:
            assert all(torch.equal(final[k], native[k]) for k in final)
            assert schedules[-1] == reference['stages'][stage - 1]['schedule_hash']
        assert record['updates'] == event_updates and record['training_examples'] == event_updates * 16
        stage_schedules.append(schedules)
        if tree:
            cache = {k: cache[k] for k in live_intervals(stage)}
            assert record['persistent_intervals'] == [interval_key(k) for k in cache]
        expected_count = 2 * stage.bit_count() - 1 if tree else 1
        assert record['persistent_native_checkpoint_count'] == expected_count
        assert record['persistent_native_bytes'] == expected_count * sum(v.numel() * v.element_size() for v in final.values())
        if stage <= 3:
            # Historical prefix payloads were not retained. The newly exported
            # bytes exactly match the SHA256 already frozen in the reference run.
            assert record['checkpoint_sha256'] == reference['stages'][stage - 1]['checkpoint_sha256']
        previous = final
    for task in plan['tasks']:
        for key in ['calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size']:
            assert run['dataset_audits'][task][key] == reference['dataset_audits'][task][key]
    return run, dict(independent_native_export_max_abs_error=maximum, schedules=stage_schedules,
        elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'],
        maximum_persistent_native_checkpoint_count=max(r['persistent_native_checkpoint_count'] for r in run['stages']),
        maximum_persistent_native_bytes=max(r['persistent_native_bytes'] for r in run['stages']))


def main():
    root = Path('runs/streaming_binary_tree_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    runs, results, audits = {}, {}, {}
    for order in plan['orders']:
        reference = json.loads(Path(plan['reference_runs'][order]).read_text())
        runs.setdefault('ordinary_chain', {})[order] = reference
        results.setdefault('ordinary_chain', {})[order] = reference['stages'][-1]['macro_BAcc']
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run, audit = audit_run(folder, plan, source, base, reference)
            runs.setdefault(condition, {})[order] = run
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            audits[f'{condition}_{order}'] = audit
            print('audited', condition, order, flush=True)
        assert audits[f'binary_tree_{order}']['schedules'] == audits[f'matched_chain_{order}']['schedules']
    contrasts = {}
    for left, right in [('binary_tree', 'matched_chain'), ('binary_tree', 'ordinary_chain'),
                        ('matched_chain', 'ordinary_chain')]:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (runs[left][o]['stages'][-1]['validation'][t]['BAcc'] -
            runs[right][o]['stages'][-1]['validation'][t]['BAcc']) for o in plan['orders']])) for t in plan['tasks']}
        trajectory = {str(s): {o: 100 * (runs[left][o]['stages'][s - 1]['macro_BAcc'] -
            runs[right][o]['stages'][s - 1]['macro_BAcc']) for o in plan['orders']} for s in plan['validation_stages']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects, mean_pp=float(np.mean(list(effects.values()))),
            task_mean_effects_pp=task_effects, stage_effects_pp=trajectory,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    for audit in audits.values():
        del audit['schedules']
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_full16_runs=4, coefficient_updates=8112, CE_examples=129792,
        additional_source_training=0, official_test_read=False, novelty_established=False,
        overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(dict(status='passed', results=results, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
