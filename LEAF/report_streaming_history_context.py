"""Independently audit the arrival-context source pilot and matched budgets."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_history_context import from_flat, historical_offset, isotropic_offset, flatten


def main():
    root = Path('runs/streaming_history_context_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    results, scores, audits = {}, {}, {}
    stage2_hashes, schedules, context_hashes = {}, {}, {}
    for order in plan['orders']:
        ref = json.loads(Path(plan['reference_runs'][order]).read_text())
        results.setdefault('unadapted_reference', {})[order] = ref['stages'][-1]['macro_BAcc']
        scores.setdefault('unadapted_reference', {})[order] = ref['stages'][-1]['validation']
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
            assert run['order'] == ref['order']
            assert run['coefficient_updates'] == 108 and run['native_updates'] == 288
            assert run['training_examples'] == 6336
            previous, maximum, history = None, 0., []
            schedules[condition, order], context_hashes[condition, order] = [], []
            for stage, record in enumerate(run['stages'], 1):
                task = run['order'][stage - 1]
                directory = folder / f'stage_{stage:02d}'
                p = Path(source['experts'][task])
                assert sha256(p) == record['source_sha256'] == plan['expert_sha256'][str(p)]
                incoming = torch.load(p, map_location='cpu', weights_only=True)
                native = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
                assert sha256(directory / 'merged.ckpt') == record['checkpoint_sha256']
                assert all(torch.equal(native[n], base[n]) for n in fixed)
                assert record['original_history_count'] == stage
                if stage == 1:
                    assert all(torch.equal(native[n], incoming[n]) for n in native)
                else:
                    assert record['updates'] == stage * 12
                    assert record['schedule_hash'] == ref['stages'][stage - 1]['schedule_hash']
                    assert sha256(directory / 'coefficients.pt') == record['coefficients_sha256']
                    cp = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
                    assert cp['snapshots'].shape == (6, 347666)
                    assert torch.equal(cp['snapshots'].double().mean(0).float(), cp['mean'])
                    a = record['adaptation']
                    assert a['updates'] == a['backward_calls'] == 96
                    assert a['training_examples'] == 1536 and a['original_history_count'] == stage - 1
                    assert a['average_count'] == len(a['snapshot_hashes']) == 6
                    assert a['source_context_updates'] == (24 if condition in
                        ('fixed_context', 'history_context', 'isotropic_context') else 0)
                    schedule = np.asarray(a['schedule'], dtype=np.int64)
                    assert schedule.shape == (96, 17) and array_hash(schedule) == a['schedule_hash']
                    allowed = [plan['tasks'].index(t) for t in run['order'][:stage]]
                    assert np.isin(schedule[:, 0], allowed).all()
                    assert (schedule[:, 1:] >= 0).all() and (schedule[:, 1:] < 512).all()
                    if condition != 'native_after':
                        assert (schedule[:, 0] == plan['tasks'].index(task)).all()
                    schedules[condition, order].append(a['schedule_hash'])
                    context_hashes[condition, order].append([(c['historical_hash'], c['isotropic_hash']) for c in a['contexts']])
                    assert len(a['contexts']) == 12
                    for c in a['contexts']:
                        assert c['original_history_tasks'] == run['order'][:stage - 1]
                        assert c['selected_history_task'] in c['original_history_tasks']
                        assert c['tensor_norm_relative_error'] < 3e-6
                    assert sha256(directory / 'adaptation_average.pt') == record['adaptation_average_sha256']
                    av = torch.load(directory / 'adaptation_average.pt', map_location='cpu', weights_only=True)
                    mean = (av['average_sum'] / av['average_count']).float()
                    assert av['average_count'] == 6 and array_hash(mean) == a['mean_hash']
                    adapted = from_flat(mean, base, plan['active'])
                    if condition == 'native_after':
                        assert all(torch.equal(adapted[n], native[n]) for n in native)
                        merged = torch.load(directory / 'before_native.ckpt', map_location='cpu', weights_only=True)
                        operand = incoming
                    else:
                        operand = torch.load(directory / 'adapted_incoming.ckpt', map_location='cpu', weights_only=True)
                        assert all(torch.equal(adapted[n], operand[n]) for n in operand)
                        merged = native
                    maximum = max(maximum, audit_export(base, previous, operand, merged, cp['mean'], plan['active']))
                    assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                        for group in record['export_checks'].values() for c in group.values())
                    if stage == 2:
                        stage2_hashes[condition, order] = record['checkpoint_sha256']
                        assert all(c['historical_norm'] == 0 for c in a['contexts'])
                previous = native
                history.append(incoming)
            for task in plan['pilot_tasks']:
                for k in ('calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size'):
                    assert run['dataset_audits'][task][k] == ref['dataset_audits'][task][k]
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            audits[condition + '_' + order] = dict(independent_export_max_abs_error=maximum,
                elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'])
            print('audited', condition, order, flush=True)
        assert len({stage2_hashes[c, order] for c in ('fixed_context', 'history_context', 'isotropic_context')}) == 1
        assert all(schedules[c, order] == schedules['source_ce', order] for c in
            ('fixed_context', 'history_context', 'isotropic_context'))
        assert all(context_hashes[c, order] == context_hashes['source_ce', order] for c in plan['conditions'])
    contrasts = {}
    pairs = [('history_context', c) for c in ('source_ce', 'fixed_context', 'isotropic_context', 'native_after', 'unadapted_reference')]
    pairs += [(c, 'unadapted_reference') for c in ('source_ce', 'fixed_context', 'isotropic_context', 'native_after')]
    for left, right in pairs:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (scores[left][o][t]['BAcc'] - scores[right][o][t]['BAcc'])
            for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects,
            mean_pp=float(np.mean(list(effects.values()))), task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_four_task_runs=10, coefficient_updates=1080, native_updates=2880,
        source_refinement_native_updates=2304, deployed_native_updates=576,
        training_examples=63360, original_source_pretraining_updates=0,
        official_test_read=False, novelty_established=False, overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(dict(status='passed', results=results, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
