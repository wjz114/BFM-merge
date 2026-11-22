"""Independent native exports and fixed-protocol source-context comparisons."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def export_error(base, old, new, native, c, residual, active):
    n = len(active)
    rows_total = sum(base[k].shape[0] if base[k].ndim else 1 for k in active)
    offset, maximum = 2 * n, 0.
    for j, name in enumerate(active):
        rows = base[name].shape[0] if base[name].ndim else 1
        shape = (-1,) + (1,) * (base[name].ndim - 1) if base[name].ndim else ()
        a, b = old[name] - base[name], new[name] - base[name]
        expected = base[name] + c[j] * a + c[n + j] * b
        expected += c[offset:offset + rows].reshape(shape) * b
        expected += c[offset + rows_total:offset + rows_total + rows].reshape(shape) * a
        expected += residual[name]
        maximum = max(maximum, float((expected - native[name]).abs().max()))
        offset += rows
    assert maximum < 2e-6 and offset + rows_total == c.numel()
    return maximum


def main():
    root = Path('runs/source_behavior_preserving_repair_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    results, task_scores, audits = {}, {}, {}
    for order in plan['orders']:
        ref = json.loads(Path(plan['reference_runs'][order]).read_text())
        results.setdefault('bilateral', {})[order] = ref['stages'][-1]['macro_BAcc']
        task_scores.setdefault('bilateral', {})[order] = ref['stages'][-1]['validation']
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['updates'] == 108 and run['training_examples'] == 1728
            assert run['order'] == ref['order'] and len(run['stages']) == 4
            previous, maximum, source_forwards = None, 0., 0
            for index, stage in enumerate(run['stages'], 1):
                task = run['order'][index - 1]
                directory = folder / f'stage_{index:02d}'
                assert sha256(directory / 'merged.ckpt') == stage['checkpoint_sha256']
                native = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
                path = Path(source['experts'][task])
                assert sha256(path) == plan['expert_sha256'][str(path)]
                incoming = torch.load(path, map_location='cpu', weights_only=True)
                assert native.keys() == base.keys()
                assert all(torch.equal(native[n], base[n]) for n in fixed)
                if index == 1:
                    assert all(torch.equal(native[n], incoming[n]) for n in native)
                else:
                    assert stage['schedule_hash'] == ref['stages'][index - 1]['schedule_hash']
                    assert stage['updates'] == index * 12
                    path = directory / 'coefficients.pt'
                    assert sha256(path) == stage['coefficients_sha256']
                    payload = torch.load(path, map_location='cpu', weights_only=True)
                    assert payload['snapshots'].shape == (6, 347666)
                    assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
                    assert len(payload['native_snapshots']) == 6
                    for name in plan['active']:
                        expected = torch.stack([v[name] for v in payload['native_snapshots']]).double().mean(0).float()
                        assert torch.equal(expected, payload['native_mean'][name])
                    maximum = max(maximum, export_error(base, previous, incoming, native,
                        payload['mean'], payload['native_mean'], plan['active']))
                    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                        for v in stage['export_checks'].values())
                    if condition == 'source_context_kl':
                        assert set(stage['source_initial_logit_errors']) == set(run['order'][:index])
                        assert max(stage['source_initial_logit_errors'].values()) < 1e-5
                    source_forwards += stage['auxiliary_student_forward_examples']
                    del payload
                previous = native
            assert source_forwards == (1728 if condition == 'source_context_kl' else 0)
            for task in plan['pilot_tasks']:
                for key in ['calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size']:
                    assert run['dataset_audits'][task][key] == ref['dataset_audits'][task][key]
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            task_scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            audits[f'{condition}_{order}'] = dict(independent_export_max_abs_error=maximum,
                source_context_extra_student_forwards=source_forwards,
                elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'])
    contrasts = {}
    for left, right in [('source_context_kl', 'plain_ce'), ('source_context_kl', 'merged_context_kl'),
                        ('source_context_kl', 'bilateral'), ('merged_context_kl', 'plain_ce'),
                        ('plain_ce', 'bilateral')]:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (task_scores[left][o][t]['BAcc'] -
            task_scores[right][o][t]['BAcc']) for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects,
            mean_pp=float(np.mean(list(effects.values()))), task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_runs=6, hybrid_optimizer_calls=648, CE_examples=10368,
        auxiliary_source_context_student_forward_examples=3456,
        actual_new_teacher_forward_examples=0, required_teacher_cache_examples_per_run=2048,
        official_test_read=False, novelty_established=False, overall_goal_achieved=False,
        scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'results', 'contrasts']}, indent=2))


if __name__ == '__main__':
    main()
