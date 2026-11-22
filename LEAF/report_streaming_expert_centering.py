"""Audit causal expert centering, matched controls and native coefficient exports."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export


def main():
    root = Path('runs/streaming_expert_centering_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    results, scores, audits, stage2_anchors = {}, {}, {}, {}
    for order in plan['orders']:
        reference = json.loads(Path(plan['reference_runs'][order]).read_text())
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['updates'] == 108 and run['training_examples'] == 1728
            assert run['order'] == reference['order'] and len(run['stages']) == 4
            previous, maximum, source_examples, ratios = None, 0., 0, []
            for stage, record in enumerate(run['stages'], 1):
                task = run['order'][stage - 1]
                directory = folder / f'stage_{stage:02d}'
                source_path = Path(source['experts'][task])
                assert sha256(source_path) == plan['expert_sha256'][str(source_path)] == record['source_sha256']
                incoming = torch.load(source_path, map_location='cpu', weights_only=True)
                assert sha256(directory / 'merged.ckpt') == record['checkpoint_sha256']
                native = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
                assert native.keys() == base.keys()
                assert all(torch.equal(native[n], base[n]) for n in fixed)
                if stage == 1:
                    assert all(torch.equal(native[n], incoming[n]) for n in native)
                else:
                    assert record['schedule_hash'] == reference['stages'][stage - 1]['schedule_hash']
                    assert record['updates'] == stage * 12 and record['source_gradient_examples'] == stage * 192
                    assert record['source_backward_calls'] == record['student_backward_calls'] == stage * 12
                    assert record['source_count'] == stage
                    expected_bytes = stage * sum(base[n].numel() * base[n].element_size() for n in plan['active'])
                    assert record['source_active_parameter_bytes'] == expected_bytes
                    assert sha256(directory / 'coefficients.pt') == record['coefficients_sha256']
                    payload = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
                    assert payload['snapshots'].shape == (6, 347666)
                    assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
                    maximum = max(maximum, audit_export(base, previous, incoming, native, payload['mean'], plan['active']))
                    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                        for v in record['export_checks'].values())
                    assert len(record['gradient_records']) == stage * 12
                    for d in record['gradient_records']:
                        a, b, dot = d['source_gradient_norm'], d['student_gradient_norm'], d['source_student_gradient_dot']
                        expected = a * a + b * b - 2 * dot if condition == 'expert_centered' else b * b
                        assert abs(d['used_gradient_norm'] ** 2 - expected) <= 2e-5 * max(a * a + b * b, 1e-20)
                        ratios.append(a / max(b, 1e-30))
                    source_examples += record['source_gradient_examples']
                    if stage == 2:
                        stage2_anchors[f'{condition}_{order}'] = [d['source_gradient_hash'] for d in record['gradient_records']]
                if condition == 'balanced_ce':
                    # Reference checkpoint bytes may have been relocated; the frozen digest is sufficient.
                    assert record['checkpoint_sha256'] == reference['stages'][stage - 1]['checkpoint_sha256']
                    if 'validation' in record:
                        assert record['validation'] == reference['stages'][stage - 1]['validation']
                previous = native
            assert source_examples == 1728
            for task in plan['pilot_tasks']:
                for key in ['calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size']:
                    assert run['dataset_audits'][task][key] == reference['dataset_audits'][task][key]
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            audits[f'{condition}_{order}'] = dict(independent_export_max_abs_error=maximum,
                elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'],
                source_gradient_examples=source_examples, median_source_to_student_gradient_norm_ratio=float(np.median(ratios)))
            print('audited', condition, order, flush=True)
        assert stage2_anchors[f'balanced_ce_{order}'] == stage2_anchors[f'expert_centered_{order}'] == stage2_anchors[f'output_kl_{order}']
    contrasts = {}
    for left, right in [('expert_centered', 'balanced_ce'), ('expert_centered', 'output_kl'), ('output_kl', 'balanced_ce')]:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (scores[left][o][t]['BAcc'] - scores[right][o][t]['BAcc'])
            for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects,
            mean_pp=float(np.mean(list(effects.values()))), task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_four_task_runs=6, coefficient_updates=648, student_training_examples=10368,
        student_CE_examples=6912, student_KL_examples=3456, expert_CE_gradient_examples=10368,
        backward_calls=1296, source_training_updates=0, official_test_read=False,
        novelty_established=False, overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(dict(status='passed', results=results, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
