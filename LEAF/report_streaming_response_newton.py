"""Audit response-curvature fits against frozen same-objective Adam controls."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export


def main():
    root = Path('runs/streaming_response_newton_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    results, scores, audits = {}, {}, {}
    backward_calls, extra_forwards, accepted_total = 0, 0, 0
    for order in plan['orders']:
        for condition in ('adam_ce', 'adam_kl'):
            p = Path(plan['adam_reference_runs'][condition][order])
            run = json.loads(p.read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
        reference = json.loads(Path(plan['adam_reference_runs']['adam_ce'][order]).read_text())
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['updates'] == 108 and run['training_examples'] == 1728
            assert run['order'] == reference['order'] and len(run['stages']) == 4
            previous, maximum, gradients = None, 0., []
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
                assert all(torch.isfinite(native[n]).all() for n in plan['active'])
                if stage == 1:
                    assert all(torch.equal(native[n], incoming[n]) for n in native)
                else:
                    assert record['schedule_hash'] == reference['stages'][stage - 1]['schedule_hash']
                    assert record['updates'] == stage * 12
                    assert record['source_forward_examples'] == record['training_examples'] == stage * 192
                    assert record['source_gradient_examples'] == stage * 192
                    assert record['source_backward_calls'] == stage * 12
                    assert record['source_count'] == stage
                    assert sha256(directory / 'coefficients.pt') == record['coefficients_sha256']
                    payload = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
                    assert payload['snapshots'].shape == (6, 347666)
                    assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
                    maximum = max(maximum, audit_export(base, previous, incoming, native, payload['mean'], plan['active']))
                    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                        for v in record['export_checks'].values())
                    assert len(record['gradient_records']) == stage * 12
                    for d in record['gradient_records']:
                        assert d['task'] in run['order'][:stage]
                        classes = len(run['dataset_audits'][d['task']]['class_weights'])
                        assert d['jacobian_rows'] == 16 * (classes - 1)
                        assert d['solve_relative_residual'] < 1e-7
                        assert d['gradient_dot_step'] <= 1e-12
                        trials = d['trial_losses']
                        assert 1 <= len(trials) <= plan['line_search_trials']
                        assert d['line_search_forward_examples'] == len(trials) * 16
                        assert d['realized_coefficient_step_norm'] <= d['radius'] + 1e-5
                        if d['accepted']:
                            factor = 0.5 ** (len(trials) - 1)
                            assert d['line_search_scale'] == factor
                            bound = d['objective'] + plan['armijo'] * factor * d['direction_clip'] * d['gradient_dot_step']
                            assert trials[-1] <= bound
                        else:
                            assert d['realized_coefficient_step_norm'] == 0
                            assert len(trials) == plan['line_search_trials']
                        gradients.append(d)
                previous = native
            for task in plan['pilot_tasks']:
                for key in ['calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size']:
                    assert run['dataset_audits'][task][key] == reference['dataset_audits'][task][key]
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            back = sum(d['jacobian_rows'] for d in gradients)
            forwards = sum(d['line_search_forward_examples'] for d in gradients)
            accepted = sum(d['accepted'] for d in gradients)
            assert back == sum(r.get('student_backward_calls', 0) for r in run['stages'])
            assert forwards == sum(r.get('line_search_forward_examples', 0) for r in run['stages'])
            assert accepted == sum(r.get('accepted_updates', 0) for r in run['stages'])
            backward_calls += back;extra_forwards += forwards;accepted_total += accepted
            audits[f'{condition}_{order}'] = dict(independent_export_max_abs_error=maximum,
                elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'],
                student_backward_calls=back, line_search_forward_examples=forwards,
                accepted_updates=accepted, clipped_proposals=sum(d['direction_clip'] < 1 for d in gradients),
                median_solve_relative_residual=float(np.median([d['solve_relative_residual'] for d in gradients])),
                mean_batch_objective_before=float(np.mean([d['objective'] for d in gradients])),
                mean_batch_objective_after=float(np.mean([d['trial_losses'][-1] if d['accepted'] else d['objective'] for d in gradients])))
            print('audited', condition, order, flush=True)
    contrasts = {}
    for left, right in [('ggn_ce', 'adam_ce'), ('ggn_kl', 'adam_kl'), ('ggn_kl', 'adam_ce'), ('ggn_kl', 'ggn_ce')]:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (scores[left][o][t]['BAcc'] - scores[right][o][t]['BAcc'])
            for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects,
            mean_pp=float(np.mean(list(effects.values()))), task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_four_task_runs=4, coefficient_update_opportunities=432,
        accepted_coefficient_updates=accepted_total, student_training_examples=6912,
        expert_forward_examples=6912, expert_gradient_examples=6912,
        expert_backward_calls=432, student_backward_calls=backward_calls,
        line_search_forward_examples=extra_forwards, source_training_updates=0, native_updates=0,
        official_test_read=False, novelty_established=False, overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2);out.write('\n')
    print(json.dumps(dict(status='passed', results=results, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
