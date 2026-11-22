"""Audit projected attention operators against dense CPU rank32 reconstruction."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from operator_residual_audit import audit


def main():
    root = Path('runs/streaming_operator_residual_v1')
    plan = json.loads((root / 'plan.json').read_text())
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    results, scores, audits = {}, {}, {}
    for order in plan['orders']:
        reference = json.loads(Path(plan['reference_runs'][order]).read_text())
        more_training = json.loads(Path(plan['native_after_reference_runs'][order]).read_text())
        results.setdefault('native_after_more_updates', {})[order] = more_training['stages'][-1]['macro_BAcc']
        scores.setdefault('native_after_more_updates', {})[order] = more_training['stages'][-1]['validation']
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
            assert run['updates'] == 108 and run['training_examples'] == 1728
            assert run['order'] == reference['order'] and len(run['stages']) == 4
            previous, stage_audits = None, {}
            for stage, record in enumerate(run['stages'], 1):
                directory = folder / f'stage_{stage:02d}'
                p = Path(source['experts'][run['order'][stage - 1]])
                assert sha256(p) == record['source_sha256'] == plan['expert_sha256'][str(p)]
                incoming = torch.load(p, map_location='cpu', weights_only=True)
                assert sha256(directory / 'merged.ckpt') == record['checkpoint_sha256']
                native = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
                assert native.keys() == base.keys()
                assert all(torch.equal(native[n], base[n]) for n in fixed)
                if stage == 1:
                    assert all(torch.equal(native[n], incoming[n]) for n in native)
                else:
                    assert record['schedule_hash'] == reference['stages'][stage - 1]['schedule_hash']
                    assert record['updates'] == stage * 12 and record['training_examples'] == stage * 192
                    assert record['coefficient_count'] == 348178 and record['operator_gate_count'] == 512
                    assert sha256(directory / 'coefficients.pt') == record['coefficients_sha256']
                    cp = torch.load(directory / 'coefficients.pt', map_location='cpu', weights_only=True)
                    assert cp['snapshots'].shape == (6, 348178)
                    assert torch.equal(cp['snapshots'].double().mean(0).float(), cp['mean'])
                    assert cp['residual_left'].shape == (512, 32, 257)
                    assert cp['residual_right'].shape == (512, 32, 256)
                    stage_audits[str(stage)] = audit(base, previous, incoming, native, cp,
                        plan['active'], stage - 1, condition)
                    assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                        for c in record['export_checks'].values())
                if condition == 'operator_zero':
                    assert record['checkpoint_sha256'] == reference['stages'][stage - 1]['checkpoint_sha256']
                    if 'validation' in record:
                        assert record['validation'] == reference['stages'][stage - 1]['validation']
                previous = native
            for task in plan['pilot_tasks']:
                for key in ['calibration_eeg_hash', 'calibration_labels_hash', 'class_weights', 'validation_size']:
                    assert run['dataset_audits'][task][key] == reference['dataset_audits'][task][key]
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            audits[condition + '_' + order] = dict(stages=stage_audits,
                elapsed_seconds=run['elapsed_seconds'], peak_reserved_bytes=run['peak_reserved_bytes'],
                gate_norm=run['stages'][-1].get('gate_norm'),
                native_after_reference_elapsed_seconds=more_training['elapsed_seconds'])
            print('audited', condition, order, flush=True)
    contrasts = {}
    for left, right in [('operator_covariance', 'operator_zero'), ('operator_covariance', 'operator_random'),
                        ('operator_covariance', 'native_after_more_updates'), ('operator_random', 'operator_zero')]:
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (scores[left][o][t]['BAcc'] - scores[right][o][t]['BAcc'])
            for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[left + '_minus_' + right] = dict(paired_effects_pp=effects,
            mean_pp=float(np.mean(list(effects.values()))), task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    result = dict(status='passed', results=results, contrasts=contrasts, audits=audits,
        scientific_four_task_runs=6, coefficient_updates=648, training_examples=10368,
        native_updates=0, source_training_updates=0, official_test_read=False,
        novelty_established=False, overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(dict(status='passed', results=results, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
