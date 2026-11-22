"""Audit the calibration provenance comparison on one common development remainder."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export


ROOT = Path('runs/calibration_domain_pilot_v1')


def independent_schedule(plan, tasks, stage, pools):
    samples, labels, rows, label_rows = {}, {}, [], []
    for task in tasks:
        labels[task] = pools[f'{task}__paired_calibration_labels']
        samples[task] = np.random.default_rng([plan['seed'], stage, plan['tasks'].index(task), 217]).integers(
            0, len(labels[task]), size=plan['rounds'] * plan['batch_size'])
    order = np.random.default_rng([plan['seed'], stage, 931])
    for rnd in range(plan['rounds']):
        for k in order.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = samples[task][rnd * plan['batch_size']:(rnd + 1) * plan['batch_size']]
            rows.append([plan['tasks'].index(task), *ids.tolist()])
            label_rows.append([plan['tasks'].index(task), *labels[task][ids].tolist()])
    return np.asarray(rows, dtype=np.int64), np.asarray(label_rows, dtype=np.int64)


def audit_run(plan, condition, domain, order, source, base, golden, pools):
    name = f'{condition}_{domain}_{order}'
    directory = ROOT / name
    run = json.loads((directory / 'run.json').read_text())
    tasks = plan['pilot_tasks'] if order == 'forward' else list(reversed(plan['pilot_tasks']))
    reference = domain == 'original512_reference'
    assert run['status'] == 'completed' and run['plan_sha256'] == sha256(ROOT / 'plan.json')
    assert run['order'] == tasks and len(run['stages']) == len(tasks)
    assert run['condition'] == condition and run['domain'] == domain
    assert run['inference_checkpoint_count'] == 1 and not run['inference_routing']
    assert not run['official_test_read'] and run['calibration_EEG_read'] == (not reference)
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert run['updates'] == (0 if reference else 108)
    assert run['training_examples'] == (0 if reference else 1728)
    assert run['source_experts_used_per_task'] == (1 if condition == 'single_full' else 3)
    previous, errors, schedules = None, [], []
    fixed = set(base) - set(plan['active'])
    if reference:
        prior_path = Path(plan['reference_runs'][f'{condition}_{order}'])
        assert run['reference_run_sha256'] == sha256(prior_path)
        prior = json.loads(prior_path.read_text())
    for stage, task in enumerate(tasks, 1):
        record = run['stages'][stage - 1]
        assert record['stage'] == stage and record['arriving_task'] == task
        checkpoint = Path(record['checkpoint_storage_path'])
        assert sha256(checkpoint) == record['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state.keys() == base.keys()
        assert all(torch.equal(state[k], base[k]) for k in fixed)
        assert all(torch.isfinite(state[k]).all() for k in plan['active'])
        if reference:
            assert record['checkpoint_sha256'] == prior['stages'][stage - 1]['checkpoint_sha256']
            assert record['updates'] == record['training_examples'] == 0
        else:
            paths = [source['experts'][task]]
            if condition == 'source_soup':
                paths += plan['half_experts'][task]
            assert record['source_paths'] == paths
            assert record['source_sha256'] == [plan['expert_sha256'][p] for p in paths]
            models = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
            incoming = {k: sum(m[k].double() for m in models).div(len(models)).float()
                        if k in plan['active'] else base[k].clone() for k in base}
            assert all(torch.equal(m[k], base[k]) for m in models for k in fixed)
            if stage == 1:
                assert record['updates'] == record['training_examples'] == 0
                assert all(torch.equal(state[k], incoming[k]) for k in base)
            else:
                assert record['updates'] == stage * plan['rounds']
                assert record['training_examples'] == stage * plan['rounds'] * plan['batch_size']
                path = directory / f'stage_{stage:02d}/coefficients.pt'
                assert sha256(path) == record['coefficients_sha256']
                payload = torch.load(path, map_location='cpu', weights_only=True)
                assert payload['snapshots'].shape == (6, 347666)
                assert torch.isfinite(payload['snapshots']).all()
                assert torch.equal(payload['mean'], payload['snapshots'].double().mean(0).float())
                expected, label_expected = independent_schedule(plan, tasks[:stage], stage, pools)
                assert np.array_equal(payload['schedule'].numpy(), expected)
                assert np.array_equal(payload['label_schedule'].numpy(), label_expected)
                assert record['schedule_hash'] == array_hash(expected)
                assert record['label_schedule_hash'] == array_hash(label_expected)
                schedules.append((record['schedule_hash'], record['label_schedule_hash']))
                errors.append(audit_export(base, previous, incoming, state, payload['mean'], plan['active']))
                assert record['sampling'] == 'with_replacement'
                assert set(record['export_checks']) == set(tasks[:stage])
                assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                           for c in record['export_checks'].values())
                del payload
            previous = state
            del incoming, models
        audit = run['dataset_audits'][task]
        expected_data = golden[task]['development_evaluation']
        assert audit['validation_eeg_hash'] == expected_data['eeg_hash']
        assert audit['validation_labels_hash'] == expected_data['labels_hash']
        assert audit['validation_size'] == expected_data['rows']
        assert audit['calibration_EEG_read'] == (not reference)
        if not reference:
            expected_data = golden[task][domain]
            assert audit['calibration_size'] == expected_data['rows']
            assert audit['calibration_eeg_hash'] == expected_data['eeg_hash']
            assert audit['calibration_labels_hash'] == expected_data['labels_hash']
            assert audit['class_weights'] == [1.] * len(expected_data['class_counts'])
        if stage in plan['validation_stages']:
            assert set(record['validation']) == set(tasks[:stage])
            score = np.mean([v['BAcc'] for v in record['validation'].values()])
            assert abs(record['macro_BAcc'] - score) < 1e-12
        else:
            assert 'validation' not in record
    assert sum(v['validation_size'] for v in run['dataset_audits'].values()) == 4398
    return run, dict(independent_export_max_abs_error=max(errors, default=0.),
                     schedules=schedules, run_sha256=sha256(directory / 'run.json'))


def main():
    torch.set_num_threads(4)
    assert not (ROOT / 'comparison.json').exists()
    plan = json.loads((ROOT / 'plan.json').read_text())
    for p,h in {**plan['source_sha256'], **plan['input_sha256'], **plan['expert_sha256']}.items():
        assert sha256(Path(p)) == h, p
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    golden = json.loads(Path(plan['data_audit']).read_text())['tasks']
    pools = np.load(plan['pools'])
    runs, audits, contrasts, scores = {}, {}, {}, {}
    for order in plan['orders']:
        for condition in plan['conditions']:
            for domain in [*plan['domains'], 'original512_reference']:
                name = f'{condition}_{domain}_{order}'
                runs[name], audits[name] = audit_run(plan, condition, domain, order, source, base, golden, pools)
                scores[name] = runs[name]['stages'][-1]['macro_BAcc']
                print('audited', name, scores[name], flush=True)
        paired = [audits[f'{c}_{d}_{order}']['schedules'] for c in plan['conditions'] for d in plan['domains']]
        assert all(s == paired[0] for s in paired)
        for condition in plan['conditions']:
            names = [f'{condition}_{d}_{order}' for d in [*plan['domains'], 'original512_reference']]
            first = [runs[n]['stages'][0]['validation'] for n in names]
            assert first[0] == first[1] == first[2]
            for a,b in [('development_calibration','training_calibration'),
                        ('development_calibration','original512_reference'),
                        ('training_calibration','original512_reference')]:
                ra, rb = [runs[f'{condition}_{d}_{order}']['stages'][-1] for d in [a,b]]
                contrasts[f'{condition}_{a}_minus_{b}_{order}'] = dict(
                    macro_BAcc_pp=100*(ra['macro_BAcc']-rb['macro_BAcc']),
                    task_BAcc_pp={t:100*(ra['validation'][t]['BAcc']-rb['validation'][t]['BAcc'])
                                  for t in plan['pilot_tasks']})
    result = dict(status='passed', plan_sha256=sha256(ROOT / 'plan.json'),
        checks=dict(sources_and_data_frozen=True, independent_schedules=True,
            paired_index_and_label_schedules=True, independent_native_exports=True,
            same_development_evaluation_rows=True, original512_references_replayed=True,
            first_arrival_metrics_identical_within_source=True, budgets_match=True),
        scores=scores, contrasts=contrasts, audits=audits, completed_fit_runs=8,
        completed_reference_replays=4, new_scientific_updates=864, new_training_CE_examples=13824,
        official_test_read=False, new_source_training_steps=0,
        interpretation_scope=plan['scope'], new_method_contribution_established=False)
    with (ROOT / 'comparison.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps({k:v['macro_BAcc_pp'] for k,v in contrasts.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
