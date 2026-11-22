"""Separate reuse of merger-fitting rows from repair-pool size at fixed budget."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {
        'reused512': 'full_control_lr1e5_balanced_ce',
        'fresh512': 'fresh512_control_balanced_ce',
        'full_pool': 'full_control_coverage_full_balanced_ce',
    }
    paths = {name: ROOT/'seed0'/directory/'run.json' for name, directory in directories.items()}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    reference, fresh = runs['reused512'], runs['fresh512']
    audit_path = fresh['plan']['fresh_data_audit']
    assert sha(audit_path) == fresh['fresh_data_audit_sha256']
    audit = json.loads(open(audit_path).read())
    assert sha(fresh['plan']['training_indices']) == fresh['training_indices_sha256']
    assert fresh['schedule_sha256'] == reference['schedule_sha256']
    assert fresh['training_pool'] == 'fresh512' and not fresh['test_data_read']
    tasks = reference['datasets']
    initial = {t: reference['initial_validation']['validation'][t]['BAcc'] for t in tasks}
    values = {'initial': initial}
    sources, costs = {}, {}
    for name, run in runs.items():
        assert run['datasets'] == tasks and not run['engineering_smoke']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['fixed_tensors_exact'] and not run['test_inference_performed']
        for field in ['initial_checkpoint_sha256', 'calibration_sha256', 'source_fit',
                      'steps', 'batch_size', 'eval_every', 'average_steps', 'seed', 'lr']:
            assert run['plan'][field] == reference['plan'][field], (name, field)
        assert run['trainable_parameters'] == 25751552 and run['plan']['lr'] == 1e-5
        assert sha(paths[name].with_name('merged.ckpt')) == run['checkpoint_sha256']
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == initial[task]
            assert run['dataset_audits'][task]['eeg_hash'] == reference['dataset_audits'][task]['eeg_hash']
            assert run['export_checks'][task]['max_abs_logit_error'] <= 1e-5
        values[name] = {t: run['average_validation'][t]['BAcc'] for t in tasks}
        sources[name] = {'path': str(paths[name]), 'sha256': sha(paths[name]),
                         'checkpoint_sha256': run['checkpoint_sha256']}
        costs[name] = run['seconds']
    for task in tasks:
        a = fresh['dataset_audits'][task]
        assert a['class_counts'] == reference['dataset_audits'][task]['class_counts']
        assert a['batch_label_sequence_matches_original']
        assert a['overlap_with_merger_calibration'] == audit['datasets'][task]['minimum_overlap']
    fully_disjoint = audit['fully_disjoint_tasks']
    assert len(fully_disjoint) == 15 and audit['total_new_rows'] == 8084
    contrasts = {}
    for left, right in [('fresh512', 'reused512'), ('fresh512', 'full_pool'),
                        ('fresh512', 'initial'), ('full_pool', 'reused512')]:
        changes = {t: 100*(values[left][t]-values[right][t]) for t in tasks}
        contrasts[left+'_vs_'+right] = {
            'mean_change_pp': float(np.mean(list(changes.values()))),
            'per_task_change_pp': changes,
            'fully_disjoint15_mean_change_pp': float(np.mean([changes[t] for t in fully_disjoint])),
            'improved_tasks': sum(v > 0 for v in changes.values()),
            'declined_tasks': sum(v < 0 for v in changes.values()),
            'leave_one_task_out_change_pp': {
                task: float(np.mean([v for t, v in changes.items() if t != task])) for task in tasks
            },
        }
    result = {
        'unix': time.time(), 'sources': sources, 'costs': costs,
        'macro_bacc': {name: float(np.mean(list(v.values()))) for name, v in values.items()},
        'contrasts': contrasts, 'per_task': values,
        'original512_reference_calibration_macro_bacc': {
            name: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks]))
            for name, run in runs.items()
        },
        'fresh512_training_subset_macro_bacc': float(np.mean([fresh['average_training_subset'][t]['BAcc'] for t in tasks])),
        'data_audit': {'path': audit_path, 'sha256': sha(audit_path),
                       'total_new_rows': audit['total_new_rows'], 'fully_disjoint_tasks': fully_disjoint},
        'code_sha256': sha(__file__),
        'scope': 'One exploratory sample-selection seed with the same frozen seed0 merger and exposed validation. Fresh512 matches the reused512 class counts, every batch label and repair budget; its two-stage union has16276 unique rows versus8192 for reuse. Fifteen tasks are disjoint by stored trial row, ADHD retains the unavoidable108 old rows. These data were already used by the original experts; row disjointness is not subject/recording independence. Full-pool has different empirical class weights. No new-method or independent-test claim. New workers avoid test arrays; the historical reused512 trainer allocated and discarded them without inference.',
    }
    output = ROOT/'analysis/fresh_calibration_control_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'],
                      'contrasts_pp': {k: v['mean_change_pp'] for k, v in contrasts.items()},
                      'reference_calibration': result['original512_reference_calibration_macro_bacc'],
                      'fresh_training_subset': result['fresh512_training_subset_macro_bacc']}, indent=2))


if __name__ == '__main__':
    main()
