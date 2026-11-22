"""Compare old, replacement and mixed repair pools under the fixed label schedule."""
import json
import time
from pathlib import Path

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {
        'reused512': 'full_control_lr1e5_balanced_ce',
        'fresh512': 'fresh512_control_balanced_ce',
        'mixture': 'reuse_mixture_control_balanced_ce_v2',
        'full_pool': 'full_control_coverage_full_balanced_ce',
    }
    paths = {k: ROOT/'seed0'/v/'run.json' for k, v in directories.items()}
    runs = {k: json.loads(v.read_text()) for k, v in paths.items()}
    reference, fresh, mixed = (runs[k] for k in ['reused512', 'fresh512', 'mixture'])
    reference_fit = json.loads((Path(reference['plan']['source_fit'])/'run.json').read_text())
    tasks = reference['datasets']
    plan = mixed['plan']
    for key in ['fresh_data_audit', 'mixture_audit', 'mixture_mask', 'training_indices']:
        assert sha(plan[key]) == mixed[key+'_sha256']
    fresh_audit = json.loads(Path(plan['fresh_data_audit']).read_text())
    mixture_audit = json.loads(Path(plan['mixture_audit']).read_text())
    disjoint = fresh_audit['fully_disjoint_tasks']
    assert len(disjoint) == 15 and fresh_audit['total_new_rows'] == 8084
    assert mixed['training_pool'] == 'reuse_mixture512'
    assert mixed['training_indices_sha256'] == fresh['training_indices_sha256']
    assert mixed['fresh_data_audit_sha256'] == fresh['fresh_data_audit_sha256']
    assert mixed['schedule_sha256'] == fresh['schedule_sha256'] == reference['schedule_sha256']
    assert not mixed['test_data_read'] and not fresh['test_data_read']
    initial = {t: reference['initial_validation']['validation'][t]['BAcc'] for t in tasks}
    values, sources, costs = {'initial': initial}, {}, {}
    for name, run in runs.items():
        assert run['datasets'] == tasks and not run['engineering_smoke']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['fixed_tensors_exact'] and not run['test_inference_performed']
        assert run['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'source_fit',
                    'steps', 'batch_size', 'eval_every', 'average_steps', 'seed', 'lr']:
            assert run['plan'][key] == reference['plan'][key], (name, key)
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
        audit = mixed['dataset_audits'][task]
        expected = mixture_audit['datasets'][task]
        assert audit['class_counts'] == reference['dataset_audits'][task]['class_counts']
        assert audit['class_weights'] == reference_fit['ce_class_weights'][task]
        assert audit['batch_label_sequence_matches_original'] and expected['exact_original_batch_labels']
        assert audit['old_pool_exposures'] == audit['replacement_pool_exposures'] == 6400
        assert audit['training_pool_size'] == expected['unique_rows_sampled']
        assert audit['actual_new_row_exposures'] == expected['actual_new_row_exposures']
        assert audit['overlap_with_merger_calibration'] == 512
        assert audit['replacement_overlap_with_merger_calibration'] == fresh_audit['datasets'][task]['minimum_overlap']
    assert sum(v['unique_rows_sampled'] for v in mixture_audit['datasets'].values()) == 16276
    contrasts = {}
    for left, right in [('mixture', 'reused512'), ('mixture', 'fresh512'),
                        ('mixture', 'full_pool'), ('mixture', 'initial'),
                        ('fresh512', 'reused512'), ('full_pool', 'reused512')]:
        gains = {t: 100*(values[left][t]-values[right][t]) for t in tasks}
        contrasts[left+'_vs_'+right] = {
            'mean_change_pp': float(np.mean(list(gains.values()))),
            'per_task_change_pp': gains,
            'fully_disjoint15_mean_change_pp': float(np.mean([gains[t] for t in disjoint])),
            'improved_tasks': sum(v > 0 for v in gains.values()),
            'declined_tasks': sum(v < 0 for v in gains.values()),
            'leave_one_task_out_change_pp': {
                t: float(np.mean([v for other, v in gains.items() if other != t])) for t in tasks
            },
        }
    failure_path = ROOT/'analysis/reuse_mixture_failed_attempt.json'
    failure = json.loads(failure_path.read_text())
    assert failure['completed_updates'] == 0
    result = {
        'unix': time.time(), 'sources': sources, 'costs': costs,
        'macro_bacc': {k: float(np.mean(list(v.values()))) for k, v in values.items()},
        'per_task': values, 'contrasts': contrasts,
        'original512_reference_calibration_macro_bacc': {
            name: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks]))
            for name, run in runs.items()
        },
        'replacement512_macro_bacc': {
            name: float(np.mean([runs[name]['average_training_subset'][t]['BAcc'] for t in tasks]))
            for name in ['fresh512', 'mixture']
        },
        'data_audit': {
            'mixture_audit': plan['mixture_audit'], 'mixture_audit_sha256': mixed['mixture_audit_sha256'],
            'fresh_audit': plan['fresh_data_audit'], 'fresh_audit_sha256': mixed['fresh_data_audit_sha256'],
            'two_stage_distinct_rows': {'reused512': 8192, 'fresh512': 16276, 'mixture': 16276},
            'fully_disjoint_tasks': disjoint, 'per_task_mixture': mixture_audit['datasets'],
        },
        'export_max_logit_error': max(v['max_abs_logit_error'] for v in mixed['export_checks'].values()),
        'failed_attempt': {'path': str(failure_path), 'sha256': sha(failure_path),
                           'completed_updates': 0, 'output_preserved': True},
        'code_sha256': sha(__file__),
        'scope': 'One frozen seed0 bank/merger and exposed validation. Old, replacement and mixture controls match every batch label, class weights and update budget. The mixture and replacement-only controls have the same cross-stage union16276, but the mixture trains on1024 distinct rows/task except ADHD916. ADHD actual exposures7753old/5047new, with up to7 duplicate identities/batch; the predefined15-task sensitivity excludes it. Full-pool has different class weights. Row disjointness is not subject independence and all rows were already available to experts. The failed mixture attempt completed zero updates and is retained; v2 changes only the indexing dtype. No new-method or independent-test claim. Historical reused512 code read and discarded test arrays without inference; new workers avoid test reads.',
    }
    output = ROOT/'analysis/reuse_mixture_control_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'],
                      'contrasts_pp': {k: v['mean_change_pp'] for k, v in contrasts.items()},
                      'fully_disjoint15_contrasts_pp': {k: v['fully_disjoint15_mean_change_pp'] for k, v in contrasts.items()},
                      'export_max_logit_error': result['export_max_logit_error']}, indent=2))


if __name__ == '__main__':
    main()
