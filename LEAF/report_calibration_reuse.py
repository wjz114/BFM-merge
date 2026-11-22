"""Summarize the fixed-model old/fresh sample comparison without causal claims."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    path = ROOT/'seed0/calibration_reuse_diagnostic/run.json'
    run = json.loads(path.read_text())
    assert run['initial_calibration_bacc_replay_exact']
    assert not run['fitting_performed'] and not run['validation_inference_performed']
    assert not run['test_data_read'] and not run['test_inference_performed']
    assert sha('diagnose_calibration_reuse.py') == run['code_sha256']
    assert all(v['max_abs_logit_error'] <= 1e-5 for v in run['expert_cache_checks'].values())
    values = run['evaluations']
    all_tasks = list(values)
    disjoint = [t for t in all_tasks if run['input_records'][t]['overlap'] == 0]
    assert len(all_tasks) == 16 and len(disjoint) == 15
    differences = {}
    for model in ['initial_merger', 'own_expert']:
        changes = {t: 100*(values[t][model]['old']['BAcc']-values[t][model]['fresh']['BAcc'])
                   for t in all_tasks}
        differences[model] = {
            'old_minus_fresh_pp': float(np.mean(list(changes.values()))),
            'fully_disjoint15_old_minus_fresh_pp': float(np.mean([changes[t] for t in disjoint])),
            'per_task_old_minus_fresh_pp': changes,
            'old_better_tasks': sum(v > 0 for v in changes.values()),
            'fresh_better_tasks': sum(v < 0 for v in changes.values()),
            'leave_one_task_out_old_minus_fresh_pp': {
                task: float(np.mean([v for t, v in changes.items() if t != task])) for task in all_tasks
            },
        }
    paired = {t: differences['initial_merger']['per_task_old_minus_fresh_pp'][t]
              - differences['own_expert']['per_task_old_minus_fresh_pp'][t] for t in all_tasks}
    result = {
        'unix': time.time(), 'source': str(path), 'source_sha256': sha(path),
        'summaries': run['summaries'], 'differences': differences,
        'merger_minus_expert_old_fresh_gap_pp': float(np.mean(list(paired.values()))),
        'fully_disjoint15_merger_minus_expert_gap_pp': float(np.mean([paired[t] for t in disjoint])),
        'per_task_merger_minus_expert_gap_pp': paired,
        'maximum_expert_cache_replay_error': max(v['max_abs_logit_error'] for v in run['expert_cache_checks'].values()),
        'expert_cache_changed_predictions': sum(v['changed_predictions'] for v in run['expert_cache_checks'].values()),
        'seconds': run['seconds'], 'code_sha256': sha(__file__),
        'interpretation': 'The frozen merger old/fresh fit gap is compared with its own experts on the identical rows and label positions. This is descriptive evidence about reuse of merger-fitting examples. It neither causally isolates memorization nor proves that fresh-row training improves validation. The fresh512 intervention is reported separately. Both pools were used to train the original experts; the15-task sensitivity removes ADHD\'s unavoidable108 overlapping rows, not all possible subject/recording dependence.',
    }
    output = ROOT/'analysis/calibration_reuse_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summaries': run['summaries'],
                      'old_minus_fresh_pp': {k: v['old_minus_fresh_pp'] for k, v in differences.items()},
                      'difference_in_gaps_pp': result['merger_minus_expert_old_fresh_gap_pp'],
                      'expert_cache_max_error': result['maximum_expert_cache_replay_error'],
                      'expert_cache_prediction_changes': result['expert_cache_changed_predictions']}, indent=2))


if __name__ == '__main__':
    main()
