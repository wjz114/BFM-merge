"""Report all predeclared oracle component swaps and their concentration."""
import json
from pathlib import Path
from statistics import mean

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'seed0/joint_component_diagnostic/run.json'
    run = json.loads(source.read_text())
    assert run['oracle_only'] and not run['test_inference_performed'] and not run['test_data_read']
    tasks = list(run['evaluations']['merged'])
    assert len(tasks) == 16
    assert list(run['evaluations']) == run['plan']['conditions']
    gap = run['summaries']['validation']['joint']-run['summaries']['validation']['merged']
    contrasts = {}
    for condition, rows in run['evaluations'].items():
        if condition == 'merged':
            continue
        deltas = {t: 100*(rows[t]['validation']['BAcc']-run['evaluations']['merged'][t]['validation']['BAcc']) for t in tasks}
        ordered = sorted(deltas, key=deltas.get, reverse=True)
        net = sum(deltas.values())
        positive = sum(max(v,0) for v in deltas.values())
        contrasts[condition] = {
            'macro_gain_pp': mean(deltas.values()),
            'fraction_joint_gap_closed': mean(deltas.values())/(100*gap),
            'per_task_gain_pp': deltas, 'improved': sum(v>0 for v in deltas.values()),
            'declined': sum(v<0 for v in deltas.values()),
            'leave_one_task_out_gain_pp': {t: mean(v for other,v in deltas.items() if other!=t) for t in tasks},
            'largest_two_gain_tasks_posthoc': ordered[:2],
            'largest_two_fraction_of_signed_net': sum(deltas[t] for t in ordered[:2])/net if abs(net)>1e-10 else None,
            'largest_two_fraction_of_positive_gains': sum(max(deltas[t],0) for t in ordered[:2])/positive if positive else None,
            'mean_other14_gain_pp': mean(v for t,v in deltas.items() if t not in ordered[:2])}
    result = {'validation_macro_bacc': run['summaries']['validation'],
              'calibration_macro_bacc': run['summaries']['calibration'],
              'contrasts_vs_current_merger': contrasts, 'factorials': run['factorials'],
              'joint_parameter_counts': run['joint_parameter_counts'],
              'source_run_sha256': sha256(source), 'code_sha256': sha256(Path(__file__)),
              'seconds': run['seconds'], 'oracle_only': True,
              'interpretation': 'The complete Joint model improves4.1466pp. Substituting its tower alone helps on average but its net gain concentrates in two tasks; other14 average nearly zero. Other predeclared component/complement swaps hurt on average. This supports context-dependent co-adaptation in these interventions, not a unique causal attribution of the gap or a deployable method. No candidate may use Joint weights or predictions.',
              'scope': 'All16 full exposed validation tasks, seed0, FP32/eval. Both current merger and Joint baseline BAcc replay exactly per task. Largest-two concentration is descriptive posthoc, with all task effects retained.'}
    (root/'analysis/joint_component_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'validation_macro_bacc': result['validation_macro_bacc'],
                      'tower_contrast': contrasts['joint_tower'],
                      'macro_interactions': {k:v['interaction_pp'] for k,v in run['factorials']['validation'].items()}}, indent=2))


if __name__ == '__main__':
    main()
