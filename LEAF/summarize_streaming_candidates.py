"""Combine frozen direct comparisons, including the two arrival orders."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256
from run_streaming_residual_merge import write_json


def difference(a, b, tasks):
    values = {t: 100 * (a[t]['BAcc'] - b[t]['BAcc']) for t in tasks}
    x = np.asarray(list(values.values()))
    return {'macro_gain_pp': float(x.mean()), 'per_task_gain_pp': values,
            'improved': int((x > 0).sum()), 'declined': int((x < 0).sum()), 'tied': int((x == 0).sum()),
            'worst_task_gain_pp': float(x.min()),
            'task_deletion_range_pp': [float(((x.sum() - x) / (len(x) - 1)).min()),
                                       float(((x.sum() - x) / (len(x) - 1)).max())]}


def main():
    root = Path('runs/streaming_merge_v3')
    destination = root / 'delivery.json'
    assert not destination.exists()
    report_paths = [Path('runs/streaming_merge_v2/comparison_all_forward.json'),
                    root / 'comparison_forward.json', root / 'comparison_reverse.json']
    runs, sources = {}, {}
    for path in report_paths:
        report = json.loads(path.read_text())
        assert report['matched_data_and_schedules'] and report['native_export_checks_pass']
        assert report['coefficient_averages_exact']
        sources[str(path)] = sha256(path)
        for label, folder in report['runs'].items():
            run_path = Path(folder) / 'run.json'
            assert sha256(run_path) == report['run_sha256'][label]
            run = json.loads(run_path.read_text())
            order = 'reverse' if folder.endswith('_reverse') else 'forward'
            runs[order, label] = run
    tasks = runs['forward', 'recursive']['order']
    comparisons, macros, last_block_retention = {}, {}, {}
    for order in ['forward', 'reverse']:
        metrics = {k: v['stages'][-1]['validation'] for (o, k), v in runs.items() if o == order}
        macros[order] = {k: float(np.mean([v[t]['BAcc'] for t in tasks])) for k, v in metrics.items()}
        comparisons[order] = {f'{a}_minus_{b}': difference(metrics[a], metrics[b], tasks)
                              for a, b in [('incoming_rows', 'recursive'), ('combined', 'recursive'),
                                           ('combined', 'incoming_rows')]}
        last_block_retention[order] = {}
        for label in ['recursive', 'incoming_rows', 'combined']:
            run = runs[order, label]
            old, new = run['order'][:12], run['order'][12:]
            before = run['stages'][11]['validation']
            after = run['stages'][15]['validation']
            initial_mean = float(np.mean([before[t]['BAcc'] for t in old]))
            final_mean = float(np.mean([after[t]['BAcc'] for t in old]))
            last_block_retention[order][label] = {
                'old12_stage12_macro_BAcc': initial_mean, 'old12_final_macro_BAcc': final_mean,
                'old12_change_pp': 100 * (final_mean - initial_mean),
                'new4_final_macro_BAcc': float(np.mean([after[t]['BAcc'] for t in new])),
                'scope': 'Change on the same12 old tasks while the last4 tasks arrive; not a dense per-arrival forgetting curve.'}
    m = macros['forward']
    result = {'goal_deliverables': 'Runnable merging candidates, actual direct comparisons and two algorithm iterations.',
              'report_sources': sources, 'macro_BAcc': macros, 'paired_comparisons': comparisons,
              'last_block_retention': last_block_retention,
              'forward_factorial_interaction_pp': 100 * ((m['combined'] - m['incoming_rows']) -
                                                        (m['history_tensor'] - m['recursive'])),
              'scientific_updates': sum(r['updates'] for r in runs.values()),
              'scientific_training_examples': sum(r['training_examples'] for r in runs.values()),
              'native_engineering_updates': 12,
              'coefficients': {'recursive': 766, 'history_tensor': 1149, 'incoming_rows': 174216,
                               'history_rows': 174216, 'combined': 174599},
              'historical_parameter_sum_bytes': 206012416,
              'formula_combined': 'theta0+a_tensor*(previous-theta0)+(b_tensor+c_row)*(incoming-theta0)+d_tensor*(mean_past_task_vectors-(previous-theta0))',
              'formula_incoming_rows': 'Set d_tensor=0 and omit historical parameter sum.',
              'deployment': 'One ordinary shared LEAF checkpoint; no expert bank or added inference branches.',
              'validation': 'All seen-task full validation at stages1/4/8/12/16; final16task comparison; all original data hashes and paired training schedules match; native export probes pass.',
              'context': {'offline_M0_BAcc': report['offline_M0_reference_macro_BAcc'],
                          'joint_BAcc': report['joint_reference_macro_BAcc'],
                          'offline_M0_training_examples': 1600 * 128,
                          'streaming_training_examples_per_run': 1620 * 16,
                          'comparison_scope': 'Offline M0 accesses all experts together and uses a different calibration budget. Joint is a different training procedure. Neither is a matched streaming control.'},
              'scope': 'One original expert bank; forward and reversed arrival orders; previously exposed development validation;512 retained training calibration rows per seen task, so replay storage grows with task count. Task-deletion ranges are not confidence intervals. Novelty and broad superiority remain unestablished.'}
    write_json(destination, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    labels = ['recursive', 'incoming_rows', 'combined']
    names = ['Recursive', 'Incoming rows', 'Incoming rows\n+ tensor history']
    colors = ['#397da4', '#d18b36']
    for j, order in enumerate(['forward', 'reverse']):
        values = [100 * macros[order][k] for k in labels]
        axes[0].bar(np.arange(3) + (j - .5) * .36, values, width=.36, color=colors[j], label=order)
        for i, v in enumerate(values):
            axes[0].text(i + (j - .5) * .36, v + .06, f'{v:.3f}', ha='center', fontsize=8)
        gain = comparisons[order]['combined_minus_incoming_rows']['per_task_gain_pp']
        axes[1].barh(np.arange(16) + (j - .5) * .36, [gain[t] for t in tasks],
                     height=.36, color=colors[j], label=order)
    axes[0].set_xticks(np.arange(3), names)
    all_values = [100 * macros[o][k] for o in macros for k in labels]
    axes[0].set_ylim(min(all_values) - .6, max(all_values) + .6)
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', lw=.8)
    axes[1].set_xlabel('Added tensor history: task gain (percentage points)')
    axes[1].legend()
    fig.suptitle('Streaming merge iterations under two arrival orders')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(root / f'delivery.{extension}', dpi=180)
    print(json.dumps({'macro_BAcc': macros,
                      'paired_gain_pp': {o: {k: v['macro_gain_pp'] for k, v in c.items()}
                                         for o, c in comparisons.items()},
                      'factorial_interaction_pp': result['forward_factorial_interaction_pp']}, indent=2))


if __name__ == '__main__':
    main()
