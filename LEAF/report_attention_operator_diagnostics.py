"""Summarize the completed attention diagnostics without further inference."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    paths = {'geometry': root/'analysis/attention_operator_geometry.json',
             'activation': root/'seed0/attention_activation_diagnostic/run.json',
             'ablation': root/'seed0/attention_quadratic_training_ablation/run.json'}
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    g, a, b = [data[k] for k in ['geometry', 'activation', 'ablation']]
    assert g['all512_head_matchings_strict_identity'] and g['total_moved_heads'] == 0
    assert a['model_state_exactly_unchanged'] and a['old512_bacc_exactly_replayed']
    assert b['model_state_exactly_unchanged'] and b['native_calibration_bacc_exactly_replayed']
    assert not a['validation_inference_performed'] and not b['validation_inference_performed']
    for name, expected in a['input_sha256'].items():
        assert sha256(Path(name)) == expected, name
    for name, expected in b['design']['input_sha256'].items():
        assert sha256(Path(name)) == expected, name
    tasks = a['datasets']
    assert tasks == list(b['records']) and len(tasks) == 16
    ratios = {mode: {metric: {'min': min(r['aggregate'][mode][metric] for r in a['records'].values()),
                            'max': max(r['aggregate'][mode][metric] for r in a['records'].values()),
                            'task_mean': float(np.mean([r['aggregate'][mode][metric] for r in a['records'].values()]))}
                     for metric in ['centered_score_relative_l2', 'output_relative_l2', 'attention_tv_mean']}
              for mode in a['aggregate']}
    contrasts = {}
    for mode, row in b['contrasts'].items():
        deltas = [r['BAcc']*100 for r in row['per_task'].values()]
        assert abs(np.mean(deltas)-row['macro_bacc_delta_pp']) < 1e-12
        contrasts[mode] = {k: v for k, v in row.items() if k not in ['per_task', 'task_deletion_bacc_pp']}
        contrasts[mode]['task_deletion_bacc_range_pp'] = [min(row['task_deletion_bacc_pp']), max(row['task_deletion_bacc_pp'])]
    figure_dir = root/'analysis/figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.3), gridspec_kw={'width_ratios': [.8, .95, 1.4]})
    ax, bx, cx = axes
    vals = [a['aggregate'][m]['output_relative_l2']*100 for m in a['aggregate']]
    ax.bar([0, 1], vals, color=['#64748b', '#d97706'], width=.6)
    for i, v in enumerate(vals):
        ax.text(i, v+.06, f'{v:.3f}%', ha='center', fontsize=10)
    ax.set_xticks([0, 1], ['Uniform factors\nvs operator mean', 'Merger: remove\nquadratic terms'], fontsize=9)
    ax.set_ylabel('Local attention output relative L2 difference (%)')
    ax.set_ylim(0, 3.7)
    ax.set_title('A. Fixed merger inputs', loc='left', weight='bold', fontsize=11)
    modes = ['remove_qk', 'remove_vo', 'remove_both']
    labels = ['QK', 'VO', 'Both']
    gains = [contrasts[m]['macro_bacc_delta_pp'] for m in modes]
    bx.bar(np.arange(3), gains, color=['#2563eb' if v >= 0 else '#d97706' for v in gains], width=.6)
    for i, m in enumerate(modes):
        lo, hi = contrasts[m]['task_deletion_bacc_range_pp']
        bx.plot([i, i], [lo, hi], color='#0f172a', lw=1.5)
        bx.plot([i-.07, i+.07], [lo, lo], color='#0f172a', lw=1.5)
        bx.plot([i-.07, i+.07], [hi, hi], color='#0f172a', lw=1.5)
        bx.text(i, gains[i]-(.025 if gains[i] < 0 else -.03), f'{gains[i]:+.3f}', ha='center', fontsize=9)
    bx.set_xticks(np.arange(3), labels)
    bx.set_xlabel('Quadratic terms removed in all32 attention modules')
    bx.set_ylabel('Macro calibration BAcc change (pp)')
    bx.set_ylim(-.23, .10)
    bx.axhline(0, color='#334155', lw=.8)
    bx.set_title('B. End-to-end intervention', loc='left', weight='bold', fontsize=11)
    per_task = [b['contrasts']['remove_both']['per_task'][t]['BAcc']*100 for t in tasks]
    y = np.arange(len(tasks))
    cx.barh(y, per_task, color=['#2563eb' if v >= 0 else '#d97706' for v in per_task], height=.65)
    cx.set_yticks(y, tasks, fontsize=8)
    cx.invert_yaxis()
    cx.axvline(0, color='#334155', lw=.8)
    cx.set_xlabel('Calibration BAcc change (pp)')
    cx.set_title('C. Removing both: all16 tasks', loc='left', weight='bold', fontsize=11)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis='x' if axis is cx else 'y', color='#e2e8f0', lw=.6)
    fig.suptitle('Attention operator diagnostics — original512 training examples per task', fontsize=13, weight='bold')
    fig.text(.5, .015, 'Bars in B: macro changes. Thin ranges: leave-one-task-out sensitivity, not confidence intervals. No validation or test inference.', ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .05, 1, .94])
    for ext in ['png', 'pdf']:
        fig.savefig(figure_dir/f'attention_operator_diagnostics.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'sources': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in paths.items()},
              'datasets': tasks, 'all512_head_matchings_strict_identity': True,
              'geometry_aggregate': g['aggregate'], 'activation_aggregate': a['aggregate'],
              'activation_per_task_ranges': ratios, 'calibration_aggregate': b['aggregate'], 'contrasts': contrasts,
              'checks': {'native_attention_max_abs_error': a['native_reconstruction_max_abs'],
                         'native_attention_max_relative_l2_error': a['native_reconstruction_max_relative_l2'],
                         'end_to_end_operator_replay_max_abs_logit_error': max(r['operator_replay_check']['max_abs_logit_error'] for r in b['records'].values()),
                         'native_and_operator_replay_bacc_equal_all_tasks': all(r['evaluations']['native']['BAcc'] == r['evaluations']['operator_replay']['BAcc'] for r in b['records'].values())},
              'diagnostic_process_seconds': {k: data[k]['seconds'] for k in paths},
              'figure_sha256': {ext: sha256(figure_dir/f'attention_operator_diagnostics.{ext}') for ext in ['png', 'pdf']},
              'decision': 'Do not expand this fixed quadratic-removal diagnostic into coefficient, layer, rank or bank sweeps. All three removals worsen macro training balanced CE; QK-only BAcc change is small and task-deletion-sensitive, while VO/both reduce BAcc. This does not refute optimized, activation-aware operator merging or establish a generalization effect. Uniform factor/operator differences are modest and do not identify the main Joint-gap mechanism.',
              'objective_status': 'Active and unachieved. These diagnostics do not establish a new method or substantial additional Joint-gap reduction.'}
    with (root/'analysis/attention_operator_summary.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'contrasts': contrasts, 'checks': result['checks'], 'decision': result['decision']}, indent=2))


if __name__ == '__main__':
    main()
