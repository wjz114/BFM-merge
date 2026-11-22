"""Show all fixed training trajectories and the paired task-mixing effects."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_coverage_control import ROOT, sha


def main():
    summary_path = ROOT/'analysis/task_mixing_summary.json'
    summary = json.loads(summary_path.read_text())
    tasks = summary['datasets']
    styles = {'native128': ('Native128 CE', '#64748b'),
              'same_task_micro4': ('Same task, four microbatches', '#d97706'),
              'mixed_task_micro4': ('Four mixed tasks', '#2563eb')}
    trajectories, inputs = {}, {str(summary_path): sha(summary_path)}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={'width_ratios': [1.35, 1]})
    for mode, (label, color) in styles.items():
        path = Path(summary['sources'][mode]['path'])
        assert sha(path) == summary['sources'][mode]['sha256']
        run = json.loads(path.read_text())
        history_path = path.with_name('history.json')
        inputs[str(history_path)] = sha(history_path)
        history = json.loads(history_path.read_text())
        assert [row['step'] for row in history] == list(range(0, 1601, 160))
        average_steps = run['average_steps']
        late = np.array([[row['validation'][t]['BAcc'] for t in tasks]
                         for row in history if row['step'] in average_steps])
        assert late.shape == (6, 16)
        average = np.array([run['average_validation'][t]['BAcc'] for t in tasks])
        macro = np.array([row['macro_bacc'] for row in history])
        trajectories[mode] = {
            'checkpoint_steps': [row['step'] for row in history],
            'checkpoint_macro_bacc': macro.tolist(),
            'fixed_average_bacc': float(average.mean()),
            'final_bacc': float(macro[-1]),
            'mean_late_checkpoint_bacc': float(late.mean()),
            'late_checkpoint_temporal_sd_pp': float(100*late.mean(1).std(ddof=1)),
            'fixed_average_minus_mean_late_checkpoint_pp': float(100*(average.mean()-late.mean())),
            'per_task_late_temporal_sd_pp': dict(zip(tasks, (100*late.std(0, ddof=1)).tolist())),
            'per_task_average_minus_mean_late_pp': dict(zip(tasks, (100*(average-late.mean(0))).tolist()))}
        ax.plot([row['step'] for row in history], 100*macro, color=color, marker='o', ms=3, lw=1.7, label=label)
        ax.scatter([1780], [100*average.mean()], color=color, marker='*', s=160, zorder=5)
    ax.axhline(100*summary['macro_bacc']['Joint'], color='#334155', ls='--', lw=1, label='Joint reference')
    ax.axvspan(800, 1600, color='#e2e8f0', alpha=.35)
    ax.axvline(1680, color='#cbd5e1', lw=1)
    ax.set_xticks([0, 400, 800, 1200, 1600, 1780], ['0', '400', '800', '1200', '1600', 'Fixed\nmean'])
    ax.set_xlabel('Optimizer updates; stars show the prescribed six-checkpoint mean')
    ax.set_ylabel('Macro balanced accuracy (%)')
    ax.set_title('A. Full validation trajectory', loc='left', weight='bold')
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    ax.grid(axis='y', color='#e2e8f0', lw=.6)
    contrast = summary['contrasts']['mixed_task_micro4_vs_same_task_micro4']
    gains = np.array([contrast['per_task_gain_pp'][t] for t in tasks])
    y = np.arange(len(tasks))
    bx.barh(y, gains, color=['#2563eb' if g >= 0 else '#d97706' for g in gains], height=.65)
    bx.set_yticks(y, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.axvline(0, color='#334155', lw=.9)
    bx.axvline(gains.mean(), color='#64748b', ls='--', lw=1.2, label=f'Macro {gains.mean():+.3f} pp')
    bx.set_xlabel('Mixed minus same task, fixed mean (percentage points)')
    bx.set_title('B. All sixteen paired task effects', loc='left', weight='bold')
    bx.grid(axis='x', color='#e2e8f0', lw=.6)
    bx.set_axisbelow(True)
    bx.legend(loc='best', fontsize=9)
    fig.suptitle('Task mixing with matched samples, updates and microbatch accumulation', fontsize=13, weight='bold')
    fig.text(.5, .015, 'Single exposed-validation pilot. Shaded interval: fixed averaging window. Temporal variation is not seed uncertainty.', ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .05, 1, .95])
    figure_dir = ROOT/'analysis/figures'
    figure_dir.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(figure_dir/f'task_mixing_control.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {'unix': time.time(), 'code_sha256': sha(__file__), 'input_sha256': inputs,
              'trajectories': trajectories,
              'figure_sha256': {e: sha(figure_dir/f'task_mixing_control.{e}') for e in ['png', 'pdf']},
              'scope': 'Descriptive full validation trajectories and all16 paired task effects. Late temporal SD is across six dependent checkpoints, not uncertainty across seeds. Weight-average minus mean checkpoint BAcc is not a comparison with logit ensembling and does not isolate a generalization mechanism. No new inference or checkpoint selection.'}
    with (ROOT/'analysis/task_mixing_trajectories.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: {j: v[j] for j in ['fixed_average_bacc', 'final_bacc', 'late_checkpoint_temporal_sd_pp', 'fixed_average_minus_mean_late_checkpoint_pp']} for k, v in trajectories.items()}, indent=2))


if __name__ == '__main__':
    main()
