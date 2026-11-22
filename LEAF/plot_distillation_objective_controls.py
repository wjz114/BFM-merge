"""Plot all fixed distillation controls against the matched CE baseline."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_coverage_control import ROOT, sha


def main():
    paths = [ROOT/'analysis/full_model_distillation_summary.json', ROOT/'analysis/angular_distillation_summary.json']
    old, new = [json.loads(p.read_text()) for p in paths]
    tasks = new['datasets']
    assert tasks == old['datasets']
    names = ['ce_teacher_kl', 'ce_full_feature', 'ce_prototype_contrast', 'ce_angular_full_feature', 'ce_angular_prototype_contrast']
    labels = ['KL', 'Fixed norm\nfull feature', 'Fixed norm\ncontrast', 'Unit feature\nfull feature', 'Unit feature\ncontrast']
    contrasts = {**old['contrasts'], **new['contrasts']}
    rows = [contrasts[c+'_vs_balanced_ce'] for c in names]
    matrix = np.array([[r['per_task_gain_pp'][t] for r in rows] for t in tasks])
    means = np.array([r['macro_gain_pp'] for r in rows])
    low = np.array([min(r['leave_one_task_out_pp'].values()) for r in rows])
    high = np.array([max(r['leave_one_task_out_pp'].values()) for r in rows])
    assert np.allclose(matrix.mean(0), means, atol=1e-12, rtol=0)
    limit = max(.5, np.ceil(np.abs(matrix).max()*2)/2)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8.4), gridspec_kw={'width_ratios': [1.75, 1]})
    heat = axes[0].imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    axes[0].set_xticks(range(len(names)), labels, fontsize=9)
    axes[0].set_yticks(range(len(tasks)), tasks)
    axes[0].set_title('Per-task change from ordinary CE (percentage points)', pad=12)
    for i in range(len(tasks)):
        for j in range(len(names)):
            axes[0].text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                         color='white' if abs(matrix[i,j]) > .65*limit else '#222222')
    fig.colorbar(heat, ax=axes[0], shrink=.62, pad=.025, label='BAcc change (pp)')
    y = np.arange(len(names))
    axes[1].axvline(0, color='#777777', linewidth=1)
    axes[1].hlines(y, low, high, color='#617487', linewidth=3)
    axes[1].scatter(means, y, color='#173d65', s=48, zorder=3)
    axes[1].set_yticks(y, [label.replace('\n', ' / ') for label in labels])
    axes[1].invert_yaxis()
    axes[1].set_ylim(len(names)-.4, -.7)
    span = max(high.max()-low.min(), .1)
    axes[1].set_xlim(low.min()-.25*span, high.max()+.45*span)
    for i, value in enumerate(means):
        axes[1].text(high[i]+.04*span, i, f'{value:+.3f}', va='center', fontsize=10)
    axes[1].set_xlabel('Macro BAcc change from CE (pp)')
    axes[1].set_title('Mean and task-deletion sensitivity', pad=12)
    axes[1].grid(axis='x', alpha=.18)
    fig.suptitle('Own-expert distillation: fixed objectives, same data and training schedule', fontsize=14, y=.975)
    fig.text(.5, .027, 'Native validation, bank 0; prescribed six-checkpoint mean. Bars span leave-one-task-out means, not confidence intervals.\n'
             'Auxiliaries match initial output-gradient RMS per task; parameter gradients and subsequent dynamics differ.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.91, wspace=.62)
    out = ROOT/'analysis/figures'
    out.mkdir(exist_ok=True)
    outputs = []
    for extension in ['png', 'pdf']:
        path = out/f'distillation_objective_controls.{extension}'
        assert not path.exists()
        fig.savefig(path, dpi=180, facecolor='white')
        outputs.append(path)
    metadata = {'code_sha256': sha(__file__), 'sources': {str(p): sha(p) for p in paths},
                'outputs': {str(p): sha(p) for p in outputs}, 'macro_gain_pp': dict(zip(names, means.tolist())),
                'scope': 'Descriptive single-bank exposed-validation comparison; task-deletion ranges are not uncertainty intervals.'}
    with (out/'distillation_objective_controls_metadata.json').open('x') as f:
        json.dump(metadata, f, indent=2)
        f.write('\n')
    print(json.dumps({'outputs': [str(p) for p in outputs]}, indent=2))


if __name__ == '__main__':
    main()
