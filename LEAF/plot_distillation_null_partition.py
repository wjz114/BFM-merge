"""Plot the fixed training-stream output-gradient energy decomposition."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/full_distillation_null_partition.json'
    data = json.loads(source.read_text())
    tasks = list(data['per_task_energy_fractions'])
    rows = [data['per_task_energy_fractions'][t] for t in tasks]
    labels = [t.replace('MI_', '').replace('EMO_', '').replace('_seg4', '') for t in tasks]
    keys = ['own_contrast', 'other_contrasts_orthogonal_to_own', 'all_contrasts_null']
    names = ['Own-task contrast', 'Other contrasts', 'Null to all contrasts']
    colors = ['#3276A8', '#E29B42', '#BBC3CA']
    fig, ax = plt.subplots(figsize=(9, 7.2))
    left = np.zeros(len(tasks))
    for key, name, color in zip(keys, names, colors):
        values = 100*np.array([r[key] for r in rows])
        ax.barh(np.arange(len(tasks)), values, left=left, color=color, height=.72, label=name)
        left += values
    ax.set_yticks(np.arange(len(tasks)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel('Fraction of squared output-gradient norm (%)')
    ax.set_title('Initial full-feature distillation gradient energy', loc='left', pad=42, fontsize=14)
    ax.legend(loc='lower left', bbox_to_anchor=(-.01, 1.015), ncol=3, frameon=False, fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E4E8EB', linewidth=.7)
    means = data['mean_task_energy_fractions']
    fig.text(.30, .075, 'Task mean: '+ ' / '.join(f'{100*means[k]:.1f}%' for k in keys), fontsize=10)
    fig.text(.08, .025, 'Exact training stream; frozen initialization. This decomposition does not measure harmful transfer.\n'
             'Contrast-null changes preserve raw-score differences and argmax, but can change cosine confidence.', fontsize=8.5, color='#444444')
    fig.subplots_adjust(left=.30, right=.97, top=.85, bottom=.15)
    out = root/'analysis/figures'
    out.mkdir(exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(out/f'distillation_null_partition.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
