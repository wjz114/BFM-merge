"""Display the existing-cache subject-group sensitivity without inferential intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source_path = root/'analysis/coverage_subject_sensitivity.json'
    source = json.loads(source_path.read_text())
    tasks = list(source['records'])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.7), gridspec_kw={'width_ratios': [1.3, 1]})
    y = np.arange(len(tasks))
    for offset, mode, label, color in [(-.13, 'full_update_vs_initial', 'Full repair', '#2563eb'),
                                       (.13, 'orthogonal_only_vs_initial', 'Outside expert span only', '#d97706')]:
        for i, task in enumerate(tasks):
            r = source['records'][task]['contrasts'][mode]
            lo, hi = r['leave_one_subject_out_range_pp']
            ax.plot([lo, hi], [i+offset, i+offset], color=color, lw=1.8)
            ax.scatter([r['task_gain_pp']], [i+offset], color=color, s=25,
                       label=label if i == 0 else None, zorder=3)
    ax.set_yticks(y, tasks, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='#334155', lw=.9)
    ax.set_xlabel('Validation BAcc change from initial merger (pp)')
    ax.set_title('A. Delete one subject group at a time', loc='left', weight='bold', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    left = np.zeros(len(tasks))
    for field, label, color in [('subject_contribution_positive', 'Positive', '#2563eb'),
                                 ('subject_contribution_zero', 'Zero', '#cbd5e1'),
                                 ('subject_contribution_negative', 'Negative', '#d97706')]:
        val = np.array([source['records'][t]['contrasts']['full_update_vs_initial'][field]/source['records'][t]['subject_groups']*100 for t in tasks])
        bx.barh(y, val, left=left, height=.65, label=label, color=color)
        left += val
    assert np.allclose(left, 100)
    bx.set_yticks(y, [f'n = {source["records"][t]["subject_groups"]}' for t in tasks], fontsize=9)
    bx.invert_yaxis()
    bx.set_xlim(0, 100)
    bx.set_xlabel('Subject groups by contribution to full-repair task gain (%)')
    bx.set_title('B. Subject contributions have mixed signs', loc='left', weight='bold', fontsize=11)
    bx.legend(loc='lower center', bbox_to_anchor=(.5, -.19), ncol=3, fontsize=9)
    for axis in [ax, bx]:
        axis.set_axisbelow(True)
        axis.grid(axis='x', color='#e2e8f0', lw=.6)
    fig.suptitle('Full-pool repair: subject sensitivity on 11 mappable tasks, bank 0', weight='bold', fontsize=13)
    fig.text(.5, .015, 'Ranges show deletion sensitivity, not confidence intervals. These validation rows come from training subject groups; no unseen-subject claim.', ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .065, 1, .95])
    directory = root/'analysis/figures'
    for ext in ['png', 'pdf']:
        fig.savefig(directory/f'coverage_subject_sensitivity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {'code_sha256': sha256(Path(__file__)), 'source': str(source_path), 'source_sha256': sha256(source_path),
              'figure_sha256': {ext: sha256(directory/f'coverage_subject_sensitivity.{ext}') for ext in ['png', 'pdf']},
              'scope': 'Same-source descriptive visualization. Group percentages are not weighted equally in the original metric; signed contributions sum exactly to task BAcc change. No new inference or fitting.'}
    with (root/'analysis/coverage_subject_sensitivity_figure.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
