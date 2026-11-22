"""Plot completed bank-seed comparisons; points show seeds, not confidence intervals."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/bank_coverage_replication_summary.json'
    data = json.loads(source.read_text())
    out = root/'analysis/figures'
    out.mkdir(exist_ok=True)
    colors = ['#1d6d9d', '#d17a18', '#7954a1']
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 6.8), gridspec_kw={'width_ratios': [1, 1.25]})
    for i, row in enumerate(data['replicas']):
        values = [row['initial_macro_bacc'], row['fixed_average_macro_bacc']['original512'],
                  row['fixed_average_macro_bacc']['full'], row['joint_macro_bacc']]
        left.plot(range(4), np.asarray(values)*100, marker='o', color=colors[i], lw=1.5, label=f'Bank seed {i}')
    left.set_xticks(range(4), ['Initial\nmerger', 'Repair\nold 512', 'Repair\nfull pool', 'Matched\nJoint'])
    left.set_ylabel('Dataset-macro balanced accuracy (%)')
    left.set_ylim(62.8, 68.5)
    left.set_title('Fixed averages across three expert banks', loc='left', fontsize=11, weight='bold')
    left.legend(frameon=False, loc='upper left')
    left.text(.03, .40, 'Full-pool gain over initial:\n'
              f"{data['mean_gain_over_initial_pp']:.2f} ± {data['sample_sd_gain_over_initial_pp']:.2f} pp (sample SD)\n"
              f"Mean Joint-gap closure: {100*data['mean_fraction_initial_joint_gap_closed']:.1f}%",
              transform=left.transAxes, fontsize=10, linespacing=1.7)
    tasks = list(data['per_task'])
    y = np.arange(len(tasks))
    gains = np.asarray([data['per_task'][t]['full_gains_over_initial_pp'] for t in tasks])
    for i in range(3):
        right.scatter(gains[:, i], y+(i-1)*.14, color=colors[i], s=23, zorder=3)
    right.scatter(gains.mean(1), y, color='#222222', marker='|', s=110, zorder=4, label='Three-seed mean')
    right.axvline(0, color='#777777', lw=.8)
    right.set_yticks(y, [t.replace('EMO_', '').replace('_seg4', '') for t in tasks], fontsize=9)
    right.invert_yaxis()
    right.set_xlabel('Full-pool repair minus initial merger (percentage points)')
    right.set_title('Task effects; each colored point is one seed', loc='left', fontsize=11, weight='bold')
    right.legend(frameon=False, loc='lower right', fontsize=9)
    for axis in [left, right]:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y' if axis is left else 'x', color='#dddddd', lw=.6, zorder=0)
    fig.suptitle('Broader training-pool repair: stability evidence for a baseline', x=.06, ha='left', fontsize=14, weight='bold')
    fig.text(.06, .022, 'Same pretrained backbone and exposed validation. Expert/calibration/optimizer seeds vary together.\n'
             'Repairs: 1,600 updates × 128 examples; fixed six-checkpoint mean. Full pool adds data access. No confidence intervals shown.',
             fontsize=9, color='#444444')
    fig.tight_layout(rect=(.02, .08, 1, .94), w_pad=3)
    paths = []
    for suffix in ['png', 'pdf']:
        p = out/f'bank_coverage_replication.{suffix}'
        fig.savefig(p, dpi=180)
        paths.append(p)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'bank_coverage_replication_metadata.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': sha(source), 'code_sha256': sha(__file__),
        'outputs': {str(p): sha(p) for p in paths},
        'scope': 'Three observed seed values and their mean; no inferential interval or independent external evidence.'}, indent=2)+'\n')
    print('\n'.join(str(p) for p in paths))


if __name__ == '__main__':
    main()
