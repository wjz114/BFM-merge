"""Plot all tasks and both pools from the frozen last-projection audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/preprojection_representation_summary.json'
    data = json.loads(source.read_text())
    tasks = data['datasets']
    y = np.arange(len(tasks))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    for j, (pool, rep, color, hatch) in enumerate([
            ('original', 'prehead', '#0072b2', ''), ('replacement', 'prehead', '#0072b2', '//'),
            ('original', 'unit', '#d55e00', ''), ('replacement', 'unit', '#d55e00', '//')]):
        values = [100*data['per_task'][t][pool]['models']['teacher']['variance']['truth'][rep]['within_fraction'] for t in tasks]
        ax.barh(y+(j-1.5)*.18, values, height=.17, color=color, hatch=hatch, edgecolor='white',
                label=f'{rep}, {pool}')
    for model, color, offset in [('initial', '#777777', -.22), ('ce', '#0072b2', 0), ('kl', '#d55e00', .22)]:
        for pool, marker, shift in [('original', 'o', -.04), ('replacement', 'x', .04)]:
            values = [data['per_task'][t][pool]['teacher_student_cka'][model]['prehead']['paired_minus_permutation_mean'] for t in tasks]
            bx.scatter([v if v is not None else np.nan for v in values], y+offset+shift,
                       color=color, marker=marker, s=30, label=f'{model}, {pool}')
    ax.set_yticks(y, tasks, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Teacher within-true-class share of total variance (%)')
    bx.set_xlabel('Prehead class-centered CKA minus within-class permutation mean')
    ax.set_title('A. Variation before projection and after normalization', loc='left')
    bx.set_title('B. Paired teacher/student structure before projection', loc='left')
    bx.axvline(0, color='#555555', lw=.8)
    for axis in [ax, bx]:
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=7, loc='lower right')
    fig.suptitle('Last-projection representation audit: all16tasks, both fixed training subsets')
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.text(.5, .03, 'Prehead =2048-dimensional final Linear input; unit =768-dimensional normalized output.', ha='center', fontsize=9)
    fig.text(.5, .01, '32 within-class permutations; class-balanced weights; dependent training data. Geometry alone does not establish useful knowledge or causality.', ha='center', fontsize=9)
    artifacts = {}
    for suffix in ['png', 'pdf']:
        path = root/f'analysis/figures/preprojection_representations.{suffix}'
        assert not path.exists()
        fig.savefig(path, dpi=170, bbox_inches='tight')
        artifacts[str(path)] = sha256(path)
    result = {'source': str(source), 'source_sha256': sha256(source), 'code_sha256': sha256(Path(__file__)), 'figures': artifacts}
    path = root/'analysis/preprojection_representation_figures.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
