"""Plot every task's teacher compression residual and corresponding gradient change."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/teacher_output_compression.json'
    data = json.loads(source.read_text())
    tasks = data['datasets']
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    y = np.arange(len(tasks))
    for group, label, offset, color in [('truth', 'Group by true label', -.18, '#0072b2'),
                                         ('teacher_argmax', 'Group by teacher prediction', .18, '#d55e00')]:
        rows = [data['per_task'][t]['conditioned'][group] for t in tasks]
        residual = [100*r['representations']['centered_logits']['within_fraction'] for r in rows]
        change = [100*r['target_comparison']['relative_output_gradient_l2_change'] for r in rows]
        for ax, values in zip(axes, [residual, change]):
            ax.barh(y+offset, values, height=.34, color=color, label=label)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
    axes[0].set_yticks(y, [f'{t} (K={data["per_task"][t]["classes"]})' for t in tasks], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Within-group share of centered-logit variance (%)')
    axes[1].set_xlabel('Relative L2 change of initial KD output gradient (%)')
    axes[0].set_title('A. Residual variation after class means', loc='left')
    axes[1].set_title('B. Probability-target replacement still changes gradients', loc='left')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Specialist output compression on the existing fitting multiset')
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.text(.5, .035, 'Class means are estimated from teacher outputs; teacher-prediction groups retain expert decisions.', ha='center', fontsize=9)
    fig.text(.5, .012, 'Class-weighted within each task. Descriptive training analysis; output gradients omit parameter Jacobians.', ha='center', fontsize=9)
    artifacts = {}
    for suffix in ['png', 'pdf']:
        path = root/f'analysis/figures/teacher_output_compression_v2.{suffix}'
        assert not path.exists()
        fig.savefig(path, dpi=170, bbox_inches='tight')
        artifacts[str(path)] = sha256(path)
    result = {'source': str(source), 'source_sha256': sha256(source),
              'code_sha256': sha256(Path(__file__)), 'figures': artifacts,
              'label_correction': 'The right panel is norm(g_replacement-g_teacher)/norm(g_teacher), not the change of the scalar gradient norm. Earlier figure retained.'}
    path = root/'analysis/teacher_output_compression_figures_v2.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
