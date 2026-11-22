"""Integrate the completed class-moment audit without additional inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    sources = [root/'analysis/prehead_class_moments_summary.json',
               root/'analysis/prehead_distillation_summary.json']
    moments, performance = [json.loads(p.read_text()) for p in sources]
    tasks = moments['datasets']
    assert tasks == performance['datasets']
    models = ['balanced_ce', 'ce_prehead_paired', 'ce_prehead_mean', 'ce_prehead_shuffled']
    keys = ['student_within_class_variance', 'centroid_mse_to_exact_teacher_mean',
            'within_class_covariance', 'true_class_centered_cosine']
    comparisons = {}
    for reference in ['balanced_ce', 'ce_prehead_mean', 'ce_prehead_shuffled']:
        rows = {}
        for task in tasks:
            values = moments['records'][task]['models']
            p = values['ce_prehead_paired']['class_moments']['weighted_totals']
            q = values[reference]['class_moments']['weighted_totals']
            change = {k: p[k]-q[k] for k in keys}
            change['minus_twice_covariance'] = -2*change['within_class_covariance']
            change['paired_mse'] = (values['ce_prehead_paired']['weighted_mse']['ce_prehead_paired']
                                   -values[reference]['weighted_mse']['ce_prehead_paired'])
            assert abs(change['paired_mse']-change['student_within_class_variance']
                       -change['centroid_mse_to_exact_teacher_mean']
                       -change['minus_twice_covariance']) < 1e-12
            rows[task] = change
        comparisons[reference] = {
            'per_task': rows,
            'macro': {k: float(np.mean([row[k] for row in rows.values()])) for k in change},
            'positive_tasks': {k: sum(row[k] > 0 for row in rows.values()) for k in change},
            'negative_tasks': {k: sum(row[k] < 0 for row in rows.values()) for k in change}}
    variance = {}
    for model in ['initial_cached', *models]:
        macro = moments['macro'][model]
        variance[model] = {
            'ratio_of_macro_student_to_teacher_variance': macro['student_within_class_variance']/macro['teacher_within_class_variance'],
            'tasks_with_student_variance_above_teacher': sum(
                moments['records'][task]['models'][model]['class_moments']['weighted_totals']['student_within_class_variance']
                > moments['records'][task]['models'][model]['class_moments']['weighted_totals']['teacher_within_class_variance']
                for task in tasks)}
    identity_error = max(v for row in moments['records'].values() for model in row['models'].values()
                         for v in model['class_moments']['loss_identity_absolute_errors'].values())
    result = {'sources': {str(p): sha256(p) for p in [*sources, Path(__file__)]},
              'datasets': tasks, 'paired_vs_reference': comparisons,
              'student_variance_reference': variance, 'macro': moments['macro'],
              'max_loss_identity_error': identity_error,
              'max_repeated_native_metric_difference': moments['max_repeated_native_metric_absolute_difference'],
              'class_moment_audit_forward_examples': moments['new_forward_examples'],
              'this_integration_forward_examples': 0, 'optimizer_updates': 0,
              'interpretation': 'Endpoint class-centered moment identities, not training-trajectory or parameter-gradient attribution. Variance reduction is not evidence of harmful feature collapse: teacher variance is smaller still in the macro average. Pairing preserves covariance and dispersion versus both target controls on every task, without a clear primary-performance advantage. These results do not justify prioritizing a residual-only loss or ruling out all teacher information.'}
    out = root/'analysis/prehead_class_moments_integration.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')

    labels = ['CE', 'Paired', 'Class mean', 'Class shuffle']
    colors = ['#64748b', '#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, title in [
        (axes[0], 'centroid_mse_to_exact_teacher_mean', 'A. Class-centroid discrepancy'),
        (axes[1], 'student_within_class_variance', 'B. Within-class variance'),
        (axes[2], 'true_class_centered_cosine', 'C. Within-class residual cosine')]:
        values = [moments['macro'][m][key] for m in models]
        ax.bar(np.arange(4), values, color=colors)
        for i, value in enumerate(values):
            ax.text(i, value+.004, f'{value:.3f}', ha='center', fontsize=9)
        ax.set_xticks(np.arange(4), labels, rotation=20)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylim(0, max(values)*1.22)
        ax.set_ylabel('Task-macro weighted moment' if ax != axes[2] else 'Task-macro centered cosine')
    teacher_variance = moments['macro']['initial_cached']['teacher_within_class_variance']
    axes[1].axhline(teacher_variance, color='#333333', ls='--', label='Teacher')
    axes[1].legend(fontsize=9)
    fig.text(.5, .012, 'Exact fitting multiset; each model centered at its own true-class means. Geometry does not establish useful transfer.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for suffix in ['png', 'pdf']:
        fig.savefig(root/f'analysis/figures/prehead_class_moments.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output': str(out), 'sha256': sha256(out),
                      'paired_vs_ce': comparisons['balanced_ce']['macro'],
                      'variance': variance, 'identity_error': identity_error}, indent=2))


if __name__ == '__main__':
    main()
