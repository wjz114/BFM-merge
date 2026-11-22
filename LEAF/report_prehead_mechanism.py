"""Integrate completed prehead controls with exact endpoint loss decomposition."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    paths = [root/f'analysis/{name}.json' for name in
             ['prehead_distillation_summary', 'prehead_fitting_summary', 'prehead_pairing_decomposition']]
    performance, fitting, pairing = [json.loads(p.read_text()) for p in paths]
    tasks = performance['datasets']
    assert tasks == fitting['datasets'] == pairing['datasets']
    order = ['balanced_ce', 'ce_prehead_paired', 'ce_prehead_mean', 'ce_prehead_shuffled']
    effects = {}
    for model in order:
        rows = {}
        for task in tasks:
            fit = fitting['records'][task]
            new = fit['models'][model]
            ce = fit['models']['balanced_ce']
            geometry = pairing['records'][task]
            distance = geometry[model]['student_error_to_teacher_class_mean']-geometry['balanced_ce']['student_error_to_teacher_class_mean']
            cross = -2*(geometry[model]['paired_residual_inner_product']-geometry['balanced_ce']['paired_residual_inner_product'])
            total = new['weighted_mse']['ce_prehead_paired']-ce['weighted_mse']['ce_prehead_paired']
            assert abs(total-distance-cross) < 1e-12
            rows[task] = {'class_mean_distance_term': distance, 'residual_cross_term': cross,
                          'paired_mse_difference': total, 'scaled_class_mean_distance_term': fit['scale']*distance,
                          'scaled_residual_cross_term': fit['scale']*cross,
                          'scaled_paired_mse_difference': fit['scale']*total}
        macro = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
        effects[model] = {'per_task': rows, 'macro': macro,
                         'tasks_with_lower_class_mean_distance': sum(r['class_mean_distance_term'] < 0 for r in rows.values()),
                         'tasks_with_increased_raw_paired_inner_product': sum(r['residual_cross_term'] < 0 for r in rows.values())}
    mse_ce = fitting['macro']['balanced_ce']['weighted_mse']['ce_prehead_paired']
    mse_paired = fitting['macro']['ce_prehead_paired']['weighted_mse']['ce_prehead_paired']
    result = {'sources': {str(p): sha256(p) for p in [*paths, Path(__file__)]}, 'datasets': tasks,
              'endpoint_paired_mse_decomposition_vs_ce': effects,
              'paired_mse_relative_reduction_of_task_macro': 1-mse_paired/mse_ce,
              'remaining_joint_gap_pp': {k: 100*(performance['joint_macro_bacc']-v) for k, v in performance['macro_bacc'].items()},
              'extra_fraction_ce_joint_gap_closed': {k: (v-performance['macro_bacc']['balanced_ce'])/(performance['joint_macro_bacc']-performance['macro_bacc']['balanced_ce'])
                                                    for k, v in performance['macro_bacc'].items()},
              'scope': 'Post-result integration of already fixed primary performance and fitting/pairing diagnostics. Exact endpoint loss algebra, not trajectory mediation or parameter-gradient attribution. Q_mean includes BOTH student within-class dispersion and student/teacher class-centroid discrepancy; these cannot be separated from the stored losses. Raw inner product and normalized cosine answer different questions. No claim that all teacher residual information is useless or that feature collapse has been established.'}
    out = root/'analysis/prehead_mechanism_integration.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    labels = ['CE', 'Paired', 'Class mean', 'Class shuffle']
    colors = ['#64748b', '#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(3)
    distance = [effects[m]['macro']['class_mean_distance_term'] for m in order[1:]]
    cross = [effects[m]['macro']['residual_cross_term'] for m in order[1:]]
    axes[0].bar(x-.17, distance, width=.34, label='Change in Q_mean', color='#0072B2')
    axes[0].bar(x+.17, cross, width=.34, label='Change in -2C', color='#D55E00')
    axes[0].scatter(x, np.asarray(distance)+cross, marker='D', color='black', s=30, label='Total paired-MSE change')
    axes[0].axhline(0, color='#777777', lw=.7)
    axes[0].set_xticks(x, labels[1:], rotation=20)
    axes[0].set_ylabel('Task-macro weighted loss difference vs CE')
    axes[0].set_title('A. Exact endpoint decomposition', loc='left')
    axes[0].legend(fontsize=8)
    for ax, key, title, ylabel in [
        (axes[1], 'paired_residual_inner_product', 'B. Raw residual inner product', 'Weighted inner product C'),
        (axes[2], 'paired_residual_cosine', 'C. Normalized residual alignment', 'Weighted residual cosine')]:
        values = [pairing['macro'][m][key] for m in order]
        ax.bar(np.arange(4), values, color=colors)
        ax.axhline(pairing['macro']['initial_cached'][key], color='#444444', ls='--', label='Initial merger')
        for i, value in enumerate(values):
            ax.text(i, value+.003, f'{value:.3f}', ha='center', fontsize=9)
        ax.set_xticks(np.arange(4), labels, rotation=20)
        ax.set_ylim(0, max([*values, pairing['macro']['initial_cached'][key]])*1.25)
        ax.set_title(title, loc='left'); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    fig.text(.5, .01, 'Training multiset; residuals centered at teacher class means. Endpoint identities do not establish useful transfer or variance collapse.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ['png', 'pdf']:
        fig.savefig(root/f'analysis/figures/prehead_fitting_mechanism.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output': str(out), 'sha256': sha256(out),
        'relative_paired_mse_reduction': result['paired_mse_relative_reduction_of_task_macro'],
        'paired_vs_ce': effects['ce_prehead_paired'], 'remaining_joint_gap_pp': result['remaining_joint_gap_pp']}, indent=2))


if __name__ == '__main__':
    main()
