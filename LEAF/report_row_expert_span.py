"""Plot and integrate the fixed-update row-partition intervention."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/row_expert_span_summary.json'
    joint_path = root/'analysis/own_expert_repair_joint_gap_decomposition.json'
    data, joint = [json.loads(p.read_text()) for p in [source, joint_path]]
    macro, geometry = data['macro']['validation'], data['geometry_totals']
    base, full = macro['initial'], macro['full_update']
    joint_bacc = joint['macro_bacc']['joint_bacc']
    interactions = {partition: 100*(full-macro[partition+'_parallel']-macro[partition+'_orthogonal']+base)
                    for partition in ['row', 'shuffled']}
    replay = [r for t in data['records'].values() for r in t['native_reference_replay'].values()]
    integration = {
        'sources': {str(p): sha256(p) for p in [source, joint_path, Path(__file__)]},
        'macro_bacc': macro, 'joint_macro_bacc': joint_bacc,
        'remaining_joint_gap_pp': {m: 100*(joint_bacc-v) for m, v in macro.items()},
        'fraction_of_full_ce_gain_retained': {m: (v-base)/(full-base) for m, v in macro.items()},
        'paired_component_interaction_pp': interactions,
        'captured_update_energy_fraction': {
            'tensor': geometry['tensor_parallel_energy']/geometry['delta_energy'],
            **{p: geometry[p]['captured_energy_fraction'] for p in ['row', 'shuffled']}},
        'coefficient_count_equal': geometry['row']['coefficient_count'] == geometry['shuffled']['coefficient_count'],
        'actual_rank_equal': geometry['row']['rank_sum'] == geometry['shuffled']['rank_sum'],
        'max_native_reference_logit_error': max(r['max_abs_logit_error'] for r in replay),
        'reference_changed_prediction_rows': sum(len(r['changed_rows']) for r in replay),
        'scope': 'Post-result integration of a predeclared fixed-update decomposition. Same numerical dimension and group sizes, but projected norms differ. The CE update was trained first; retention is descriptive and not the performance of an independently optimized coefficient method. Component interaction is a finite checkpoint intervention, not a Hessian estimate. Joint scores appear only as a report reference.'}
    output = root/'analysis/row_expert_span_integration.json'
    with output.open('x') as f:
        json.dump(integration, f, indent=2); f.write('\n')

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    energy_labels = ['Whole tensor', 'Operator row', 'Shuffled groups']
    energies = [100*integration['captured_update_energy_fraction'][k] for k in ['tensor', 'row', 'shuffled']]
    colors = ['#64748b', '#0072B2', '#D55E00']
    axes[0].bar(np.arange(3), energies, color=colors)
    for i, v in enumerate(energies): axes[0].text(i, v+.6, f'{v:.2f}%', ha='center', fontsize=9)
    axes[0].set_xticks(np.arange(3), energy_labels, rotation=20)
    axes[0].set_ylim(0, max(energies)*1.2)
    axes[0].set_ylabel('Captured squared update norm (%)')
    axes[0].set_title('A. Fixed CE-update geometry', loc='left')
    x = np.arange(2)
    for i, part in enumerate(['parallel', 'orthogonal']):
        values = [100*(macro[p+'_'+part]-base) for p in ['row', 'shuffled']]
        axes[1].bar(x+(i-.5)*.34, values, width=.34, label='Projected' if i == 0 else 'Complement', color=['#0072B2', '#56B4E9'][i])
    axes[1].axhline(100*(full-base), color='#333333', ls='--', label='Full CE update')
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set_xticks(x, ['Operator row', 'Shuffled groups'])
    axes[1].set_ylabel('Validation macro BAcc gain (pp)')
    axes[1].set_title('B. Native checkpoint interventions', loc='left')
    axes[1].legend(fontsize=8)
    effects = data['contrasts']['validation']['row_parallel_vs_shuffled_parallel']['per_task_gain_pp']
    tasks = list(effects)
    axes[2].barh(np.arange(len(tasks)), [effects[t] for t in tasks], color=['#0072B2' if effects[t] >= 0 else '#D55E00' for t in tasks])
    axes[2].set_yticks(np.arange(len(tasks)), [t.replace('MI_', '').replace('EMO_', '').replace('_seg4', '') for t in tasks], fontsize=7)
    axes[2].invert_yaxis(); axes[2].axvline(0, color='#777777', lw=.8)
    axes[2].set_xlabel('Row minus shuffled projection (pp)')
    axes[2].set_title('C. All16 exposed validation tasks', loc='left')
    fig.text(.5, .01, 'Projection of a previously trained CE update; same row/control dimension. This is a diagnostic, not a newly trained merging method.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for suffix in ['png', 'pdf']:
        fig.savefig(root/f'analysis/figures/row_expert_span.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output': str(output), 'sha256': sha256(output),
                      'integration': {k: v for k, v in integration.items() if k not in ['sources', 'scope']}}, indent=2))


if __name__ == '__main__':
    main()
