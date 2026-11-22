"""Report both frozen native-dropout controls after full runs finish."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def contrast(treatment, control, tasks):
    delta = (treatment-control)*100
    deleted = {task: float(np.delete(delta, i).mean()) for i, task in enumerate(tasks)}
    return {'macro_gain_pp': float(delta.mean()), 'per_task_gain_pp': dict(zip(tasks, delta.tolist())),
            'improved': int((delta > 0).sum()), 'declined': int((delta < 0).sum()),
            'ties': int((delta == 0).sum()), 'leave_one_task_out_pp': deleted,
            'task_deletion_range_pp': [min(deleted.values()), max(deleted.values())]}


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/dropout_ce_design.json'
    design = json.loads(design_path.read_text())
    sources = {str(design_path): sha256(design_path)}
    tasks, runs, values = design['datasets'], {}, {}
    base_control = json.loads(Path(design['arms']['base']['control']).read_text())
    for name, digest in design['source_sha256'].items():
        assert sha256(Path(name)) == digest, name
    for arm, spec in design['arms'].items():
        path, plan_path, control_path = Path(spec['outdir'])/'run.json', Path(spec['plan']), Path(spec['control'])
        run, plan, control = [json.loads(p.read_text()) for p in [path, plan_path, control_path]]
        assert sha256(control_path) == spec['control_sha256']
        assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
        assert plan['design_sha256'] == sha256(design_path)
        assert run['code_sha256'] == sha256(Path(design['worker']))
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['average_steps'] == design['average_steps']
        assert run['schedule_sha256'] == control['schedule_sha256'] == design['schedule_sha256']
        assert run['dataset_audits'] == base_control['dataset_audits']
        for task in tasks:
            new_audit, old_audit = run['dataset_audits'][task], control['dataset_audits'][task]
            if arm == 'merged':
                # The old CE worker also recorded an unused teacher cache. The
                # newer teacher-free worker instead records explicit readouts.
                new_extra = {'labels', 'instruction_hash', 'prototypes_hash'}
                old_extra = {'teacher_hash', 'teacher_expert_sha256'}
                assert set(new_audit)-set(old_audit) == new_extra
                assert set(old_audit)-set(new_audit) == old_extra
                assert {k: v for k, v in new_audit.items() if k not in new_extra} == {k: v for k, v in old_audit.items() if k not in old_extra}
            else:
                assert new_audit == old_audit
        assert run['initial_validation'] == control['initial_validation']
        assert run['training_mode'] == 'native_dropout_fixed_bn' and run['fixed_tensors_exact']
        assert run['trainable_parameters'] == 25751552
        assert not run['engineering_smoke'] and not run['test_data_read'] and not run['test_inference_performed']
        assert run['checkpoint_sha256'] == sha256(path.with_name('merged.ckpt'))
        assert max(check['max_abs_logit_error'] for check in run['export_checks'].values()) <= 1e-5
        for key in ['eval_replay_exact', 'forward_state_unchanged', 'diagnostic_rng_restored', 'all_bn_in_eval_during_updates']:
            assert run['mode_check'][key]
        assert run['mode_check']['training_logits_max_difference'] > 0
        for name, digest in plan['source_sha256'].items():
            assert sha256(Path(name)) == digest, name
        sources.update({str(p): sha256(p) for p in [path, plan_path, control_path]})
        for mode, data in [('on', run), ('off', control)]:
            key = f'{arm}_{mode}'
            runs[key] = data
            values[key] = np.array([data['average_validation'][t]['BAcc'] for t in tasks])
    comparisons = {f'{arm}_on_vs_off': contrast(values[f'{arm}_on'], values[f'{arm}_off'], tasks)
                   for arm in design['arms']}
    comparisons.update({f'merged_vs_base_{mode}': contrast(values[f'merged_{mode}'], values[f'base_{mode}'], tasks)
                        for mode in ['on', 'off']})
    comparisons['initialization_by_dropout_interaction'] = contrast(values['merged_on']-values['merged_off'],
                                                                    values['base_on']-values['base_off'], tasks)
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    sources[str(joint_path)] = sha256(joint_path)
    joint_mean = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    merger_mean = float(np.mean([runs['merged_off']['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    means = {key: float(v.mean()) for key, v in values.items()}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.4]})
    x = np.arange(2)
    for offset, mode, label, color in [(-.18, 'off', 'Dropout off', '#64748b'), (.18, 'on', 'Native dropout on', '#2563eb')]:
        y = [means[f'{arm}_{mode}']*100 for arm in ['merged', 'base']]
        ax.bar(x+offset, y, width=.34, label=label, color=color)
        for px, py in zip(x+offset, y):
            ax.text(px, py+.08, f'{py:.2f}', ha='center', fontsize=9)
    ax.axhline(joint_mean*100, ls='--', color='#334155', label='Historical Joint reference')
    ax.axhline(merger_mean*100, ls=':', color='#94a3b8', label='Initial merger')
    ax.set_xticks(x, ['Merged initialization', 'Released base'])
    ax.set_ylim(min(min(means.values()), merger_mean)*100-1, max(max(means.values()), joint_mean)*100+1)
    ax.set_ylabel('Full16 validation macro BAcc (%)')
    ax.set_title('A. Fixed six-checkpoint means', loc='left', weight='bold')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.08))
    yy = np.arange(16)
    for offset, arm, color in [(-.18, 'merged', '#2563eb'), (.18, 'base', '#d97706')]:
        row = comparisons[f'{arm}_on_vs_off']
        bx.barh(yy+offset, [row['per_task_gain_pp'][t] for t in tasks], height=.34, color=color,
                label=f'{arm}: macro {row["macro_gain_pp"]:+.3f} pp')
    bx.set_yticks(yy, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.axvline(0, color='#334155', lw=.8)
    bx.set_xlabel('Native dropout on minus off (pp)')
    bx.set_title('B. All16 paired effects', loc='left', weight='bold')
    bx.legend(fontsize=8)
    for axis in [ax, bx]:
        axis.set_axisbelow(True)
        axis.grid(axis='y' if axis is ax else 'x', color='#e2e8f0', lw=.6)
    fig.suptitle('Native training dropout at fixed Adam 1e-5, data stream and 1600 updates', weight='bold')
    fig.text(.5, .012, 'One exposed validation bank/order. BN statistics fixed. Joint optimizer, rate groups and training length remain different.',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .08, 1, .95])
    figure_dir = root/'analysis/figures'
    for ext in ['png', 'pdf']:
        fig.savefig(figure_dir/f'dropout_ce_controls.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'sources': sources, 'datasets': tasks,
              'macro_bacc': means, 'contrasts': comparisons, 'initial_merger_macro_bacc': merger_mean,
              'joint_macro_bacc': joint_mean,
              'initial_merger_joint_gap_closed_percent': {k: (v-merger_mean)/(joint_mean-merger_mean)*100 for k, v in means.items()},
              'calibration_macro_bacc': {k: float(np.mean([r['average_calibration'][t]['BAcc'] for t in tasks])) for k, r in runs.items()},
              'seconds': {k: r['seconds'] for k, r in runs.items()},
              'checks': {'source_data_initial_metrics_and_exports_passed': True, 'both_fixed_initializations_retained': True,
                         'all_new_data_and_readout_audits_equal_base_control': True,
                         'legacy_merged_audit_schema': 'Compared every common field; the only old-only fields are unused teacher_hash/expert_sha256 and the only new-only fields are labels/instruction_hash/prototypes_hash.'},
              'figure_sha256': {ext: sha256(figure_dir/f'dropout_ce_controls.{ext}') for ext in ['png', 'pdf']},
              'scope': design['scope']+' Task-deletion ranges are sensitivity analyses, not confidence intervals. No independent confirmation.'}
    with (root/'analysis/dropout_ce_control_summary.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': means, 'contrasts': {k: v['macro_gain_pp'] for k, v in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
