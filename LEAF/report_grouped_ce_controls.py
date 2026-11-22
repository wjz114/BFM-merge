"""Report the fixed tokenizer/rest LR-ratio pair against existing flat controls."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def contrast(treatment, control, tasks):
    delta = (treatment-control)*100
    deleted = {task: float(np.delete(delta, i).mean()) for i, task in enumerate(tasks)}
    return {'macro_gain_pp': float(delta.mean()), 'per_task_gain_pp': dict(zip(tasks, delta.tolist())),
            'improved': int((delta > 0).sum()), 'declined': int((delta < 0).sum()),
            'ties': int((delta == 0).sum()), 'leave_one_task_out_pp': deleted,
            'task_deletion_range_pp': [min(deleted.values()), max(deleted.values())]}


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/grouped_ce_design.json'
    design = json.loads(design_path.read_text())
    sources = {str(design_path): sha256(design_path)}
    for name, digest in design['source_sha256'].items():
        assert sha256(Path(name)) == digest, name
    tasks, runs, values, movement = design['datasets'], {}, {}, {}
    base_control = json.loads(Path(design['arms']['base']['control']).read_text())
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
                new_extra = {'labels', 'instruction_hash', 'prototypes_hash'}
                old_extra = {'teacher_hash', 'teacher_expert_sha256'}
                assert set(new_audit)-set(old_audit) == new_extra
                assert set(old_audit)-set(new_audit) == old_extra
                assert {k: v for k, v in new_audit.items() if k not in new_extra} == {k: v for k, v in old_audit.items() if k not in old_extra}
            else:
                assert new_audit == old_audit
        assert run['initial_validation'] == control['initial_validation']
        assert run['training_mode'] == 'eval_dropout_off_fixed_bn' and run['fixed_tensors_exact']
        assert run['trainable_parameters'] == 25751552
        assert not run['engineering_smoke'] and not run['test_data_read'] and not run['test_inference_performed']
        assert max(check['max_abs_logit_error'] for check in run['export_checks'].values()) <= 1e-5
        audit = run['optimizer_audit']
        assert audit['optimizer'] == 'Adam' and audit['weight_decay'] == 0 and audit['tokenizer_multiplier'] == 10
        fast, slow = audit['groups']['tokenizer'], audit['groups']['rest']
        assert fast['lr'] == 1e-4 and slow['lr'] == 1e-5
        assert len(fast['names']) == 6 and len(slow['names']) == 377
        assert fast['parameters'] == 269056 and fast['parameters']+slow['parameters'] == 25751552
        assert all(n.startswith('tower.tokenizer.') for n in fast['names'])
        assert not any(n.startswith('tower.tokenizer.') for n in slow['names'])
        for name, digest in plan['source_sha256'].items():
            assert sha256(Path(name)) == digest, name
        sources.update({str(p): sha256(p) for p in [path, plan_path, control_path]})
        initial_path = Path(plan['initial_checkpoint'])
        assert sha256(initial_path) == plan['initial_checkpoint_sha256']
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        sources[str(initial_path)] = plan['initial_checkpoint_sha256']
        for mode, data, checkpoint in [('grouped', run, path.with_name('merged.ckpt')),
                                       ('flat', control, control_path.with_name('merged.ckpt'))]:
            key = f'{arm}_{mode}'
            assert sha256(checkpoint) == data['checkpoint_sha256']
            sources[str(checkpoint)] = data['checkpoint_sha256']
            runs[key] = data
            values[key] = np.array([data['average_validation'][t]['BAcc'] for t in tasks])
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            movement[key] = {}
            for group, group_audit in audit['groups'].items():
                energy = sum(float((state[n].double()-initial[n].double()).square().sum()) for n in group_audit['names'])
                movement[key][group] = {'displacement_norm': energy**.5,
                                        'displacement_rms': (energy/group_audit['parameters'])**.5}
            del state
        del initial
    comparisons = {f'{arm}_grouped_vs_flat': contrast(values[f'{arm}_grouped'], values[f'{arm}_flat'], tasks)
                   for arm in design['arms']}
    comparisons.update({f'merged_vs_base_{mode}': contrast(values[f'merged_{mode}'], values[f'base_{mode}'], tasks)
                        for mode in ['grouped', 'flat']})
    comparisons['initialization_by_group_ratio_interaction'] = contrast(
        values['merged_grouped']-values['merged_flat'], values['base_grouped']-values['base_flat'], tasks)
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    sources[str(joint_path)] = sha256(joint_path)
    joint_mean = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    merger_mean = float(np.mean([runs['merged_flat']['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    means = {key: float(v.mean()) for key, v in values.items()}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'sources': sources,
              'datasets': tasks, 'macro_bacc': means, 'contrasts': comparisons,
              'joint_macro_bacc': joint_mean, 'initial_merger_macro_bacc': merger_mean,
              'fraction_joint_gap_closed_from_merger': {mode: (means[f'merged_{mode}']-merger_mean)/(joint_mean-merger_mean) for mode in ['grouped', 'flat']},
              'calibration_macro_bacc': {key: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for key, run in runs.items()},
              'process_seconds': {key: run['seconds'] for key, run in runs.items()},
              'parameter_displacement_from_own_initialization': movement,
              'all_initial_metrics_and_data_checks_passed': True,
              'scope': 'Both fixed historical10:1 LR-ratio interventions versus existing1:1 controls on one exposed bank/order. Same full-pool schedule,1600updates and six-state mean. Training recipe control, not a novel method, full Joint recipe or independent confirmation. Task deletion is descriptive sensitivity, not a CI. Additional adaptation budgets match within contrasts; expert/coefficient costs remain different across initializations. Final parameter movement does not by itself identify functional mediation.'}
    output = root/'analysis/grouped_ce_control_summary.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.4]})
    x = np.arange(2)
    for offset, mode, label, color in [(-.18, 'flat', 'Tokenizer/rest 1:1', '#64748b'), (.18, 'grouped', 'Tokenizer/rest 10:1', '#2563eb')]:
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
        row = comparisons[f'{arm}_grouped_vs_flat']
        bx.barh(yy+offset, [row['per_task_gain_pp'][t] for t in tasks], height=.34, color=color,
                label=f'{arm}: {row["macro_gain_pp"]:+.3f} pp mean')
    bx.axvline(0, color='#64748b', lw=.8)
    bx.set_yticks(yy, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel('10:1 minus 1:1 BAcc (percentage points)')
    bx.set_title('B. All task effects', loc='left', weight='bold')
    bx.legend(fontsize=8)
    fig.suptitle('Historical tokenizer learning-rate ratio: fixed paired controls')
    fig.tight_layout(rect=(0, .04, 1, .96))
    figure = root/'analysis/figures/grouped_ce_controls'
    for suffix in ['.png', '.pdf']:
        fig.savefig(figure.with_suffix(suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'macro_bacc': means,
                      'contrasts': {k: {n: v for n, v in row.items() if n not in ['per_task_gain_pp', 'leave_one_task_out_pp']} for k, row in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
