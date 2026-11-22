"""Compare both fixed two-stage direct-base controls after their workers finish."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/staged_base_control_design.json'
    design = json.loads(design_path.read_text())
    tasks = design['datasets']
    reference = json.loads(Path(design['base_reference']).read_text())
    assert sha256(Path(design['base_reference'])) == design['base_reference_sha256']
    sources = {str(design_path): sha256(design_path)}
    runs, stages, controllers = {}, {}, {}
    for key, record in design['matched_controls'].items():
        path = Path(record['path'])
        assert sha256(path) == record['sha256']
        sources[str(path)] = record['sha256']
        runs[key] = json.loads(path.read_text())
    for rate, arm in design['arms'].items():
        controller_path = Path(arm['controller_outdir'])/'run.json'
        controller = json.loads(controller_path.read_text())
        assert controller['design_sha256'] == sha256(design_path)
        assert controller['updates'] == 3200 and controller['training_exposures'] == 409600
        assert controller['stage_transition_bacc_exact']
        sources[str(controller_path)] = sha256(controller_path)
        controllers[rate] = controller
        for stage in [1, 2]:
            path = Path(arm[f'stage{stage}_outdir'])/'run.json'
            assert str(path) == controller[f'stage{stage}']['path']
            assert sha256(path) == controller[f'stage{stage}']['sha256']
            sources[str(path)] = sha256(path)
            run = json.loads(path.read_text())
            plan_path = Path(arm[f'stage{stage}_plan'])
            plan = json.loads(plan_path.read_text())
            assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
            assert plan['design_sha256'] == sha256(design_path)
            assert plan['lr'] == arm['lr'] and plan['seed'] == 0
            assert run['datasets'] == tasks and run['condition'] == 'balanced_ce'
            assert run['updates'] == 1600 and run['training_examples'] == 204800
            assert run['average_steps'] == design['average_steps']
            assert run['training_pool'] == ('original512' if stage == 1 else 'full_training')
            assert run['trainable_parameters'] == 25751552 and run['fixed_tensors_exact']
            assert not run['engineering_smoke'] and not run['test_data_read'] and not run['test_inference_performed']
            assert run['checkpoint_sha256'] == sha256(path.with_name('merged.ckpt'))
            assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
            for name, expected in plan['source_sha256'].items():
                assert sha256(Path(name)) == expected, name
            assert sha256(Path(plan['schedule'])) == plan['schedule_sha256'] == run['schedule_sha256']
            assert sha256(Path(plan['initial_checkpoint'])) == plan['initial_checkpoint_sha256']
            stages[f'{rate}_stage{stage}'] = run
        first, second = stages[f'{rate}_stage1'], stages[f'{rate}_stage2']
        assert first['plan']['initial_checkpoint_sha256'] == design['base_checkpoint_sha256']
        assert first['initial_reference_used'] == design['base_reference']
        assert all(first['initial_validation']['validation'][t]['BAcc'] == reference['folded_val'][t]['BAcc'] for t in tasks)
        assert second['plan']['initial_checkpoint_sha256'] == first['checkpoint_sha256']
        assert second['plan']['previous_stage_run_sha256'] == sources[str(Path(arm['stage1_outdir'])/'run.json')]
        assert all(second['initial_validation']['validation'][t]['BAcc'] == first['average_validation'][t]['BAcc'] for t in tasks)
        assert second['dataset_audits'] == runs[f'single_base_{rate}']['dataset_audits']
        assert second['schedule_sha256'] == runs[f'single_base_{rate}']['schedule_sha256'] == runs[f'merged_{rate}']['schedule_sha256']
        runs[f'staged_base_{rate}'] = second
    assert stages['lr1e5_stage1']['initial_validation'] == stages['lr1e4_stage1']['initial_validation']
    assert stages['lr1e5_stage1']['dataset_audits'] == stages['lr1e4_stage1']['dataset_audits']
    initial_path = root/'seed0/original_fp32_tensor_balanced_avg/run.json'
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    pipeline_path = root/'analysis/merge_pipeline_budget.json'
    initial, joint, pipeline = [json.loads(p.read_text()) for p in [initial_path, joint_path, pipeline_path]]
    sources.update({str(p): sha256(p) for p in [initial_path, joint_path, pipeline_path]})
    values = {key: np.array([run['average_validation'][t]['BAcc'] for t in tasks]) for key, run in runs.items()}
    values.update({f'staged_stage1_{rate}': np.array([stages[f'{rate}_stage1']['average_validation'][t]['BAcc'] for t in tasks]) for rate in design['arms']})
    values['initial_merger'] = np.array([initial['folded_val'][t]['BAcc'] for t in tasks])
    values['released_base'] = np.array([reference['folded_val'][t]['BAcc'] for t in tasks])
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    contrasts = {}
    pairs = []
    for rate in design['arms']:
        pairs += [(f'merged_{rate}', f'staged_base_{rate}'),
                  (f'staged_base_{rate}', f'single_base_{rate}'),
                  (f'staged_base_{rate}', f'staged_stage1_{rate}'),
                  ('initial_merger', f'staged_stage1_{rate}'),
                  (f'staged_base_{rate}', 'initial_merger')]
    pairs.append(('staged_base_lr1e4', 'staged_base_lr1e5'))
    for treatment, control in pairs:
        delta = (values[treatment]-values[control])*100
        loo = {t: float(np.delete(delta, i).mean()) for i, t in enumerate(tasks)}
        contrasts[treatment+'_vs_'+control] = {'macro_gain_pp': float(delta.mean()),
            'per_task_gain_pp': dict(zip(tasks, delta.tolist())), 'improved': int((delta > 0).sum()),
            'declined': int((delta < 0).sum()), 'ties': int((delta == 0).sum()),
            'leave_one_task_out_pp': loo, 'task_deletion_range_pp': [min(loo.values()), max(loo.values())]}
    original_gap = (values['Joint']-values['initial_merger']).mean()
    gap_closed = {k: float((v-values['initial_merger']).mean()/original_gap*100) for k, v in values.items()
                  if k.startswith(('staged_base_', 'merged_', 'single_base_'))}
    cost = {}
    for rate in design['arms']:
        first, second = stages[f'{rate}_stage1'], stages[f'{rate}_stage2']
        cost[rate] = {'staged_base_total_process_seconds': first['seconds']['total']+second['seconds']['total'],
                      'staged_base_stages': {str(i): stages[f'{rate}_stage{i}']['seconds'] for i in [1, 2]},
                      'merger_fit_mean_repair_process_seconds': initial['seconds']['total']+runs[f'merged_{rate}']['seconds']['total'],
                      'single_base_process_seconds': runs[f'single_base_{rate}']['seconds'],
                      'staged_base_updates': 3200, 'merger_fit_repair_updates': 3200, 'single_base_updates': 1600,
                      'expert_bank_recorded_updates_additional_to_merging': pipeline['expert_bank_total_recorded_updates']}
    figure_dir = root/'analysis/figures'
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 7), gridspec_kw={'width_ratios': [1, 1.3]})
    x = np.arange(2)
    labels = ['Adam 1e-5', 'Adam 1e-4']
    for offset, prefix, label, color in [(-.24, 'merged_', 'Merger + repair (3200 updates)', '#2563eb'),
                                       (0, 'staged_base_', 'Staged direct base (3200 updates)', '#d97706'),
                                       (.24, 'single_base_', 'Direct base, full pool (1600 updates)', '#64748b')]:
        y = [values[prefix+rate].mean()*100 for rate in ['lr1e5', 'lr1e4']]
        ax.bar(x+offset, y, width=.23, label=label, color=color)
        for px, py in zip(x+offset, y):
            ax.text(px, py+.07, f'{py:.2f}', ha='center', fontsize=8)
    ax.axhline(values['Joint'].mean()*100, color='#334155', ls='--', label='Joint reference')
    ax.axhline(values['initial_merger'].mean()*100, color='#94a3b8', ls=':', label='Initial merger')
    ymin = min(v.mean()*100 for k, v in values.items() if k.startswith(('merged_', 'staged_base_', 'single_base_')))-1
    ax.set_ylim(min(ymin, values['initial_merger'].mean()*100-.5), values['Joint'].mean()*100+.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Full16 validation macro BAcc (%)')
    ax.set_title('A. Both fixed learning rates retained', loc='left', weight='bold')
    ax.legend(loc='upper center', fontsize=8, bbox_to_anchor=(.5, -.08))
    yy = np.arange(16)
    for offset, rate, color in [(-.18, 'lr1e5', '#2563eb'), (.18, 'lr1e4', '#d97706')]:
        row = contrasts[f'merged_{rate}_vs_staged_base_{rate}']
        delta = [row['per_task_gain_pp'][t] for t in tasks]
        bx.barh(yy+offset, delta, height=.34, color=color, label=f'{rate}: macro {row["macro_gain_pp"]:+.3f} pp')
    bx.set_yticks(yy, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.axvline(0, color='#334155', lw=.8)
    bx.set_xlabel('Merger + repair minus staged direct base (pp)')
    bx.set_title('B. All16 paired effects', loc='left', weight='bold')
    bx.legend(loc='best', fontsize=8)
    for axis in [ax, bx]:
        axis.set_axisbelow(True)
        axis.grid(axis='y' if axis is ax else 'x', color='#e2e8f0', lw=.6)
    fig.suptitle('Direct-base control including the coefficient-fitting update and label budget', weight='bold', fontsize=12)
    fig.text(.5, .012, 'Single exposed-validation comparison. Expert training adds48000prior updates to the merger pipeline; wall time and FLOPs are not matched.', ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .09, 1, .95])
    for ext in ['png', 'pdf']:
        fig.savefig(figure_dir/f'staged_base_controls.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'datasets': tasks, 'sources': sources,
              'macro_bacc': {k: float(v.mean()) for k, v in values.items()}, 'contrasts': contrasts,
              'initial_merger_joint_gap_closed_percent': gap_closed,
              'calibration_macro_bacc': {k: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for k, run in {**runs, **stages}.items()},
              'cost': cost, 'checks': {'all_input_hashes_and_exports_passed': True, 'stage_transition_bacc_exact': True,
                                     'initial_base_metrics_exact_across_rates': True, 'both_prespecified_rates_retained': True},
              'figure_sha256': {ext: sha256(figure_dir/f'staged_base_controls.{ext}') for ext in ['png', 'pdf']},
              'scope': design['budget_scope']+' All performance results are exposed validation from a single bank/order; task-deletion ranges are sensitivity analyses, not confidence intervals. No independent confirmation or new algorithm follows from this control.'}
    with (root/'analysis/staged_base_control_summary.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'primary_contrasts': {rate: contrasts[f'merged_{rate}_vs_staged_base_{rate}']['macro_gain_pp'] for rate in design['arms']}}, indent=2))


if __name__ == '__main__':
    main()
