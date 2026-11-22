"""Same primary comparisons; retain both failures and read the idle-GPU retry2."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_duration_ce_controls_v2 import verify_average
from report_grouped_ce_controls import contrast


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    plan_path = root/'inputs/row_projected_training_plan.json'
    plan = json.loads(plan_path.read_text())
    recovery_path = root/'inputs/row_projected_full_retry2_recovery.json'
    assert sha256(recovery_path) == '41c5b1916d9bcc1b1a8df933a4a09afb62e2f5227a7a6c416858dac57333dc0f'
    recovery = json.loads(recovery_path.read_text())
    assert recovery['original_aborted_attempt']['exit_code'] == -15
    assert recovery['retry1_failed_attempt']['exit_code'] == 1
    assert all(recovery[k]['optimizer_updates'] == 0 for k in ['original_aborted_attempt', 'retry1_failed_attempt'])
    retry_receipt_path = root/'seed0/row_projected_full_retry2_queue/run.json'
    retry_receipt = json.loads(retry_receipt_path.read_text())
    assert retry_receipt['status'] == 'complete' and len(retry_receipt['jobs']) == 1
    assert retry_receipt['jobs'][0]['exit_code'] == 0
    gate_path = root/'analysis/row_projected_engineering_verification.json'
    gate = json.loads(gate_path.read_text())
    assert gate['full_ce_reference_metrics_histories_dynamics_exact']
    for p,h in plan['source_sha256'].items(): assert sha256(Path(p)) == h,p
    assert sha256(Path(plan['projection_design'])) == plan['projection_design_sha256']
    assert sha256(Path(plan['basis_manifest'])) == plan['basis_manifest_sha256']
    paths = [plan_path, gate_path, recovery_path, retry_receipt_path, Path('report_row_projected_training.py'), Path('report_row_projected_training_v2.py'), Path(plan['projection_design']), Path(plan['basis_manifest']),
             Path(__file__), Path('report_duration_ce_controls_v2.py'), Path('report_grouped_ce_controls.py')]
    sources = {str(p):sha256(p) for p in paths}
    runs, wrappers, values, checks = {}, {}, {}, {}
    for kind in ['full', 'row', 'shuffled']:
        folder = root/('seed0/row_projected_training_full_retry2' if kind == 'full' else f'seed0/row_projected_training_{kind}')
        run_path, wrapper_path = folder/'run.json', folder/'wrapper.json'
        run, wrapper = [json.loads(p.read_text()) for p in [run_path, wrapper_path]]
        assert run['plan'] == plan and run['plan_sha256'] == wrapper['plan_sha256'] == sha256(plan_path)
        assert wrapper['status'] == 'complete' and wrapper['condition'] == kind
        assert wrapper['run_sha256'] == sha256(run_path)
        assert wrapper['wrapper_sha256'] == sha256(Path('train_row_projected_adam.py'))
        assert not wrapper['engineering_smoke'] and not wrapper['deterministic_algorithms']
        assert run['updates'] == wrapper['updates'] == 1600 and run['training_examples'] == 204800
        assert run['schedule_sha256'] == plan['schedule_sha256']
        assert run['condition'] == 'balanced_ce' and run['scales_sha256'] is None
        assert all(v['auxiliary_scale'] == 0 for v in run['distillation_audits'].values())
        assert run['fixed_tensors_exact'] and run['trainable_parameters'] == 25751552
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert not any(run['numerical_flags'].values())
        checks[kind] = verify_average({**run, 'checkpoint':str(folder/'merged.ckpt')}, plan['average_steps'])
        runs[kind], wrappers[kind] = run, wrapper
        sources.update({str(p):sha256(p) for p in [run_path, wrapper_path, folder/'merged.ckpt']})
        values[kind] = np.array([run['average_validation'][t]['BAcc'] for t in run['datasets']])
    base, tasks = runs['full'], runs['full']['datasets']
    for kind, run in runs.items():
        assert run['datasets'] == tasks and run['dataset_audits'] == base['dataset_audits']
        assert run['initial_validation'] == base['initial_validation']
    comparisons = {f'{a}_vs_{b}':contrast(values[a],values[b],tasks)
                   for a,b in [('row','full'),('shuffled','full'),('row','shuffled')]}
    means = {k:float(v.mean()) for k,v in values.items()}
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    joint_mean = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    sources[str(joint_path)] = sha256(joint_path)
    initial = float(np.mean([base['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    historical_path = root/'seed0/full_control_coverage_full_balanced_ce/run.json'
    historical = json.loads(historical_path.read_text()); sources[str(historical_path)] = sha256(historical_path)
    old = np.array([historical['average_validation'][t]['BAcc'] for t in tasks])
    result = {'unix':time.time(),'code_sha256':sha256(Path(__file__)),'sources':sources,
        'datasets':tasks,'macro_bacc':means,'primary_contrasts':comparisons,
        'initial_macro_bacc':initial,'joint_macro_bacc':joint_mean,
        'remaining_joint_gap_pp':{k:100*(joint_mean-v) for k,v in means.items()},
        'fraction_initial_joint_gap_closed':{k:(v-initial)/(joint_mean-initial) for k,v in means.items()},
        'fresh_full_minus_historical_full':contrast(values['full'],old,tasks),
        'average_reconstruction_checks':checks,
        'projection_runtime_checks':{k:{x:v for x,v in w.items() if x in [
            'coordinate_count','average_coordinate_audit','step_norm_ratio_range','restoration_scale_range','zero_projection_steps']}
            for k,w in wrappers.items()},
        'process_seconds':{k:r['seconds'] for k,r in runs.items()},
        'aborted_attempts':{k:recovery[k] for k in ['original_aborted_attempt','retry1_failed_attempt']},
        'aborted_attempt_seconds':sum(recovery[k]['elapsed_seconds'] for k in ['original_aborted_attempt','retry1_failed_attempt']),
        'aborted_optimizer_updates':0,
        'aborted_retry1_training_forward_examples':128,
        'aborted_retry1_backward_attempts':1,
        'scientific_optimizer_updates':sum(r['updates'] for r in runs.values()),
        'engineering_optimizer_updates':gate['engineering_optimizer_updates'],
        'scope':plan['scope']+' Full and constrained arms all maintain full native-coordinate Adam states. Constraint rank is not the number of autograd parameters. Norm restoration refers to each arms own current proposal. Task-deletion ranges are sensitivity, not confidence intervals. Historical full-CE difference is descriptive native variation, not a noise correction.'}
    output = root/'analysis/row_projected_training_summary_v3.json'
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1.5]})
    order=['full','row','shuffled'];labels=['Full CE','Row space','Shuffled space']
    ax.bar(np.arange(3),[100*means[k] for k in order],color=['#64748b','#0072B2','#D55E00'])
    for i,k in enumerate(order):ax.text(i,100*means[k]+.08,f'{100*means[k]:.3f}',ha='center',fontsize=9)
    ax.axhline(100*joint_mean,ls='--',color='#333333',label='Historical Joint')
    ax.axhline(100*initial,ls=':',color='#777777',label='Initial merger')
    ax.set_ylim(100*min([initial,*means.values()])-1,100*max([joint_mean,*means.values()])+1)
    ax.set_xticks(np.arange(3),labels);ax.set_ylabel('Validation macro BAcc (%)');ax.legend(fontsize=8)
    ax.set_title('A. Fixed six-checkpoint means',loc='left')
    for i,(name,color) in enumerate(zip(['row_vs_full','shuffled_vs_full','row_vs_shuffled'],['#0072B2','#D55E00','#009E73'])):
        row=comparisons[name]
        bx.scatter([row['per_task_gain_pp'][t] for t in tasks],np.arange(16)+(i-1)*.2,s=22,color=color,
                   label=f'{name.replace("_"," ")}: {row["macro_gain_pp"]:+.3f} pp')
    bx.axvline(0,color='#777777',lw=.8);bx.set_yticks(np.arange(16),tasks,fontsize=8);bx.invert_yaxis()
    bx.set_xlabel('BAcc difference (pp)');bx.set_title('B. All16 task contrasts',loc='left');bx.legend(fontsize=8)
    fig.text(.5,.012,'One expert bank; exposed validation; shared CE/data/budget. Norm-restored fixed-space Adam proposals; ordinary checkpoint exports.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    for suffix in ['png','pdf']:fig.savefig(root/f'analysis/figures/row_projected_training_v3.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output':str(output),'sha256':sha256(output),'macro_bacc':means,
                      'primary_effects_pp':{k:v['macro_gain_pp'] for k,v in comparisons.items()}},indent=2))


if __name__=='__main__':main()
