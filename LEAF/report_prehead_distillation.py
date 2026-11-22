"""Report all fixed prehead target controls against the fresh matched CE run."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_grouped_ce_controls import contrast
from report_duration_ce_controls_v2 import verify_average


def main():
    torch.set_num_threads(4)
    root=Path('runs/paper_merge_v2')
    design_path=root/'inputs/prehead_distillation_design.json'
    design=json.loads(design_path.read_text())
    plan_path=Path(design['training_plan'])
    plan=json.loads(plan_path.read_text())
    for name,digest in {**design['source_sha256'],**design['input_sha256']}.items():
        assert sha256(Path(name))==digest,name
    assert plan['design_sha256']==sha256(design_path)
    scales=json.loads(Path(design['scales']).read_text())
    targets=json.loads(Path(design['target_manifest']).read_text())
    assert sha256(Path(design['scales']))==plan['scales_sha256']
    assert sha256(Path(design['target_manifest']))==plan['target_cache_sha256']
    engineering=root/'analysis/prehead_engineering_verification.json'
    gate=json.loads(engineering.read_text())
    assert gate['all_zero_auxiliary_metrics_histories_data_gradients_exact']
    assert gate['paired_smoke_finite_active_gradients_and_native_export']
    amendment_path=Path(gate['reference_amendment']['path'])
    assert sha256(amendment_path)==gate['reference_amendment']['sha256']
    paths=[design_path,plan_path,Path(design['scales']),Path(design['target_manifest']),engineering,
           Path(__file__),Path('report_grouped_ce_controls.py'),Path('report_duration_ce_controls_v2.py')]
    sources={str(p):sha256(p) for p in [*paths,amendment_path]}
    tasks=design['datasets']
    runs,values,checks={},{},{}
    for condition in design['conditions']:
        path=Path(design['training_arms'][condition]['outdir'])/'run.json'
        run=json.loads(path.read_text())
        assert run['condition']==condition and run['datasets']==tasks
        assert run['plan']==plan and run['plan_sha256']==sha256(plan_path)
        assert run['code_sha256']==sha256(Path('train_prehead_distillation.py'))
        assert run['updates']==1600 and run['training_examples']==204800
        assert run['schedule_sha256']==plan['schedule_sha256']
        assert not run['engineering_smoke'] and not any(run['numerical_flags'].values())
        assert run['fixed_tensors_exact'] and run['trainable_parameters']==25751552
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert run['target_cache_sha256']==plan['target_cache_sha256']
        assert run['scales_sha256']==(None if condition=='balanced_ce' else plan['scales_sha256'])
        for task in tasks:
            audit=run['distillation_audits'][task]
            assert audit['target_sha256']==targets['records'][task]['target_sha256']
            assert audit['auxiliary_scale']==(0. if condition=='balanced_ce' else scales['per_task'][task]['scale'])
            assert audit['shuffle']['fitting_target_multiset_exact']
            assert audit['shuffle']['one_fixed_target_per_unique_input']
        product={**run,'checkpoint':str(path.with_name('merged.ckpt'))}
        checks[condition]=verify_average(product,design['average_steps'])
        runs[condition]=run
        values[condition]=np.asarray([run['average_validation'][t]['BAcc'] for t in tasks])
        sources[str(path)]=sha256(path)
        sources[str(path.with_name('merged.ckpt'))]=run['checkpoint_sha256']
    base=runs['balanced_ce']
    for run in runs.values():
        assert run['dataset_audits']==base['dataset_audits']
        assert run['initial_validation']==base['initial_validation']
    primary={f'paired_vs_{name}':contrast(values['ce_prehead_paired'],values[condition],tasks)
             for name,condition in [('ce','balanced_ce'),('class_mean','ce_prehead_mean'),('class_shuffle','ce_prehead_shuffled')]}
    secondary={f'{name}_vs_ce':contrast(values[condition],values['balanced_ce'],tasks)
               for name,condition in [('class_mean','ce_prehead_mean'),('class_shuffle','ce_prehead_shuffled')]}
    historical={}
    for key,folder in [('ce','full_control_coverage_full_balanced_ce'),
                       ('kl','full_distillation_ce_teacher_kl'),('final_feature','full_distillation_ce_full_feature')]:
        path=root/f'seed0/{folder}/run.json'
        run=json.loads(path.read_text());sources[str(path)]=sha256(path)
        old=np.asarray([run['average_validation'][t]['BAcc'] for t in tasks])
        historical[key]={'macro_bacc':float(old.mean()),
            'paired_prehead_minus_reference':contrast(values['ce_prehead_paired'],old,tasks)}
        if key=='ce':historical[key]['fresh_ce_minus_historical_ce']=contrast(values['balanced_ce'],old,tasks)
    joint_path=root/'analysis/bank_joint_fp32_validation/run.json'
    joint=json.loads(joint_path.read_text());sources[str(joint_path)]=sha256(joint_path)
    joint_mean=float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    initial=float(np.mean([base['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    means={k:float(v.mean()) for k,v in values.items()}
    result={'unix':time.time(),'code_sha256':sha256(Path(__file__)),'sources':sources,'datasets':tasks,
        'macro_bacc':means,'primary_contrasts':primary,'secondary_contrasts':secondary,
        'historical_references':historical,'initial_merger_macro_bacc':initial,'joint_macro_bacc':joint_mean,
        'fraction_initial_joint_gap_closed':{k:(v-initial)/(joint_mean-initial) for k,v in means.items()},
        'calibration_macro_bacc':{k:float(np.mean([r['average_calibration'][t]['BAcc'] for t in tasks])) for k,r in runs.items()},
        'initial_loss_scale_and_gradient_geometry':scales['per_task'],
        'average_reconstruction_checks':checks,'process_seconds':{k:r['seconds'] for k,r in runs.items()},
        'original_planned_budget':design['budget'],
        'engineering_optimizer_updates_actual':gate['engineering_optimizer_updates'],
        'scale_calibration_seconds':scales['seconds'],
        'shuffle_unchanged_exposures':{t:targets['records'][t]['shuffle']['unchanged_fitting_exposures'] for t in tasks},
        'scope':plan['scope']+' Primary comparisons use a fresh same-worker CE control. Mean/shuffle use the paired-target alpha without rematching. Historical768-feature/KL comparisons change the gradient interface and are descriptive, not isolated layer-location effects. Task deletion is sensitivity, not a confidence interval. Paired beating shuffle without beating CE does not establish useful knowledge transfer.'}
    output=root/'analysis/prehead_distillation_summary.json'
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    labels={'balanced_ce':'CE','ce_prehead_paired':'Paired feature','ce_prehead_mean':'Class mean','ce_prehead_shuffled':'Class shuffle'}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1,1.6]})
    order=design['conditions']
    colors=['#64748b','#0072B2','#D55E00','#009E73']
    ax.bar(np.arange(4),[100*means[k] for k in order],color=colors)
    for i,key in enumerate(order):ax.text(i,100*means[key]+.08,f'{100*means[key]:.3f}',ha='center',fontsize=9)
    ax.axhline(100*joint_mean,ls='--',color='#333333',label='Historical Joint')
    ax.axhline(100*initial,ls=':',color='#888888',label='Initial merger')
    ax.set_xticks(np.arange(4),[labels[k] for k in order],rotation=20,ha='right')
    ax.set_ylim(100*min([initial,*means.values()])-1,100*max([joint_mean,*means.values()])+1)
    ax.set_ylabel('Full16 validation macro BAcc (%)');ax.legend(fontsize=9)
    ax.set_title('A. Prespecified six-checkpoint means',loc='left')
    y=np.arange(len(tasks))
    for offset,(key,color) in enumerate(zip(primary,['#64748b','#D55E00','#009E73'])):
        r=primary[key]
        bx.scatter([r['per_task_gain_pp'][t] for t in tasks],y+(offset-1)*.2,
                   s=23,color=color,label=f'{key.replace("_"," ")}: {r["macro_gain_pp"]:+.3f} pp')
    bx.axvline(0,color='#777777',lw=.8)
    bx.set_yticks(y,tasks,fontsize=9);bx.invert_yaxis()
    bx.set_xlabel('Paired feature minus comparator BAcc (percentage points)')
    bx.set_title('B. All task effects',loc='left');bx.legend(fontsize=8)
    fig.suptitle('Native prehead distillation: fixed target interventions')
    fig.text(.5,.015,'One expert bank; exposed validation; common initial feature-loss scale. No independent-confirmation or novelty claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ['png','pdf']:fig.savefig(root/f'analysis/figures/prehead_target_controls.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output':str(output),'sha256':sha256(output),'macro_bacc':means,
        'primary_pp':{k:v['macro_gain_pp'] for k,v in primary.items()}},indent=2))


if __name__=='__main__':main()
