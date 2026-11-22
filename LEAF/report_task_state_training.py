"""Fixed global-versus-task Adam-state comparison and native export verification."""
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from d_downstream_qformer_ce import sha256
from report_query_rotation_merges import contrast


def main():
    torch.set_num_threads(4)
    root=Path('runs/paper_merge_v2');output=root/'analysis/task_state_training_summary.json'
    assert not output.exists()
    plan_path=root/'inputs/task_state_training_plan.json';plan=json.loads(plan_path.read_text())
    assert sha256(Path(__file__))==plan['reporter_sha256']
    tasks=plan['task_state_datasets'];source=json.loads((Path(plan['source_fit'])/'run.json').read_text())
    active=set(source['groups']);runs={};wrappers={};exports={};sources={}
    for arm in ['global','task']:
        folder=root/f'seed0/task_state_training_{arm}';path=folder/'run.json'
        run=json.loads(path.read_text());wrapper=json.loads((folder/'wrapper.json').read_text())
        assert wrapper['status']=='complete' and wrapper['condition']==arm
        assert wrapper['run_sha256']==sha256(path) and run['plan_sha256']==sha256(plan_path)
        assert run['updates']==plan['steps'] and run['average_steps']==plan['average_steps']
        assert run['fixed_tensors_exact'] and not run['test_data_read']
        assert sha256(folder/'merged.ckpt')==run['checkpoint_sha256']
        states=[]
        for row in run['average_snapshots']:
            assert sha256(Path(row['path']))==row['sha256']
            states.append(torch.load(row['path'],map_location='cpu',weights_only=True))
        merged=torch.load(folder/'merged.ckpt',map_location='cpu',weights_only=True)
        for name,value in merged.items():
            expected=torch.stack([s[name].double() for s in states]).mean(0).float() if name in active else states[0][name]
            assert torch.equal(value,expected),(arm,name)
        assert all(c['max_abs_logit_error']<=1e-5 and all(r['margin_within_twice_error'] for r in c['changed_rows']) for c in run['export_checks'].values())
        assert all(v['auxiliary_scale']==0 for v in run['distillation_audits'].values())
        exports[arm]={'tensor_count':len(merged),'all_saved_tensors_reconstruct_exactly':True,'native_export_checks_pass':True}
        runs[arm]=run;wrappers[arm]=wrapper;sources[str(path)]=sha256(path)
        sources[str(folder/'wrapper.json')]=sha256(folder/'wrapper.json')
        del states,merged
    a,b=runs['global'],runs['task']
    assert a['dataset_audits']==b['dataset_audits'] and a['schedule_sha256']==b['schedule_sha256']
    assert a['distillation_audits']==b['distillation_audits']
    assert all(a['initial_validation']['validation'][t]['BAcc']==b['initial_validation']['validation'][t]['BAcc'] for t in tasks)
    initial_differences=[{'task':t,'metric':k,'global':v,'task_state':b['initial_validation']['validation'][t][k]}
                         for t,m in a['initial_validation']['validation'].items() for k,v in m.items()
                         if v!=b['initial_validation']['validation'][t][k]]
    split=wrappers['task']['optimizer_state_audit'];shared=wrappers['global']['optimizer_state_audit']
    assert split['all_optimizers_share_the_same_parameters'] and split['moment_storages_disjoint']
    assert all(s['visits']==100 and s['bias_correction_counter_values']==[100] for s in split['tasks'].values())
    assert shared['bias_correction_counter_values']==[1600]
    assert split['moment_tensor_elements']==16*shared['moment_tensor_elements']
    prior_path=root/'analysis/contrast_head_training_summary.json';prior=json.loads(prior_path.read_text())
    recent_path=root/'seed0/contrast_head_training_full/run.json';recent=json.loads(recent_path.read_text())
    sources[str(prior_path)]=sha256(prior_path);sources[str(recent_path)]=sha256(recent_path)
    metrics={k:r['average_validation'] for k,r in runs.items()}
    macro={k:float(np.mean([m[t]['BAcc'] for t in tasks])) for k,m in metrics.items()}
    result={'unix':time.time(),'plan_sha256':sha256(plan_path),'reporter_sha256':sha256(Path(__file__)),
            'sources':sources,'datasets':tasks,'macro_bacc':macro,
            'primary_contrast':contrast(metrics['task'],metrics['global'],tasks),
            'fresh_global_minus_recent_global_context':contrast(metrics['global'],recent['average_validation'],tasks),
            'initial_macro_bacc':prior['initial_macro_bacc'],'joint_macro_bacc':prior['joint_macro_bacc'],
            'remaining_joint_gap_pp':{k:100*(prior['joint_macro_bacc']-v) for k,v in macro.items()},
            'fraction_initial_joint_gap_closed':{k:(v-prior['initial_macro_bacc'])/(prior['joint_macro_bacc']-prior['initial_macro_bacc']) for k,v in macro.items()},
            'initial_metric_differences':initial_differences,'initial_bacc_exact':True,'data_and_schedule_exact':True,
            'average_reconstruction_checks':exports,'wrappers':wrappers,
            'scientific_optimizer_updates':3200,'engineering_native_leaf_updates':0,
            'engineering_synthetic_optimizer_updates':88,'test_data_read':False,
            'scope':plan['scope'],'conditional_controls':plan['conditional_controls']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1,1.7]})
    names=['M0','Global Adam','Task-state Adam','Joint reference']
    values=[100*prior['initial_macro_bacc'],100*macro['global'],100*macro['task'],100*prior['joint_macro_bacc']]
    axes[0].barh(names,values,color=['#777777','#4e8a67','#4a80a6','#444444']);axes[0].invert_yaxis()
    axes[0].set_xlim(min(60,min(values)-1),max(70,max(values)+1));axes[0].set_xlabel('Macro balanced accuracy (%)')
    for i,v in enumerate(values):axes[0].text(v+.08,i,f'{v:.3f}',va='center',fontsize=9)
    delta=result['primary_contrast']['per_task_gain_pp'];d=[delta[t] for t in tasks];y=np.arange(len(tasks))
    axes[1].barh(y,d,color=['#4a80a6' if v>=0 else '#b5504d' for v in d]);axes[1].axvline(0,color='grey',lw=.8)
    axes[1].set_yticks(y,[t.replace('MI_','').replace('EMO_','') for t in tasks],fontsize=8);axes[1].invert_yaxis()
    axes[1].set_xlabel('Task-state minus global Adam (percentage points)')
    fig.suptitle('Shared-model repair: task-specific Adam histories')
    fig.text(.5,.015,'One bank; development validation; same 1600 batches; fixed six-checkpoint mean; 16x optimizer moment storage.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:fig.savefig(root/f'analysis/figures/task_state_training.{ext}',dpi=180)
    print(json.dumps({'macro_bacc':macro,'primary':result['primary_contrast'],'remaining_joint_gap_pp':result['remaining_joint_gap_pp']},indent=2))


if __name__=='__main__':main()
