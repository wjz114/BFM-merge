"""Matched target-coordinate screen, with ordinary-export and fixed-mean checks."""
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
    root=Path('runs/paper_merge_v2');output=root/'analysis/aligned_query_training_summary.json';assert not output.exists()
    torch.set_num_threads(4)
    design_path=root/'inputs/aligned_query_training_design.json';design=json.loads(design_path.read_text());tasks=design['datasets']
    source=json.loads(Path(design['source_fit']).read_text());active=set(source['groups'])
    runs={};exports={};sources={};wrappers={}
    for arm in design['conditions']:
        folder=root/f'seed0/aligned_query_training_{arm}_v3';path=folder/'run.json';run=json.loads(path.read_text())
        wrapper=json.loads((folder/'wrapper.json').read_text());assert wrapper['status']=='complete' and wrapper['run_sha256']==sha256(path)
        assert run['updates']==1600 and run['average_steps']==design['average_steps']
        assert not run['test_data_read'] and run['fixed_tensors_exact']
        assert sha256(folder/'merged.ckpt')==run['checkpoint_sha256']
        states=[]
        for row in run['average_snapshots']:
            assert sha256(Path(row['path']))==row['sha256'];states.append(torch.load(row['path'],map_location='cpu',weights_only=True))
        merged=torch.load(folder/'merged.ckpt',map_location='cpu',weights_only=True)
        for n,v in merged.items():
            expected=torch.stack([state[n].double() for state in states]).mean(0).float() if n in active else states[0][n]
            assert torch.equal(v,expected),(arm,n)
        assert all(c['max_abs_logit_error']<=1e-5 and all(r['margin_within_twice_error'] for r in c['changed_rows']) for c in run['export_checks'].values())
        exports[arm]={'tensor_count':len(merged),'all_saved_tensors_reconstruct_exactly':True,'native_export_checks_pass':True}
        runs[arm]=run;wrappers[arm]=wrapper;sources[str(path)]=sha256(path)
        del states,merged
    a,b=runs['original'],runs['aligned']
    assert a['schedule_sha256']==b['schedule_sha256'] and a['dataset_audits']==b['dataset_audits']
    assert all(a['initial_validation']['validation'][t]['BAcc']==b['initial_validation']['validation'][t]['BAcc'] for t in tasks)
    assert all(a['distillation_audits'][t]['auxiliary_scale']==b['distillation_audits'][t]['auxiliary_scale'] for t in tasks)
    initial_differences=[{'task':t,'metric':k,'original':v,'aligned':b['initial_validation']['validation'][t][k]}
                         for t,m in a['initial_validation']['validation'].items() for k,v in m.items()
                         if v!=b['initial_validation']['validation'][t][k]]
    full_path=root/'seed0/contrast_head_training_full/run.json';full=json.loads(full_path.read_text());sources[str(full_path)]=sha256(full_path)
    old_path=root/'seed0/prehead_distillation_ce_prehead_paired/run.json';old=json.loads(old_path.read_text());sources[str(old_path)]=sha256(old_path)
    prior_path=root/'analysis/contrast_head_training_summary.json';prior=json.loads(prior_path.read_text());sources[str(prior_path)]=sha256(prior_path)
    metrics={arm:run['average_validation'] for arm,run in runs.items()}
    macro={arm:float(np.mean([m[t]['BAcc'] for t in tasks])) for arm,m in metrics.items()}
    preparation=json.loads(Path(design['preparation_audit']).read_text())
    result={'unix':time.time(),'design_sha256':sha256(design_path),'reporter_sha256':sha256(Path(__file__)),
            'sources':sources,'datasets':tasks,'macro_bacc':macro,
            'primary_contrast':contrast(metrics['aligned'],metrics['original'],tasks),
            'versus_recent_full_ce_context':{arm:contrast(m,full['average_validation'],tasks) for arm,m in metrics.items()},
            'fresh_original_minus_historical_original':contrast(metrics['original'],old['average_validation'],tasks),
            'joint_macro_bacc':prior['joint_macro_bacc'],'initial_macro_bacc':prior['initial_macro_bacc'],
            'remaining_joint_gap_pp':{k:100*(prior['joint_macro_bacc']-v) for k,v in macro.items()},
            'fraction_initial_joint_gap_closed':{k:(v-prior['initial_macro_bacc'])/(prior['joint_macro_bacc']-prior['initial_macro_bacc']) for k,v in macro.items()},
            'initial_metric_differences':initial_differences,'initial_bacc_exact':True,'dataset_audits_exact':True,
            'auxiliary_scales_exact':True,'average_reconstruction_checks':exports,'wrappers':wrappers,
            'aligned_aux_gradient_ratio_range':[min(v['aligned_to_original_initial_aux_gradient_rms'] for v in preparation['records'].values()),max(v['aligned_to_original_initial_aux_gradient_rms'] for v in preparation['records'].values())],
            'scientific_optimizer_updates':3200,'failed_scientific_prefix_updates_range':[160,199],'engineering_optimizer_updates':16,'test_data_read':False,
            'scope':design['scientific_scope'],'conditional_controls':design['conditional_controls']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1,1.7]})
    names=['M0','Recent full CE','Original paired','Aligned paired','Joint reference']
    values=[100*prior['initial_macro_bacc'],100*full['average_validation_macro_bacc'],100*macro['original'],100*macro['aligned'],100*prior['joint_macro_bacc']]
    axes[0].barh(names,values,color=['#777777','#4e8a67','#a68b50','#4a80a6','#444444']);axes[0].invert_yaxis()
    axes[0].set_xlim(60,70);axes[0].set_xlabel('Macro balanced accuracy (%)')
    for i,v in enumerate(values):axes[0].text(v+.08,i,f'{v:.3f}',va='center',fontsize=9)
    delta=result['primary_contrast']['per_task_gain_pp'];y=np.arange(len(tasks));d=[delta[t] for t in tasks]
    axes[1].barh(y,d,color=['#4a80a6' if v>=0 else '#b5504d' for v in d]);axes[1].axvline(0,color='grey',lw=.8)
    axes[1].set_yticks(y,[t.replace('MI_','').replace('EMO_','') for t in tasks],fontsize=8);axes[1].invert_yaxis()
    axes[1].set_xlabel('Aligned minus original paired targets (percentage points)')
    fig.suptitle('Teacher feature coordinates: matched 1600-step distillation screen')
    fig.text(.5,.015,'One bank; existing development validation; original auxiliary weights unchanged; fixed six-checkpoint mean.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:fig.savefig(root/f'analysis/figures/aligned_query_training.{ext}',dpi=180)
    print(json.dumps({'macro_bacc':macro,'primary':result['primary_contrast'],'remaining_joint_gap_pp':result['remaining_joint_gap_pp']},indent=2))


if __name__=='__main__':main()
