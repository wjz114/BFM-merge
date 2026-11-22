"""Report every fixed query-coordinate counterfactual on existing development data."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def contrast(after,before,tasks):
    delta=np.asarray([100*(after[t]['BAcc']-before[t]['BAcc']) for t in tasks])
    loo=[float(np.delete(delta,i).mean()) for i in range(len(tasks))]
    return {'macro_gain_pp':float(delta.mean()),'per_task_gain_pp':dict(zip(tasks,delta.tolist())),
            'improved':int((delta>0).sum()),'declined':int((delta<0).sum()),'ties':int((delta==0).sum()),
            'task_deletion_range_pp':[min(loo),max(loo)],'task_deletion_is_confidence_interval':False}


def main():
    root=Path('runs/paper_merge_v2')
    path=root/'seed0/query_rotation_merges_v3/run.json'
    output=root/'analysis/query_rotation_merges_summary.json'
    assert not output.exists()
    run=json.loads(path.read_text());plan=run['plan'];tasks=plan['datasets']
    assert run['status']=='complete' and run['new_forward_examples']==plan['forward_budget']
    initial_path=Path(plan['initial_run']);initial=json.loads(initial_path.read_text())
    full_path=root/'seed0/contrast_head_training_full/run.json';full=json.loads(full_path.read_text())
    prior_path=root/'analysis/contrast_head_training_summary.json';prior=json.loads(prior_path.read_text())
    assert all(v['state_reload_exact'] and v['non_qformer_tensors_exact_initial'] for v in run['checkpoints'].values())
    for v in run['checkpoints'].values():assert sha256(Path(v['path']))==v['sha256']
    metrics={'original':initial['folded_val'],'full_ce':full['average_validation'],**run['validation']}
    macro={k:float(np.mean([v[t]['BAcc'] for t in tasks])) for k,v in metrics.items()}
    contrasts={}
    for kind in plan['conditions']:
        for baseline in ['original','full_ce']:
            contrasts[kind+'_minus_'+baseline]=contrast(metrics[kind],metrics[baseline],tasks)
    for kind in ['response','parameter']:
        contrasts[kind+'_minus_canonical']=contrast(metrics[kind],metrics['canonical'],tasks)
    checks=[v for bytask in run['source_equivalence'].values() for v in bytask.values()]
    summary={'unix':time.time(),'sources':{str(p):sha256(p) for p in [path,initial_path,full_path,prior_path]},
             'reporter_sha256':sha256(Path(__file__)),'datasets':tasks,'macro_bacc':macro,'contrasts':contrasts,
             'joint_macro_bacc':prior['joint_macro_bacc'],
             'joint_gap_pp':{k:100*(prior['joint_macro_bacc']-v) for k,v in macro.items()},
             'source_equivalence_max_logit_error':max(v['max_abs_logit_error'] for v in checks),
             'source_equivalence_changed_predictions':sum(v['changed_predictions'] for v in checks),
             'cached_native_replay_max_logit_error':max(v['max_abs_logit_error'] for v in run['cached_native_replay'].values()),'source_checks_count':len(checks),'checkpoints':run['checkpoints'],
             'new_forward_examples':run['new_forward_examples'],'optimizer_updates':0,'test_data_read':False,
             'seconds':run['seconds'],'scope':plan['scope']}
    output.write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1.6]})
    names=['original','canonical','response','parameter','full_ce']
    labels=['Original merger','Affine folding only','Response alignment','Parameter alignment','Full CE repair']
    colors=['#707070','#ad8a4b','#b5504d','#4a80a6','#4e8a67']
    values=[100*macro[k] for k in names]
    axes[0].barh(labels,values,color=colors);axes[0].invert_yaxis()
    for i,v in enumerate(values):axes[0].text(v+.25,i,f'{v:.3f}',va='center',fontsize=9)
    axes[0].axvline(100*prior['joint_macro_bacc'],color='black',ls='--',label='Joint reference')
    axes[0].set_xlim(0,74);axes[0].set_xlabel('Macro balanced accuracy (%)');axes[0].legend(fontsize=8)
    y=np.arange(len(tasks))
    for kind,offset,color in [('response',-.16,colors[2]),('parameter',.16,colors[3])]:
        delta=contrasts[kind+'_minus_canonical']['per_task_gain_pp']
        axes[1].scatter([delta[t] for t in tasks],y+offset,label=kind,color=color,s=24)
    axes[1].axvline(0,color='grey',lw=.8);axes[1].set_yticks(y,[t.replace('MI_','').replace('EMO_','') for t in tasks],fontsize=8)
    axes[1].invert_yaxis();axes[1].set_xlabel('Gain over affine folding only (percentage points)');axes[1].legend(fontsize=8)
    fig.suptitle('Fixed coefficients after function-preserving query-coordinate changes')
    fig.text(.5,.015,'Existing development validation; one expert bank; no coefficient refit or optimizer updates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:fig.savefig(root/f'analysis/figures/query_rotation_merges.{ext}',dpi=180)
    print(json.dumps({'macro_bacc':macro,'source_equivalence_max_logit_error':summary['source_equivalence_max_logit_error'],'seconds':run['seconds']},indent=2))


if __name__=='__main__':main()
