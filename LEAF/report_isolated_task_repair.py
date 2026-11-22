"""Compare task-isolated repair copies with matched shared-model histories."""
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_query_rotation_merges import contrast


def main():
    torch.set_num_threads(4);root=Path('runs/paper_merge_v2')
    plan_path=root/'inputs/isolated_task_repair_plan.json';plan=json.loads(plan_path.read_text())
    output=root/'analysis/isolated_task_repair_summary.json';assert not output.exists()
    assert sha256(Path(__file__))==plan['reporter_sha256'];tasks=plan['datasets']
    active=set(json.loads((Path(plan['source_fit'])/'run.json').read_text())['groups'])
    source_paths={};runs={};exports={};metrics={};initial_differences=[]
    schedule=np.load(plan['schedule']);folder=Path(plan['outdir'])
    for shard in [0,1]:
        worker=json.loads((folder/f'worker_{shard}.json').read_text());assert worker['status']=='complete'
        assert worker['plan_sha256']==sha256(plan_path)
    for k,task in enumerate(tasks):
        task_folder=folder/task;path=task_folder/'run.json';run=json.loads(path.read_text())
        assert run['plan_sha256']==sha256(plan_path) and run['updates']==100 and run['reset_to_initial_exact']
        assert run['average_steps']==plan['task_average_steps'] and not run['test_data_read']
        assert run['local_batch_indices_hash']==array_hash(schedule['local_indices'][schedule['task']==k])
        assert run['optimizer_audit']['bias_correction_counter_values']==[100]
        assert sha256(task_folder/'merged.ckpt')==run['checkpoint_sha256']
        states=[]
        for row in run['average_snapshots']:
            assert sha256(Path(row['path']))==row['sha256']
            states.append(torch.load(row['path'],map_location='cpu',weights_only=True))
        merged=torch.load(task_folder/'merged.ckpt',map_location='cpu',weights_only=True)
        for n,v in merged.items():
            expected=torch.stack([s[n].double() for s in states]).mean(0).float() if n in active else states[0][n]
            assert torch.equal(v,expected),(task,n)
        check=run['export_check'];assert check['max_abs_logit_error']<=1e-5 and all(v['margin_within_twice_error'] for v in check['changed_rows'])
        exports[task]={'tensor_count':len(merged),'all_saved_tensors_reconstruct_exactly':True,'native_own_task_export_pass':True}
        source_paths[str(path)]=sha256(path);runs[task]=run;metrics[task]=run['average_validation'];del states,merged
    baselines={}
    for arm in ['global','task']:
        path=Path(plan['shared_runs'][arm]);assert sha256(path)==plan['shared_run_sha256'][arm]
        run=json.loads(path.read_text());baselines[arm]=run;source_paths[str(path)]=sha256(path)
        for task in tasks:
            assert abs(runs[task]['initial_validation']['BAcc']-run['initial_validation']['validation'][task]['BAcc'])<1e-12
            for metric,value in runs[task]['initial_validation'].items():
                if value!=run['initial_validation']['validation'][task][metric]:
                    initial_differences.append({'arm':arm,'task':task,'metric':metric,'isolated':value,'shared':run['initial_validation']['validation'][task][metric]})
    prior=json.loads((root/'analysis/task_state_training_summary.json').read_text())
    macro=float(np.mean([metrics[t]['BAcc'] for t in tasks]))
    result={'unix':time.time(),'plan_sha256':sha256(plan_path),'reporter_sha256':sha256(Path(__file__)),
            'datasets':tasks,'sources':source_paths,'isolated_macro_bacc':macro,'shared_macro_bacc':prior['macro_bacc'],
            'isolated_minus_shared':{arm:contrast(metrics,r['average_validation'],tasks) for arm,r in baselines.items()},
            'isolated_minus_initial':contrast(metrics,baselines['global']['initial_validation']['validation'],tasks),
            'remaining_joint_gap_pp':100*(prior['joint_macro_bacc']-macro),
            'initial_metric_differences':initial_differences,'initial_bacc_exact':True,'batch_order_and_multiplicity_exact':True,
            'average_reconstruction_checks':exports,'isolated_validation':metrics,
            'original512_calibration':{t:r['average_calibration'] for t,r in runs.items()},
            'scientific_optimizer_updates':1600,'training_examples':204800,'number_of_distinct_models':16,
            'static_shared_candidate':False,'test_data_read':False,'scope':plan['scope']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,6));y=np.arange(len(tasks));d=result['isolated_minus_shared']['task']['per_task_gain_pp']
    ax.barh(y,[d[t] for t in tasks],color=['#4a80a6' if d[t]>=0 else '#b5504d' for t in tasks]);ax.axvline(0,color='grey',lw=.8)
    ax.set_yticks(y,[t.replace('MI_','').replace('EMO_','') for t in tasks]);ax.invert_yaxis()
    ax.set_xlabel('Isolated copy minus shared task-state model (percentage points)')
    ax.set_title('Do other tasks help or hinder this fixed repair budget?')
    fig.text(.5,.015,'Same 100 batches per task and visit-clock Adam; 16 diagnostic copies, not a deployable shared-model result.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['png','pdf']:fig.savefig(root/f'analysis/figures/isolated_task_repair.{ext}',dpi=180)
    print(json.dumps({'isolated_macro_bacc':macro,'isolated_minus_shared':result['isolated_minus_shared'],'remaining_joint_gap_pp':result['remaining_joint_gap_pp']},indent=2))


if __name__=='__main__':main()
