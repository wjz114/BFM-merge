"""Audit gradient-table controls and compare effects with explicit probe costs."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_task_gradient_cache import CONDITIONS


ROOT=Path('runs/streaming_task_gradient_cache_v1')


def expected_schedule(plan,tasks,stage):
    batches={t:np.random.default_rng([plan['seed'],stage,plan['tasks'].index(t),217]).permutation(512)
             for t in tasks[:stage]}
    init=[[plan['tasks'].index(t),*batches[t][:plan['batch_size']].tolist()] for t in tasks[:stage]]
    rng=np.random.default_rng([plan['seed'],stage,931]);schedule=[];ages=[]
    stamps=np.zeros(stage,dtype=np.int64)
    for rnd in range(plan['rounds']):
        for k in rng.permutation(stage):
            index=int(k);task=tasks[index];step=len(schedule)
            old_age=step-int(stamps[index]);stamps[index]=step
            ages.append(dict(task=task,round=rnd+1,refreshed_entry_age=old_age,
                             table_entry_ages=(step-stamps).tolist()))
            ids=batches[task][rnd*plan['batch_size']:(rnd+1)*plan['batch_size']]
            schedule.append([plan['tasks'].index(task),*ids.tolist()])
    return np.asarray(schedule,dtype=np.int64),np.asarray(init,dtype=np.int64),ages


def audit_run(plan,source,base,condition,order):
    folder=ROOT/f'{condition}_{order}';run=json.loads((folder/'run.json').read_text())
    tasks=plan['pilot_tasks'] if order=='forward' else list(reversed(plan['pilot_tasks']))
    assert run['status']=='completed' and run['order']==tasks and len(run['stages'])==4
    assert run['plan_sha256']==sha256(ROOT/'plan.json') and run['condition']==condition
    assert run['updates']==108 and run['training_examples']==1872
    assert run['inference_checkpoint_count']==1 and not run['inference_routing']
    assert not run['official_test_read'] and run['source_experts_used_per_task']==1
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    old_path=Path(plan['reference_runs']['original_sequential_ce'][order])
    old=json.loads(old_path.read_text())
    assert old['status']=='completed' and old['order']==tasks
    previous,errors,schedules,initializations=None,[],[],[]
    for stage,task in enumerate(tasks,1):
        record=run['stages'][stage-1]
        assert record['stage']==stage and record['arriving_task']==task
        assert record['source_path']==source['experts'][task]
        assert sha256(Path(record['source_path']))==record['source_sha256']==plan['expert_sha256'][record['source_path']]
        incoming=torch.load(record['source_path'],map_location='cpu',weights_only=True)
        path=folder/f'stage_{stage:02d}/merged.ckpt'
        assert path.is_symlink() and str(path.resolve())==record['checkpoint_storage_path']
        assert sha256(path)==record['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state.keys()==base.keys()
        assert all(torch.equal(state[k],base[k]) for k in set(base)-set(plan['active']))
        assert all(torch.isfinite(state[k]).all() for k in plan['active'])
        if stage==1:
            assert record['updates']==record['training_examples']==0
            assert all(torch.equal(state[k],incoming[k]) for k in base)
        else:
            path=folder/f'stage_{stage:02d}/coefficients.pt'
            assert sha256(path)==record['coefficients_sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert payload['snapshots'].shape==(6,347666) and torch.isfinite(payload['snapshots']).all()
            assert torch.equal(payload['mean'],payload['snapshots'].double().mean(0).float())
            errors.append(audit_export(base,previous,incoming,state,payload['mean'],plan['active']))
            schedule,init,ages=expected_schedule(plan,tasks,stage)
            assert np.array_equal(payload['schedule'].numpy(),schedule)
            assert np.array_equal(payload['initialization_schedule'].numpy(),init)
            assert record['schedule_hash']==array_hash(schedule)
            assert record['initialization_schedule_hash']==array_hash(init)
            schedules.append(record['schedule_hash']);initializations.append(record['initialization_schedule_hash'])
            assert record['updates']==12*stage and record['training_examples']==208*stage
            assert record['update_training_examples']==192*stage and record['initialization_training_examples']==16*stage
            assert record['training_forward_batches']==record['backward_calls']==13*stage
            assert record['cache_entries']==stage and record['cache_bytes']==stage*347666*4
            assert record['cache_reset_at_arrival'] and len(record['round_table_hashes'])==12
            assert len(record['cache_records'])==12*stage and len(record['initial_losses'])==stage
            for actual,expected in zip(record['cache_records'],ages):
                assert all(actual[k]==v for k,v in expected.items())
                assert actual['current_gradient_norm']>=0 and actual['used_gradient_norm']>=0
                assert np.isfinite([actual['loss'],actual['current_to_used_cosine']]).all()
            assert set(record['export_checks'])==set(tasks[:stage])
            assert all(v['max_abs_logit_error']<=1e-5 and v['changed_predictions']==0
                       for v in record['export_checks'].values())
            if condition=='direct_ce':
                prior=torch.load(old_path.parent/f'stage_{stage:02d}/coefficients.pt',map_location='cpu',weights_only=True)
                assert torch.equal(payload['snapshots'],prior['snapshots'])
                assert record['round_losses']==old['stages'][stage-1]['round_losses']
                assert record['schedule_hash']==old['stages'][stage-1]['schedule_hash']
                del prior
            del payload
        if condition=='direct_ce':
            prior=torch.load(old['stages'][stage-1]['checkpoint_storage_path'],map_location='cpu',weights_only=True)
            assert all(torch.equal(state[k],prior[k]) for k in base)
            del prior
        if stage in plan['validation_stages']:
            assert set(record['validation'])==set(tasks[:stage])
            assert record['macro_BAcc']==np.mean([v['BAcc'] for v in record['validation'].values()])
            if condition=='direct_ce':assert record['validation']==old['stages'][stage-1]['validation']
        previous=state
    assert run['dataset_audits']==old['dataset_audits']
    return run,dict(run_sha256=sha256(folder/'run.json'),independent_export_max_abs_error=max(errors),
        schedules=schedules,initialization_schedules=initializations,
        direct_matches_original_bitwise=condition=='direct_ce',peak_reserved_bytes=run['peak_reserved_bytes'],
        training_seconds=sum(s.get('training_seconds',0.) for s in run['stages']))


def main():
    torch.set_num_threads(4);assert not (ROOT/'comparison.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    runs,audits,results={},{},{}
    for condition in CONDITIONS:
        results[condition]={}
        for order in plan['orders']:
            name=f'{condition}_{order}'
            runs[name],audits[name]=audit_run(plan,source,base,condition,order)
            results[condition][order]=runs[name]['stages'][-1]['macro_BAcc']
            print('audited',name,results[condition][order],flush=True)
        results[condition]['mean']=float(np.mean(list(results[condition].values())))
    datasets=[r['dataset_audits'] for r in runs.values()];assert all(d==datasets[0] for d in datasets)
    for order in plan['orders']:
        for key in ['schedules','initialization_schedules']:
            a=[audits[f'{c}_{order}'][key] for c in CONDITIONS];assert all(v==a[0] for v in a)
        initial=[runs[f'{c}_{order}']['stages'][1]['initial_table_hash'] for c in CONDITIONS]
        assert len(set(initial))==1
        reference=json.loads(Path(plan['reference_runs']['simultaneous_mean_adam'][order]).read_text())
        assert reference['status']=='completed' and reference['dataset_audits']==datasets[0]
        runs[f'simultaneous_mean_adam_{order}']=reference
    results['simultaneous_mean_adam']={o:runs[f'simultaneous_mean_adam_{o}']['stages'][-1]['macro_BAcc']
                                       for o in plan['orders']}
    results['simultaneous_mean_adam']['mean']=float(np.mean(list(results['simultaneous_mean_adam'].values())))
    contrasts={}
    for a,b in [('table_mean','direct_ce'),('table_difference','direct_ce'),('table_difference','table_mean'),
                ('table_mean','simultaneous_mean_adam'),('table_difference','simultaneous_mean_adam')]:
        values={o:100*(results[a][o]-results[b][o]) for o in plan['orders']}
        values['mean_pp']=float(np.mean(list(values.values())))
        values['task_deltas_pp']={o:{t:100*(runs[f'{a}_{o}']['stages'][-1]['validation'][t]['BAcc']-
            runs[f'{b}_{o}']['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']}
            for o in plan['orders']}
        contrasts[f'{a}_minus_{b}']=values
    result=dict(status='passed',plan_sha256=sha256(ROOT/'plan.json'),results=results,contrasts=contrasts,audits=audits,
        checks=dict(frozen_sources=True,paired_update_and_initialization_schedules=True,paired_CE_and_update_budgets=True,
            independent_native_exports=True,same_full_development_data=True,independent_cache_age_schedule=True,
            first_merge_initial_table_identical=True,direct_trajectory_bitwise_matches_original=True),
        scientific_runs=6,scientific_updates=648,training_CE_examples=11232,
        initialization_CE_examples=864,update_CE_examples=10368,new_source_steps=0,official_test_read=False,
        gradient_exposure_fraction_of_simultaneous_reference=1872/5568,
        novelty_established=False,overall_goal_achieved=False,
        scope='Known within-stage gradient-table components inside Adam; one source/calibration seed,four tasks,two orders,exposed development data. These results do not establish a novel method or full16 improvement.')
    with (ROOT/'comparison.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v['mean_pp'] for k,v in contrasts.items()},indent=2),flush=True)


if __name__=='__main__':main()
