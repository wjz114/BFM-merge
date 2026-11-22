"""Audit unchanged gradient-table difference on a second source/calibration bank."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_task_gradient_cache import CONDITIONS


ROOT=Path('runs/streaming_task_gradient_cache_replication_v1/seed2')


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
            del payload
        if stage in plan['validation_stages']:
            assert set(record['validation'])==set(tasks[:stage])
            assert record['macro_BAcc']==np.mean([v['BAcc'] for v in record['validation'].values()])
        previous=state
    reference=json.loads(Path(plan['calibration_reference']).read_text())['dataset_audits']
    for task in tasks:
        audit=run['dataset_audits'][task]
        assert audit['calibration_eeg_hash']==reference[task]['eeg_hash']
        assert audit['calibration_size']==reference[task]['calibration_size']==512
        assert audit['class_weights']==reference[task]['class_weights']==source['ce_class_weights'][task]
        assert audit['validation_size']==reference[task]['validation_size']
    return run,dict(run_sha256=sha256(folder/'run.json'),independent_export_max_abs_error=max(errors),
        schedules=schedules,initialization_schedules=initializations,
        source_calibration_seed=plan['seed'],peak_reserved_bytes=run['peak_reserved_bytes'],
        training_seconds=sum(s.get('training_seconds',0.) for s in run['stages']))


def main():
    torch.set_num_threads(4);assert not (ROOT/'comparison.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    runs,audits,results={},{},{}
    for condition in plan['conditions']:
        results[condition]={}
        for order in plan['orders']:
            name=f'{condition}_{order}';runs[name],audits[name]=audit_run(plan,source,base,condition,order)
            results[condition][order]=runs[name]['stages'][-1]['macro_BAcc']
            print('audited',name,results[condition][order],flush=True)
        results[condition]['mean']=float(np.mean(list(results[condition].values())))
    datasets=[r['dataset_audits'] for r in runs.values()];assert all(d==datasets[0] for d in datasets)
    for order in plan['orders']:
        for key in ['schedules','initialization_schedules']:
            a=[audits[f'{c}_{order}'][key] for c in plan['conditions']];assert a[0]==a[1]
        assert runs[f'direct_ce_{order}']['stages'][1]['initial_table_hash']==runs[f'table_difference_{order}']['stages'][1]['initial_table_hash']
    effect={o:100*(results['table_difference'][o]-results['direct_ce'][o]) for o in plan['orders']}
    effect['mean_pp']=float(np.mean(list(effect.values())))
    effect['task_deltas_pp']={o:{t:100*(runs[f'table_difference_{o}']['stages'][-1]['validation'][t]['BAcc']-
        runs[f'direct_ce_{o}']['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']}
        for o in plan['orders']}
    selection=json.loads(Path(plan['selection_report']).read_text())
    assert all(selection['checks'].values())
    prior=selection['contrasts']['table_difference_minus_direct_ce']
    combined={f'seed{s}_{o}':e[o] for s,e in [(0,prior),(2,effect)] for o in plan['orders']}
    taskmeans={t:float(np.mean([e['task_deltas_pp'][o][t] for e in [prior,effect] for o in plan['orders']]))
               for t in plan['pilot_tasks']}
    result=dict(status='passed',plan_sha256=sha256(ROOT/'plan.json'),results=results,
        table_difference_minus_direct_ce=effect,combined_selection_and_replication=dict(
            paired_effects_pp=combined,mean_pp=float(np.mean(list(combined.values()))),
            task_mean_effects_pp=taskmeans,leave_one_task_out_mean_pp={t:float(np.mean([v for k,v in taskmeans.items() if k!=t])) for t in taskmeans}),
        audits=audits,checks=dict(frozen_sources=True,unchanged_training_implementation=True,
            paired_initialization_and_update_schedules=True,matched_CE_and_optimizer_budget=True,
            independent_native_exports=True,matched_dataset_audits=True,independent_cache_age_schedule=True,
            first_merge_initial_table_identical=True),scientific_runs=4,scientific_updates=432,
        training_CE_examples=7488,initialization_CE_examples=576,new_source_training_steps=0,
        official_test_read=False,novelty_established=False,overall_goal_achieved=False,
        scope='Second existing source/calibration seed changes both expert training and calibration sampling together. Four tasks,two orders,exposed full development data. This tests a known optimizer component,not a new method claim.')
    with (ROOT/'comparison.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(replication=effect,combined=result['combined_selection_and_replication']),indent=2),flush=True)


if __name__=='__main__':main()


