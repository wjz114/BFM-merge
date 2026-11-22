"""Audit matched examples, native exports and observed projection records."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_step_projection import CONDITIONS


ROOT=Path('runs/streaming_step_projection_v1')


def independent_schedule(plan,tasks,stage):
    batches={t:np.random.default_rng([plan['seed'],stage,plan['tasks'].index(t),217]).permutation(512)
             for t in tasks[:stage]}
    rng=np.random.default_rng([plan['seed'],stage,931]);rows=[];orders=[]
    for rnd in range(plan['rounds']):
        order=[tasks[int(k)] for k in rng.permutation(stage)]
        for repeat in range(stage):
            orders.append(order)
            for t in order:
                ids=batches[t][rnd*plan['batch_size']:(rnd+1)*plan['batch_size']]
                rows.append([rnd+1,repeat+1,plan['tasks'].index(t),*ids.tolist()])
    return np.asarray(rows,dtype=np.int64),orders


def audit_run(plan,source,base,condition,order):
    folder=ROOT/f'{condition}_{order}'
    run=json.loads((folder/'run.json').read_text())
    tasks=plan['pilot_tasks'] if order=='forward' else list(reversed(plan['pilot_tasks']))
    assert run['status']=='completed' and run['order']==tasks and len(run['stages'])==4
    assert run['plan_sha256']==sha256(ROOT/'plan.json') and run['condition']==condition
    assert run['updates']==108 and run['training_examples']==5568
    assert run['inference_checkpoint_count']==1 and not run['inference_routing']
    assert not run['official_test_read'] and run['source_experts_used_per_task']==1
    assert run['strict_deterministic_algorithms'] and run['cudnn_deterministic']
    assert run['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    previous,errors,schedules,projection_count=None,[],[],0
    minimum_slack=0.;maximum_chain_error=0.
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
            schedule,orders=independent_schedule(plan,tasks,stage)
            assert np.array_equal(payload['schedule'].numpy(),schedule)
            assert record['schedule_hash']==array_hash(schedule)
            schedules.append(record['schedule_hash'])
            assert record['updates']==12*stage and record['training_examples']==192*stage*stage
            assert record['training_forward_batches']==record['backward_calls']==12*stage*stage
            assert len(record['projection_records'])==record['updates']
            assert record['metric_damping']==plan['metric_damping']
            projected=0
            for i,r in enumerate(record['projection_records']):
                assert r['task_order']==orders[i] and r['round']==i//stage+1 and r['repeat']==i%stage+1
                assert len(r['task_losses'])==len(r['before_margins'])==stage
                assert r['chain_rule_relative_error']<2e-4
                maximum_chain_error=max(maximum_chain_error,r['chain_rule_relative_error'])
                if condition=='simultaneous_mean_adam':
                    assert not r['projected'] and r['before_margins']==r['after_margins']
                else:
                    assert r['normalized_min_slack']>=-2e-6
                    minimum_slack=min(minimum_slack,r['normalized_min_slack'])
                    assert r['realized_FP32_to_requested_correction_max_abs_error']>=0
                    if r['projected']:
                        dual=np.asarray(r['dual_multipliers'])
                        assert np.all(dual>=-1e-10)
                        assert r['normalized_complementarity']<=2e-6*(1+np.linalg.norm(dual))
                    else:
                        assert min(r['before_margins'])>=0
                projected+=r['projected']
            assert record['projected_updates']==projected
            projection_count+=projected
            assert set(record['export_checks'])==set(tasks[:stage])
            assert all(r['max_abs_logit_error']<=1e-5 and r['changed_predictions']==0
                       for r in record['export_checks'].values())
            del payload
        if stage in plan['validation_stages']:
            assert set(record['validation'])==set(tasks[:stage])
            assert record['macro_BAcc']==np.mean([v['BAcc'] for v in record['validation'].values()])
        previous=state
    return run,dict(run_sha256=sha256(folder/'run.json'),independent_export_max_abs_error=max(errors),
        schedules=schedules,projected_updates=projection_count,minimum_normalized_solver_slack=minimum_slack,
        max_chain_rule_relative_error=maximum_chain_error)


def main():
    torch.set_num_threads(4)
    assert not (ROOT/'comparison.json').exists()
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
    datasets=[r['dataset_audits'] for r in runs.values()]
    assert all(d==datasets[0] for d in datasets)
    for order in plan['orders']:
        schedules=[audits[f'{c}_{order}']['schedules'] for c in CONDITIONS]
        assert all(s==schedules[0] for s in schedules)
        reference=json.loads(Path(plan['reference_runs'][order]).read_text())
        assert reference['status']=='completed' and reference['dataset_audits']==datasets[0]
        runs[f'original_sequential_ce_{order}']=reference
    results['original_sequential_ce']={o:runs[f'original_sequential_ce_{o}']['stages'][-1]['macro_BAcc']
                                       for o in plan['orders']}
    results['original_sequential_ce']['mean']=float(np.mean(list(results['original_sequential_ce'].values())))
    contrasts={}
    for a,b in [('native_metric_step_projection','euclidean_step_projection'),
                ('native_metric_step_projection','simultaneous_mean_adam'),
                ('euclidean_step_projection','simultaneous_mean_adam'),
                *[(c,'original_sequential_ce') for c in CONDITIONS]]:
        values={o:100*(results[a][o]-results[b][o]) for o in plan['orders']}
        values['mean_pp']=float(np.mean(list(values.values())))
        values['task_deltas_pp']={o:{t:100*(runs[f'{a}_{o}']['stages'][-1]['validation'][t]['BAcc']-
            runs[f'{b}_{o}']['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']}
            for o in plan['orders']}
        contrasts[f'{a}_minus_{b}']=values
    result=dict(status='passed',plan_sha256=sha256(ROOT/'plan.json'),results=results,contrasts=contrasts,
        audits=audits,checks=dict(frozen_sources=True,matched_schedules=True,matched_CE_and_update_budgets=True,
            independent_native_exports=True,same_full_development_data=True,recorded_QP_tolerances=True),
        scientific_updates=648,training_CE_examples=33408,new_source_steps=0,official_test_read=False,
        superiority_established=False,novelty_established=False,overall_goal_achieved=False,
        limitation='Projection is first-order on calibration batches; tolerances are numerical, not a finite-step or generalization guarantee. Original sequential CE has fewer CE exposures; matched new mean-Adam is the primary control.')
    with (ROOT/'comparison.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v['mean_pp'] for k,v in contrasts.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
