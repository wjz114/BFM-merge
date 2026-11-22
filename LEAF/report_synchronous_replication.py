"""Audit replication of the simultaneous-gradient control on source/calibration seed2."""
import json
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export
from streaming_step_projection import CONDITIONS


ROOT=Path('runs/streaming_synchronous_replication_v1/seed2')


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
    torch.set_num_threads(4);assert not (ROOT/'comparison.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    prior_cache=json.loads(Path(plan['reference_audit']).read_text());assert all(prior_cache['checks'].values())
    runs,refs,audits,results={},{},{},{'simultaneous_mean_adam':{},'direct_ce':{}}
    for order in plan['orders']:
        runs[order],audits[order]=audit_run(plan,source,base,'simultaneous_mean_adam',order)
        refs[order]=json.loads(Path(plan['reference_runs'][order]).read_text())
        assert refs[order]['status']=='completed' and refs[order]['condition']=='direct_ce'
        assert refs[order]['updates']==108 and refs[order]['training_examples']==1872
        assert runs[order]['dataset_audits']==refs[order]['dataset_audits']
        assert runs[order]['stages'][0]['validation']==refs[order]['stages'][0]['validation']
        results['simultaneous_mean_adam'][order]=runs[order]['stages'][-1]['macro_BAcc']
        results['direct_ce'][order]=refs[order]['stages'][-1]['macro_BAcc']
        print('audited',order,results['simultaneous_mean_adam'][order],flush=True)
    for c in results:results[c]['mean']=float(np.mean(list(results[c].values())))
    effect={o:100*(results['simultaneous_mean_adam'][o]-results['direct_ce'][o]) for o in plan['orders']}
    effect['mean_pp']=float(np.mean(list(effect.values())))
    effect['task_deltas_pp']={o:{t:100*(runs[o]['stages'][-1]['validation'][t]['BAcc']-
        refs[o]['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']} for o in plan['orders']}
    selection=json.loads(Path(plan['selection_report']).read_text());assert all(selection['checks'].values())
    prior=selection['contrasts']['simultaneous_mean_adam_minus_original_sequential_ce']
    combined={f'seed{s}_{o}':e[o] for s,e in [(0,prior),(2,effect)] for o in plan['orders']}
    taskmeans={t:float(np.mean([e['task_deltas_pp'][o][t] for e in [prior,effect] for o in plan['orders']]))
               for t in plan['pilot_tasks']}
    result=dict(status='passed',plan_sha256=sha256(ROOT/'plan.json'),results=results,
        simultaneous_minus_direct_ce=effect,combined_selection_and_replication=dict(
            paired_effects_pp=combined,mean_pp=float(np.mean(list(combined.values()))),
            task_mean_effects_pp=taskmeans,leave_one_task_out_mean_pp={t:float(np.mean([v for k,v in taskmeans.items() if k!=t])) for t in taskmeans}),
        audits=audits,checks=dict(frozen_sources=True,unchanged_simultaneous_training=True,
            independent_gradient_schedule=True,independent_native_exports=True,
            matched_data_and_initial_stage=True,explicit_extra_gradient_cost=True),
        scientific_runs=2,scientific_updates=216,training_CE_examples=11136,new_source_steps=0,
        reference_CE_per_run=1872,simultaneous_CE_per_run=5568,original_sequential_without_probes_CE_per_run=1728,
        official_test_read=False,novelty_established=False,overall_goal_achieved=False,
        scope='Second existing expert/calibration bank,four tasks,two orders. A standard synchronized-gradient baseline,not a new merger. Reference direct-CE includes unused startup probes; its update rule is the original sequential one. Higher gradient cost and exposed development data remain explicit.')
    with (ROOT/'comparison.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(replication=effect,combined=result['combined_selection_and_replication']),indent=2),flush=True)


if __name__=='__main__':main()


