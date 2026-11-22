"""Independent table arithmetic, direct-CE replay, and native export checks."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf,sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData,strict_settings
from streaming_crossfit_training import train_stage as original_train_stage
from streaming_task_gradient_cache import CONDITIONS,train_stage,update_direction


def main():
    root=Path('runs/streaming_task_gradient_cache_v1');path=root/'engineering_plan.json'
    plan=json.loads(path.read_text());assert not (root/'engineering.json').exists()
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    strict_settings();device=configure(plan['gpu'],plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'],device)
    # Non-identical old/new entries expose the old-mean/new-mean SAGA off-by-one error.
    toy=torch.tensor([[1.,3.,-2.],[4.,-1.,5.],[-3.,2.,6.]],dtype=torch.float64)
    fresh=torch.tensor([7.,-4.,1.],dtype=torch.float64)
    expected={'direct_ce':fresh,'table_mean':torch.tensor([5.,1.,5.],dtype=torch.float64)/3,
              'table_difference':fresh-toy[1]+toy.sum(0)/3}
    toy_errors={}
    for condition in CONDITIONS:
        table=toy.clone();result=update_direction(table,1,fresh,condition)
        toy_errors[condition]=float((result-expected[condition]).abs().max())
        assert toy_errors[condition]<1e-12 and torch.equal(table[1],fresh)
        assert torch.equal(table[[0,2]],toy[[0,2]])
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    records,schedules,inits={}, {}, {}
    updates,examples,reference_updates,reference_examples,max_error=0,0,0,0,0.
    started=time.monotonic()
    for condition in CONDITIONS:
        data=TaskData(plan,source,device)
        leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        previous,records[condition],schedules[condition],inits[condition]=None,[],[],[]
        for count,task in enumerate(plan['pilot_tasks']):
            data.arrive(task);p=Path(source['experts'][task]);assert sha256(p)==plan['expert_sha256'][str(p)]
            incoming=torch.load(p,map_location='cpu',weights_only=True)
            if count==0:previous=incoming;continue
            state,payload,record=train_stage(leaf,base,previous,incoming,count,data,
                plan['pilot_tasks'][:count+1],plan,condition,device,audit_directions=True)
            assert torch.equal(payload['mean'],payload['snapshots'].double().mean(0).float())
            error=audit_export(base,previous,incoming,state,payload['mean'],plan['active'])
            max_error=max(max_error,error);record['independent_export_max_abs_error']=error
            if condition=='direct_ce':
                reference,_,ref_payload,ref_record=original_train_stage(leaf,base,previous,incoming,[],[],
                    count,data,plan['pilot_tasks'][:count+1],{},plan,'single_full',device)
                assert torch.equal(payload['snapshots'],ref_payload['snapshots'])
                assert all(torch.equal(state[k],reference[k]) for k in state)
                assert record['round_losses']==ref_record['round_losses']
                assert record['schedule_hash']==ref_record['schedule_hash']
                record['direct_trajectory_bitwise_matches_original']=True
                reference_updates+=ref_record['updates'];reference_examples+=ref_record['training_examples']
                del reference,ref_payload
            records[condition].append(record);schedules[condition].append(record['schedule_hash'])
            inits[condition].append(record['initialization_schedule_hash'])
            updates+=record['updates'];examples+=record['training_examples'];previous=state
            del payload,incoming;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(condition=condition,status='checked')),flush=True)
        del previous,leaf,data
    assert all(v==schedules['direct_ce'] for v in schedules.values())
    assert all(v==inits['direct_ce'] for v in inits.values())
    assert (updates,examples,reference_updates,reference_examples)==(54,324,18,72)
    result=dict(status='passed',plan_sha256=sha256(path),records=records,toy_errors=toy_errors,
        updates=updates+reference_updates,training_CE_examples=examples+reference_examples,
        candidate_updates=updates,candidate_CE_examples=examples,
        direct_reference_updates=reference_updates,direct_reference_CE_examples=reference_examples,
        direct_trajectory_bitwise_matches_original=True,paired_initialization_and_update_schedules=True,
        independent_export_max_abs_error=max_error,development_EEG_read=False,official_test_read=False,
        elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    with (root/'engineering.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
