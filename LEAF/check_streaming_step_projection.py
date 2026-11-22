"""Real-model projection engineering without development/test evaluation."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf,sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData,strict_settings
from streaming_step_projection import CONDITIONS
from streaming_step_projection_training import train_stage


def main():
    root=Path('runs/streaming_step_projection_v1')
    path=root/'engineering_plan.json'
    plan=json.loads(path.read_text())
    assert not (root/'engineering.json').exists()
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    strict_settings()
    device=configure(plan['gpu'],plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'],device)
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    started=time.monotonic()
    records,schedules,updates,examples,max_error={},{},0,0,0.
    for condition in CONDITIONS:
        data=TaskData(plan,source,device)
        leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        previous,records[condition],schedules[condition]=None,[],[]
        for count,task in enumerate(plan['pilot_tasks']):
            data.arrive(task)
            path_source=Path(source['experts'][task])
            assert sha256(path_source)==plan['expert_sha256'][str(path_source)]
            incoming=torch.load(path_source,map_location='cpu',weights_only=True)
            if count==0:
                previous=incoming
                continue
            state,payload,record=train_stage(leaf,base,previous,incoming,count,data,
                plan['pilot_tasks'][:count+1],plan,condition,device)
            assert torch.equal(payload['mean'],payload['snapshots'].double().mean(0).float())
            error=audit_export(base,previous,incoming,state,payload['mean'],plan['active'])
            record['independent_export_max_abs_error']=error
            max_error=max(max_error,error)
            records[condition].append(record)
            schedules[condition].append(record['schedule_hash'])
            updates+=record['updates'];examples+=record['training_examples']
            previous=state
            del payload,incoming
            gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(condition=condition,stages=len(records[condition]),
            projected_updates=sum(r['projected_updates'] for r in records[condition]))),flush=True)
        del previous,data,leaf
    assert all(v==next(iter(schedules.values())) for v in schedules.values())
    assert updates==54 and examples==696
    result=dict(status='passed',plan_sha256=sha256(path),records=records,updates=updates,
        training_CE_examples=examples,matched_schedules=True,independent_export_max_abs_error=max_error,
        development_EEG_read=False,official_test_read=False,elapsed_seconds=time.monotonic()-started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    with (root/'engineering.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
