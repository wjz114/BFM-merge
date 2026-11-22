"""Check that proxy coefficient learning ignores the deployment target."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_strict_history import TaskData, strict_settings
from streaming_crossfit_training import source_mean, train_stage


def main():
    root=Path('runs/crossfit_source_mean_target_v1')
    path=root/'engineering_plan.json'
    plan=json.loads(path.read_text())
    assert not (root/'engineering.json').exists()
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    strict_settings()
    device=configure(0,plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2,device)
    source=json.loads(Path(plan['source_fit']).read_text())
    data=TaskData(plan,source,device)
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    models,assignments={},{}
    for task in plan['engineering_tasks']:
        data.arrive(task)
        paths=[source['experts'][task],*plan['half_experts'][task]]
        models[task]=[]
        for p in paths:
            assert sha256(Path(p))==plan['expert_sha256'][p]
            models[task].append(torch.load(p,map_location='cpu',weights_only=True))
        with np.load(plan['folds']) as folds:
            assignments[task]=folds[f'{task}__calibration_fold'].copy()
    a,b=plan['engineering_tasks']
    checks={}
    started=time.monotonic()
    for condition in ('out_of_fold','in_fold'):
        outputs=[]
        for use_mean in (False,True):
            previous=source_mean(models[a],plan['active']) if use_mean else models[a][0]
            incoming=source_mean(models[b],plan['active']) if use_mean else models[b][0]
            outputs.append(train_stage(leaf,base,previous,incoming,models[a][1:],models[b][1:],
                1,data,plan['engineering_tasks'],assignments,plan,condition,device))
            gc.collect()
            torch.cuda.empty_cache()
        full,mean=outputs
        assert torch.equal(full[2]['snapshots'],mean[2]['snapshots'])
        assert torch.equal(full[2]['mean'],mean[2]['mean'])
        assert full[3]['schedule_hash']==mean[3]['schedule_hash']
        assert full[3]['routing_hash']==mean[3]['routing_hash']
        assert all(torch.equal(x[n],y[n]) for x,y in zip(full[1],mean[1]) for n in base)
        averaged=source_mean([full[0],*full[1]],plan['active'])
        error=max(float((averaged[n]-mean[0][n]).abs().max()) for n in plan['active'])
        assert error<2e-6,error
        checks[condition]={'coefficient_snapshots_bitwise_equal':True,'proxy_states_bitwise_equal':True,
            'schedules_and_routing_equal':True,'source_mean_target_vs_three_world_average_max_error':error,
            'updates':sum(x[3]['updates'] for x in outputs),
            'training_CE_examples':sum(x[3]['training_examples'] for x in outputs)}
    result={'status':'passed','plan_sha256':sha256(path),'checks':checks,
        'updates':16,'training_CE_examples':64,'validation_EEG_read':False,'official_test_read':False,
        'elapsed_seconds':time.monotonic()-started,'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'scope':'Actual half-source experts, two arrivals and both fold mappings. Coefficient equality checks deployment-target independence; affine averaging is additionally audited on every full pilot replay stage.'}
    with (root/'engineering.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
