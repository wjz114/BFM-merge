"""Reproduce the first full-length control violation and verify both projections."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier,load_leaf,sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData,strict_settings
from streaming_bilateral_merge import StreamingMerger
from streaming_step_projection import RowGeometry,project_proposal


def main():
    root=Path('runs/streaming_step_projection_v1')
    spec=json.loads((root/'violation_engineering_spec.json').read_text())
    for p,h in spec['sha256'].items():assert sha256(Path(p))==h,p
    assert not (root/'violation_engineering.json').exists()
    plan=json.loads((root/'plan.json').read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    run=json.loads(Path(spec['control_run']).read_text())
    assert run['status']=='completed'
    target_stage,target_update=spec['stage'],spec['update_index']
    target=run['stages'][target_stage-1]
    assert min(target['projection_records'][target_update]['before_margins'])<0
    tasks=run['order'][:target_stage]
    strict_settings();device=configure(spec['gpu'],plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'],device)
    source=json.loads(Path(plan['source_fit']).read_text());data=TaskData(plan,source,device)
    for task in tasks:data.arrive(task)
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    previous_path=Path(run['stages'][target_stage-2]['checkpoint_storage_path'])
    assert sha256(previous_path)==run['stages'][target_stage-2]['checkpoint_sha256']
    previous=torch.load(previous_path,map_location='cpu',weights_only=True)
    source_path=Path(source['experts'][tasks[-1]])
    assert sha256(source_path)==plan['expert_sha256'][str(source_path)]
    incoming=torch.load(source_path,map_location='cpu',weights_only=True)
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    merger=StreamingMerger(leaf,base,previous,incoming,None,target_stage-1,plan['active'],'old_rows',None)
    geometry=RowGeometry(base,previous,incoming,plan['active'],plan['metric_damping'])
    wrappers={t:PrototypeClassifier(leaf,*data.readouts[t]).eval() for t in tasks}
    c=merger.coefficients;optimizer=torch.optim.Adam([c],lr=plan['learning_rate'])
    samples={t:np.random.default_rng([plan['seed'],target_stage,plan['tasks'].index(t),217]).permutation(512)
             for t in tasks}
    rng=np.random.default_rng([plan['seed'],target_stage,931]);bs=plan['batch_size']
    index=-1;loss_errors=[];started=time.monotonic()
    for rnd in range(plan['rounds']):
        order=[tasks[int(k)] for k in rng.permutation(target_stage)]
        for repeat in range(target_stage):
            index+=1;gradients=[];losses=[]
            for t in order:
                ids=samples[t][rnd*bs:(rnd+1)*bs];x,y=data.train[t];labels=y[ids].to(device)
                output=merger.logits(wrappers[t],x[ids].to(device))
                loss=(F.cross_entropy(output,labels,reduction='none')*data.weights[t][labels]).mean()
                gradients.append(torch.autograd.grad(loss,c)[0]);losses.append(float(loss.detach()))
            expected=target['projection_records'][index]
            assert expected['task_order']==order
            error=max(abs(a-b) for a,b in zip(losses,expected['task_losses']))
            loss_errors.append(error);assert error<1e-7,error
            before=c.detach().cpu().clone();g=torch.stack(gradients)
            optimizer.zero_grad(set_to_none=True);c.grad=g.mean(0);optimizer.step()
            if index==target_update:break
        if index==target_update:break
    proposal=geometry.collapse(before.double()-c.detach().cpu().double())
    rows=geometry.row_gradients(g.detach().cpu().double())
    margins=(rows*proposal).sum((1,2))
    margin_error=max(abs(a-b) for a,b in zip(margins.tolist(),expected['before_margins']))
    assert margin_error<1e-10 and min(margins)<0
    checks={}
    for condition in ['euclidean_step_projection','native_metric_step_projection']:
        metric,inverse=geometry.matrices(condition)
        projected,record=project_proposal(proposal,rows,metric,inverse)
        assert record['projected']
        corrected=c.detach().clone()-geometry.lift(projected-proposal).to(device=device,dtype=c.dtype)
        realized=geometry.collapse(before.double()-corrected.detach().cpu().double())
        norms=(rows*torch.einsum('rij,trj->tri',inverse,rows)).sum((1,2)).sqrt()
        normalized_realized=(rows*realized).sum((1,2))/(norms*record['proposal_metric_norm'])
        assert float(normalized_realized.min())>=-2e-4
        state=merger.materialize(corrected)
        native_error=audit_export(base,previous,incoming,state,corrected.cpu(),plan['active'])
        with torch.no_grad():
            parameters=merger.compose(corrected)
            functional={t:merger.logits(wrappers[t],data.train[t][0][:4].to(device),parameters).cpu()
                        for t in tasks}
            leaf.load_state_dict(state,strict=True)
            exports={t:compare(functional[t],wrappers[t](data.train[t][0][:4].to(device)).cpu(),
                               data.train[t][1][:4]) for t in tasks}
        assert all(v['max_abs_logit_error']<=1e-5 and v['changed_predictions']==0 for v in exports.values())
        checks[condition]=dict(projection=record,minimum_realized_normalized_margin=float(normalized_realized.min()),
            independent_native_export_max_abs_error=native_error,exports=exports)
    result=dict(status='passed',spec_sha256=sha256(root/'violation_engineering_spec.json'),
        control_run=spec['control_run'],stage=target_stage,update_index=target_update,
        replayed_optimizer_updates=target_update+1,training_CE_examples=(target_update+1)*target_stage*bs,
        max_replayed_loss_error=max(loss_errors),replayed_margin_max_abs_error=margin_error,
        checks=checks,development_EEG_read=False,official_test_read=False,
        elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    with (root/'violation_engineering.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
