"""Replay fixed proxy coefficients on a uniform same-source-mean target stream."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, prepare_text, sha256
from report_streaming_replica_filter import audit_export
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json
from streaming_bilateral_merge import StreamingMerger
from streaming_crossfit_training import source_mean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--condition',choices=['out_of_fold','in_fold'],required=True)
    parser.add_argument('--order',choices=['forward','reverse'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    assert json.loads(Path(plan['engineering']).read_text())['status']=='passed'
    name=f'{args.condition}_{args.order}'
    outdir=args.plan.parent/name
    outdir.mkdir(exist_ok=False)
    storage=Path(plan['checkpoint_storage_root'])/name
    storage.mkdir(parents=True,exist_ok=False)
    reference=Path(plan['coefficient_trace_root'])/name
    fit=json.loads((reference/'run.json').read_text())
    assert fit['status']=='completed' and fit['condition']==args.condition
    tasks=fit['order']
    assert tasks==(plan['pilot_tasks'] if args.order=='forward' else list(reversed(plan['pilot_tasks'])))
    strict_settings()
    device=configure(args.gpu,plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2,device)
    source=json.loads(Path(plan['source_fit']).read_text())
    data=TaskData(plan,source,device)
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records=[]
    run={'status':'running','plan_sha256':sha256(args.plan),'condition':args.condition,'order':tasks,
        'coefficient_trace_run_sha256':sha256(reference/'run.json'),'stages':records,
        'new_optimizer_updates':0,'new_training_CE_examples':0,
        'reused_coefficient_updates':fit['updates'],'reused_training_CE_examples':fit['training_examples'],
        'inference_checkpoint_count':1,'inference_routing':False,'calibration_EEG_read':False,
        'official_test_read':False,'strict_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
    previous=None
    started=time.monotonic()
    for count,task in enumerate(tasks):
        stage=count+1
        labels,text,instruction,prototypes=prepare_text(data.config,task)
        assert labels==source['labels'][task] and text==source['instruction'][task]
        data.readouts[task]=instruction.to(device),prototypes.to(device)
        data.records[task]={}
        paths=[source['experts'][task],*plan['half_experts'][task]]
        models=[]
        for p in paths:
            assert sha256(Path(p))==plan['expert_sha256'][p]
            models.append(torch.load(p,map_location='cpu',weights_only=True))
        incoming=source_mean(models,plan['active'])
        directory=outdir/f'stage_{stage:02d}'
        directory.mkdir()
        record={'stage':stage,'arriving_task':task,'source_paths':paths}
        if count==0:
            state=incoming
        else:
            path=reference/f'stage_{stage:02d}'/'coefficients.pt'
            assert sha256(path)==fit['stages'][count]['coefficients_sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert torch.equal(payload['mean'],payload['snapshots'].double().mean(0).float())
            merger=StreamingMerger(leaf,base,previous,incoming,None,count,plan['active'],'old_rows',None)
            state=merger.materialize(payload['mean'].to(device))
            record.update(coefficient_path=str(path),coefficient_payload_sha256=sha256(path),
                independent_bilateral_export_error=audit_export(base,previous,incoming,state,payload['mean'],plan['active']))
            del merger,payload
        worlds=[]
        original=reference/f'stage_{stage:02d}'
        world_hashes=[fit['stages'][count]['checkpoint_sha256'],
                      *[p['sha256'] for p in fit['stages'][count]['proxies']]]
        for filename,expected in zip(('merged.ckpt','proxy0.ckpt','proxy1.ckpt'),world_hashes):
            assert sha256(original/filename)==expected
            worlds.append(torch.load(original/filename,map_location='cpu',weights_only=True))
        averaged=source_mean(worlds,plan['active'])
        error=max(float((state[n]-averaged[n]).abs().max()) for n in plan['active'])
        assert error<2e-6,error
        record['source_mean_target_vs_three_world_average_max_error']=error
        assert all(torch.equal(state[n],base[n]) for n in set(base)-set(plan['active']))
        checkpoint=save_tensor_file(state,directory/'merged.ckpt',outdir,storage)
        record.update(checkpoint_storage_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
        previous=state
        leaf.load_state_dict(state,strict=True)
        del models,incoming,worlds,averaged,state
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation']=evaluate(leaf,data,tasks[:stage],plan['validation_batch_size'],device)
            record['macro_BAcc']=float(np.mean([v['BAcc'] for v in record['validation'].values()]))
        records.append(record)
        run.update(elapsed_seconds=time.monotonic()-started,dataset_audits=data.records,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(outdir/'run.json',run)
        print(json.dumps({'condition':args.condition,'stage':stage,'macro_BAcc':record.get('macro_BAcc')}),flush=True)
    run['status']='completed'
    write_json(outdir/'run.json',run)


if __name__=='__main__':
    main()
