"""Causal native repair with task-expert behavior constraints."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf,sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData,strict_settings,evaluate,write_json
from source_behavior_repair_training import CONDITIONS,train_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--condition',choices=CONDITIONS,required=True)
    parser.add_argument('--order',choices=['forward','reverse'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    name=f'{args.condition}_{args.order}'
    outdir=args.plan.parent/name;outdir.mkdir(exist_ok=False)
    storage=Path(plan['checkpoint_storage_root'])/name;storage.mkdir(parents=True,exist_ok=False)
    strict_settings();device=configure(args.gpu,plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'],device)
    source=json.loads(Path(plan['source_fit']).read_text())
    data=TaskData(plan,source,device)
    data.experts,data.teacher_logits={},{}
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    tasks=plan['pilot_tasks'] if args.order=='forward' else list(reversed(plan['pilot_tasks']))
    previous,records,started=None,[],time.monotonic()
    run=dict(plan=str(args.plan),plan_sha256=sha256(args.plan),condition=args.condition,
        order=tasks,gpu=args.gpu,status='running',stages=records,inference_checkpoint_count=1,
        inference_routing=False,official_test_read=False,source_experts_used_per_task=1,
        strict_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_version=torch.__version__,cuda_version=torch.version.cuda)
    for count,task in enumerate(tasks):
        stage=count+1;data.arrive(task)
        assert set(data.train)==set(tasks[:stage])
        path=Path(source['experts'][task])
        assert sha256(path)==plan['expert_sha256'][str(path)]
        incoming=torch.load(path,map_location='cpu',weights_only=True)
        if args.condition != 'plain_ce':
            teacher_path=Path(plan['teacher_targets'][task])
            assert sha256(teacher_path)==plan['teacher_sha256'][task]
            teacher=torch.load(teacher_path,map_location='cpu',weights_only=True)
            assert torch.equal(teacher['labels'],data.train[task][1])
            data.teacher_logits[task]=teacher['logits']
            data.experts[task]=incoming
        directory=outdir/f'stage_{stage:02d}';directory.mkdir()
        if count==0:
            state=incoming;leaf.load_state_dict(state,strict=True)
            record=dict(updates=0,training_examples=0)
        else:
            state,payload,record=train_stage(leaf,base,previous,incoming,count,data,tasks[:stage],
                plan,args.condition,device)
            save_tensor_file(payload,directory/'coefficients.pt',outdir,storage)
            record['coefficients_sha256']=sha256(directory/'coefficients.pt')
            del payload
        previous=state
        checkpoint=save_tensor_file(state,directory/'merged.ckpt',outdir,storage)
        record.update(stage=stage,arriving_task=task,source_path=str(path),
            source_sha256=plan['expert_sha256'][str(path)],checkpoint_storage_path=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint))
        del state,incoming;gc.collect();torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation']=evaluate(leaf,data,tasks[:stage],plan['validation_batch_size'],device)
            record['macro_BAcc']=float(np.mean([v['BAcc'] for v in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records,elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(outdir/'run.json',run)
        print(json.dumps(dict(name=name,stage=stage,macro_BAcc=record.get('macro_BAcc'),
            auxiliary_KL_examples=record.get('auxiliary_KL_examples',0),seconds=run['elapsed_seconds'])),flush=True)
    run.update(status='completed',updates=sum(r['updates'] for r in records),
               training_examples=sum(r['training_examples'] for r in records))
    write_json(outdir/'run.json',run)


if __name__=='__main__':
    main()
