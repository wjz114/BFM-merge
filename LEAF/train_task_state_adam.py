"""Matched shared-model CE with either global or per-task native Adam histories."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import train_prehead_distillation as trainer
from d_downstream_qformer_ce import sha256
from task_state_adam import TaskStateAdam


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--condition',choices=['global','task'],required=True);parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert sha256(Path(__file__))==plan['source_sha256'][Path(__file__).name]
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==plan['cuda_allocator']
    assert sha256(Path(plan['engineering_audit']))==plan['engineering_audit_sha256']
    audit=json.loads(Path(plan['engineering_audit']).read_text());assert audit['status']=='complete'
    start=time.time();torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'],args.gpu)
    schedule=np.load(plan['schedule']);instances=[]
    def factory(parameters,**kwargs):
        parameters=list(parameters)
        if args.condition=='global':optimizer=torch.optim.Adam(parameters,**kwargs)
        else:optimizer=TaskStateAdam(parameters,schedule['task'],len(plan['task_state_datasets']),**kwargs)
        instances.append(optimizer);return optimizer
    class TorchProxy:
        optim=SimpleNamespace(Adam=factory)
        def __getattr__(self,name):return getattr(torch,name)
    trainer.torch=TorchProxy()
    sys.argv=[trainer.__file__,'--plan',str(args.plan),'--condition','balanced_ce','--outdir',str(args.outdir),'--gpu',str(args.gpu)]
    trainer.main();assert len(instances)==1
    if args.condition=='task':
        state_audit=instances[0].audit();assert state_audit['updates']==plan['steps']
        assert all(v['visits']==100 for v in state_audit['tasks'].values())
    else:
        counters=[int(s['step'].item()) for s in instances[0].state.values()]
        assert counters and all(v==plan['steps'] for v in counters)
        state_audit={'updates':plan['steps'],'initialized_parameter_states':len(counters),
                     'bias_correction_counter_values':sorted(set(counters)),
                     'moment_tensor_elements':sum(s[k].numel() for s in instances[0].state.values() for k in ['exp_avg','exp_avg_sq'])}
    run_path=args.outdir/'run.json';run=json.loads(run_path.read_text())
    result={'status':'complete','condition':args.condition,'plan_sha256':sha256(args.plan),'run_sha256':sha256(run_path),
            'wrapper_sha256':sha256(Path(__file__)),'optimizer_state_audit':state_audit,'updates':run['updates'],
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(args.gpu),'seconds':time.time()-start,
            'execution':'Unchanged native balanced-CE trainer and one shared LEAF. Global arm returns native Adam directly; task arm maintains16native Adam instances referencing the same Parameters and advances only the scheduled task. No task-specific inference parameters, losses, extra forwards or optimizer-rate search. This alternating-task adaptation is not an exact synchronous AdaTask reproduction.'}
    (args.outdir/'wrapper.json').write_text(json.dumps(result,indent=2)+'\n');print('Wrapper completed',args.condition,flush=True)


if __name__=='__main__':main()
