"""Diagnostic: each task repairs its own copy of M0 using its same100batches."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from coverage_support import training_and_validation
from d_downstream_qformer_ce import ROOT,PrototypeClassifier,classification_metrics,load_leaf,make_loader,prepare_text,sha256
from load_config import build_model_config,load_yaml
from merge_poc import UNUSED_PREFIXES
from prehead_distillation_support import objective


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--shard',type=int,choices=[0,1],required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text())
    for name,expected in plan['source_sha256'].items():assert sha256(Path(name))==expected,name
    for key in ['initial_checkpoint','calibration','schedule','coverage_audit']:
        assert sha256(Path(plan[key]))==plan[key+'_sha256']
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==plan['cuda_allocator']
    device=configure(args.gpu,plan['seed']);torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'],args.gpu)
    start=time.time();root=Path(plan['outdir']);root.mkdir(parents=True,exist_ok=True)
    source=json.loads((Path(plan['source_fit'])/'run.json').read_text());tasks=source['datasets']
    assert tasks==plan['datasets'];schedule=np.load(plan['schedule']);indices=np.load(plan['calibration'])
    coverage=json.loads(Path(plan['coverage_audit']).read_text())
    initial_run=json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    config=build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf=load_leaf(config,Path(plan['initial_checkpoint'])).to(device).eval()
    initial={n:v.detach().cpu().clone() for n,v in leaf.state_dict().items()}
    active={n for n,_ in leaf.named_parameters() if not n.startswith(UNUSED_PREFIXES)}
    assert active==set(source['groups'])
    for n,v in leaf.named_parameters():v.requires_grad_(n in active)
    fixed={n:v for n,v in initial.items() if n not in active};completed={}
    for task_index,task in enumerate(tasks):
        if task_index%2!=args.shard:continue
        task_start=time.time();out=root/task;out.mkdir(exist_ok=False)
        leaf.load_state_dict(initial,strict=True);leaf.zero_grad(set_to_none=True)
        assert all(torch.equal(v.cpu(),initial[n]) for n,v in leaf.state_dict().items())
        labels,text,instruction,prototypes=prepare_text(config,task)
        assert labels==source['labels'][task] and text==source['instruction'][task]
        readout=instruction.to(device),prototypes.to(device)
        wrapper=PrototypeClassifier(leaf,*readout).to(device).eval()
        tx,ty,vx,vy=training_and_validation(task);audit=coverage['audits'][task]
        assert len(ty)==audit['train_size'] and len(vy)==audit['validation_size']
        for values,key in [(tx,'train_eeg_hash'),(ty,'train_labels_hash'),(vx,'validation_eeg_hash'),(vy,'validation_labels_hash')]:
            assert array_hash(values)==audit[key],(task,key)
        x,y=torch.from_numpy(tx),torch.from_numpy(ty);counts=torch.bincount(y,minlength=len(labels))
        assert counts.tolist()==audit['train_class_counts']
        weights=(len(y)/(len(labels)*counts.float())).to(device)
        global_positions=np.flatnonzero(schedule['task']==task_index)
        batches=schedule['local_indices'][global_positions];assert len(batches)==plan['task_steps']==100
        own_average_steps=[int((global_positions<s).sum()) for s in plan['shared_average_steps']]
        assert own_average_steps==plan['task_average_steps']
        validation=make_loader(vx,vy,config,len(labels),plan['batch_size'],False)
        probe_ids=np.concatenate([np.flatnonzero(vy==c)[:8] for c in range(len(labels))])
        probe=torch.from_numpy(np.asarray(vx[probe_ids],dtype=np.float32));probe_truth=torch.from_numpy(vy[probe_ids])
        history=[];snapshots=[]
        @torch.no_grad()
        def evaluate():
            logits=[];truth=[]
            for xx,yy in validation:logits.append(wrapper(xx.to(device)).cpu());truth.append(yy)
            return classification_metrics(torch.cat(logits),torch.cat(truth))
        def cpu_state():
            state={n:v.detach().cpu().clone() for n,v in leaf.state_dict().items()}
            assert all(torch.equal(state[n],v) for n,v in fixed.items())
            assert all(torch.isfinite(state[n]).all() for n in active)
            return state
        initial_metrics=evaluate();assert abs(initial_metrics['BAcc']-initial_run['folded_val'][task]['BAcc'])<1e-12
        history.append({'step':0,'validation':initial_metrics})
        optimizer=torch.optim.Adam([v for v in leaf.parameters() if v.requires_grad],lr=plan['lr'],weight_decay=0)
        assert not optimizer.state
        for step,ids in enumerate(batches,1):
            logits=wrapper(x[ids].to(device));truth=y[ids].to(device)
            loss=objective(logits,None,truth,weights,None,0)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([v for v in leaf.parameters() if v.requires_grad],float('inf'),error_if_nonfinite=True)
            assert torch.isfinite(loss);optimizer.step()
            if step in own_average_steps:
                path=out/f'step_{step}.ckpt';torch.save(cpu_state(),path)
                snapshots.append({'step':step,'path':str(path),'sha256':sha256(path)})
                history.append({'step':step,'validation':evaluate()})
                (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        counters=[int(s['step'].item()) for s in optimizer.state.values()]
        assert counters and all(s==100 for s in counters)
        optimizer_audit={'initialized_parameter_states':len(counters),'bias_correction_counter_values':sorted(set(counters))}
        del optimizer
        states=[torch.load(s['path'],map_location='cpu',weights_only=True) for s in snapshots]
        averaged={n:torch.stack([s[n].double() for s in states]).mean(0).float() if n in active else v.clone() for n,v in states[0].items()}
        assert all(torch.equal(averaged[n],v) for n,v in fixed.items())
        checkpoint=out/'merged.ckpt';torch.save(averaged,checkpoint);leaf.load_state_dict(averaged,strict=True)
        reloaded=load_leaf(config,checkpoint).to(device).eval()
        with torch.no_grad():
            before=wrapper(probe.to(device)).cpu()
            after=PrototypeClassifier(reloaded,*readout).to(device).eval()(probe.to(device)).cpu()
        check=compare(before,after,probe_truth)
        assert check['max_abs_logit_error']<=1e-5 and all(v['margin_within_twice_error'] for v in check['changed_rows'])
        del reloaded,states,averaged
        average_metrics=evaluate()
        with torch.no_grad():
            ids=indices[task];cal_logits=torch.cat([wrapper(x[ids[i:i+plan['batch_size']]].to(device)).cpu() for i in range(0,len(ids),plan['batch_size'])])
            calibration_metrics=classification_metrics(cal_logits,y[ids])
        result={'task':task,'task_index':task_index,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
                'updates':100,'training_examples':len(batches)*plan['batch_size'],'schedule_sha256':plan['schedule_sha256'],
                'local_batch_indices_hash':array_hash(batches),'global_step_positions_zero_based':global_positions.tolist(),
                'initial_checkpoint_sha256':plan['initial_checkpoint_sha256'],'reset_to_initial_exact':True,'optimizer_audit':optimizer_audit,
                'dataset_audit':audit,'ce_class_weights':weights.cpu().tolist(),'initial_validation':initial_metrics,
                'final_validation':history[-1]['validation'],'average_steps':own_average_steps,'average_snapshots':snapshots,
                'average_validation':average_metrics,'average_calibration':calibration_metrics,
                'checkpoint_sha256':sha256(checkpoint),'fixed_tensors_exact':True,'export_check':check,
                'seconds':time.time()-task_start,'test_data_read':False,'scope':plan['scope']}
        (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        completed[task]={'path':str(out/'run.json'),'sha256':sha256(out/'run.json')}
        (root/f'worker_{args.shard}.json').write_text(json.dumps({'status':'running','completed':completed},indent=2)+'\n')
        print(task,'complete mean BAcc',average_metrics['BAcc'],flush=True)
        del x,y,tx,ty,vx,vy,validation,wrapper,probe,probe_truth,cal_logits,logits,loss,norm
        leaf.zero_grad(set_to_none=True);gc.collect()
    (root/f'worker_{args.shard}.json').write_text(json.dumps({'status':'complete','completed':completed,'seconds':time.time()-start,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(args.gpu),'plan_sha256':sha256(args.plan)},indent=2)+'\n')


if __name__=='__main__':main()
