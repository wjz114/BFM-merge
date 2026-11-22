"""Disposable native Adam steps for every actual task batch under a memory cap."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from d_downstream_qformer_ce import ROOT,PrototypeClassifier,load_leaf,prepare_text,sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config,load_yaml
from merge_poc import UNUSED_PREFIXES
from prehead_distillation_support import objective


def main():
    root=Path('runs/paper_merge_v2');path=root/'inputs/aligned_query_memory_plan.json';plan=json.loads(path.read_text())
    for p,h in plan['source_sha256'].items():assert sha256(Path(p))==h,p
    output=Path(plan['output']);assert not output.exists();start=time.time();device=configure(0,0)
    torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'],device)
    training=json.loads(Path(plan['training_plan']).read_text());assert sha256(Path(plan['training_plan']))==plan['training_plan_sha256']
    for p,h in training['source_sha256'].items():assert sha256(Path(p))==h,p
    source=json.loads((Path(training['source_fit'])/'run.json').read_text());coverage=json.loads(Path(training['coverage_audit']).read_text())
    targets=json.loads(Path(training['target_cache']).read_text());scales=json.loads(Path(training['scales']).read_text());schedule=np.load(training['schedule'])
    config=build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'));model=load_leaf(config,Path(training['initial_checkpoint'])).to(device).eval()
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for n,p in model.named_parameters():p.requires_grad_(not n.startswith(UNUSED_PREFIXES))
    captured={}
    def hook(module,args,output):captured['prehead']=args[0]
    handle=model.qformer.output_embed[2].register_forward_hook(hook)
    records={};status='running';attempted=0;completed=0;forward_examples=0;stage='setup'
    try:
        for ti,task in enumerate(source['datasets']):
            stage=task+':data';model.load_state_dict(state,strict=True)
            ids=schedule['local_indices'][np.flatnonzero(schedule['task']==ti)[0]]
            ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256'];cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
            positions=np.searchsorted(cache['pool_indices'].numpy(),ids);assert np.array_equal(cache['pool_indices'][positions].numpy(),ids)
            with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as file:
                rows,y=row_map(file,task,'calibration');assert array_hash(np.asarray(y,dtype=np.int64))==coverage['audits'][task]['train_labels_hash']
                x=torch.from_numpy(selected_eeg(file,[rows[int(i)] for i in ids])).to(device);truth=cache['truth'][positions].to(device)
                assert np.array_equal(y[ids],truth.cpu().numpy())
            _,_,inst,proto=prepare_text(config,task);wrapper=PrototypeClassifier(model,inst,proto).to(device).eval()
            weights=torch.tensor(ref['class_weights'],device=device);target=cache['teacher_prehead'][positions].to(device)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=training['lr'],weight_decay=0)
            torch.cuda.reset_peak_memory_stats(device);stage=task+':forward';logits=wrapper(x);forward_examples+=len(x)
            loss=objective(logits,captured['prehead'],truth,weights,target,scales['per_task'][task]['scale'])
            optimizer.zero_grad(set_to_none=True);stage=task+':backward';loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float('inf'),error_if_nonfinite=True)
            stage=task+':step';attempted+=1;optimizer.step();completed+=1;torch.cuda.synchronize(device)
            records[task]={'input_shape':list(x.shape),'loss':float(loss),'gradient_norm':float(norm),
                           'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}
            print(task,records[task],flush=True)
            del optimizer,loss,logits,target,cache,x,truth;model.zero_grad(set_to_none=True);captured.clear();torch.cuda.empty_cache()
        status='complete'
    finally:
        handle.remove();result={'unix':time.time(),'status':status,'last_stage':stage,'plan_sha256':sha256(path),
            'records':records,'attempted_disposable_adam_steps':attempted,'completed_disposable_adam_steps':completed,
            'new_forward_examples':forward_examples,'scientific_updates':0,'test_data_read':False,'seconds':time.time()-start,
            'scope':'One actual128-example frozen-schedule task batch per task at M0, native paired loss and Adam; model and optimizer reset each task. Engineering resource measurement only, no validation or candidate checkpoint.'}
        output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
