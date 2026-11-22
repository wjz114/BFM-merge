"""Fixed-coefficient query-coordinate merge counterfactuals with native export checks."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from qformer_residual_rotation import rotate_qformer_state


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    out=Path(plan['outdir']);out.mkdir(exist_ok=False)
    started=time.time();device=configure(args.gpu,plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'],device)
    source=json.loads(Path(plan['source_fit']).read_text());coverage=json.loads(Path(plan['coverage_audit']).read_text())
    targets=json.loads(Path(plan['target_manifest']).read_text());initial_run=json.loads(Path(plan['initial_run']).read_text())
    calibration=np.load(plan['calibration']);alpha=torch.from_numpy(np.load(plan['coefficients'])).to(device)
    assert alpha.shape==(16,len(source['groups']))
    config=build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    initial=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    response=torch.load(plan['response_rotations'],map_location='cpu',weights_only=True)
    parameter=torch.load(plan['parameter_rotations'],map_location='cpu',weights_only=True)
    eye=torch.eye(256,dtype=torch.float64)
    transforms={t:{'canonical':eye,'response':response[t]['fit_all'],'parameter':parameter[t]} for t in plan['datasets']}
    qnames=[n for n in source['groups'] if n.startswith('qformer.')]
    base_q={k:v for k,v in base.items() if k.startswith('qformer.')}
    base_canonical=rotate_qformer_state(base_q,eye)
    transformed_experts={k:[] for k in plan['conditions']};original_experts=[]
    for task in plan['datasets']:
        path=Path(source['experts'][task]);assert sha256(path)==plan['expert_sha256'][task]
        expert=torch.load(path,map_location='cpu',weights_only=True)
        q={k:v for k,v in expert.items() if k.startswith('qformer.')};original_experts.append(q)
        for kind in plan['conditions']:transformed_experts[kind].append(rotate_qformer_state(q,transforms[task][kind]))
    replay_error=0.
    for n in qnames:
        values=torch.stack([e[n]-base_q[n] for e in original_experts]).to(device)
        reconstructed=base_q[n].to(device)+torch.tensordot(alpha[:,source['groups'].index(n)],values,dims=1)
        replay_error=max(replay_error,float((reconstructed.cpu()-initial[n]).abs().max()))
    assert replay_error<plan['original_composition_replay_atol']
    models={};checkpoints={}
    for kind in plan['conditions']:
        merged={k:v.clone() for k,v in initial.items()}
        for n in qnames:
            values=torch.stack([e[n]-base_canonical[n] for e in transformed_experts[kind]]).to(device)
            merged[n]=(base_canonical[n].to(device)+torch.tensordot(alpha[:,source['groups'].index(n)],values,dims=1)).cpu()
        assert all(torch.equal(v,initial[n]) for n,v in merged.items() if not n.startswith('qformer.'))
        assert list(merged)==list(initial) and all(torch.isfinite(merged[n]).all() for n in qnames)
        # The unchanged native tower causal mask intentionally contains -inf.
        folder=out/kind;folder.mkdir();cp=folder/'merged.ckpt';torch.save(merged,cp)
        model=load_leaf(config,cp).to(device).eval().requires_grad_(False)
        assert all(torch.equal(v.cpu(),merged[n]) for n,v in model.state_dict().items())
        models[kind]=model;checkpoints[kind]={'path':str(cp),'sha256':sha256(cp),'native_tensor_count':len(merged),'state_reload_exact':True,'non_qformer_tensors_exact_initial':True}
    # Check transformed source experts against cached native logits on original512.
    expert_model=load_leaf(config,Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    source_checks={};native_replay_checks={};forward_examples={'source_equivalence':0,'candidate_validation':0}
    readouts={}
    for ti,task in enumerate(plan['datasets']):
        labels,text,instruction,prototypes=prepare_text(config,task)
        assert labels==source['labels'][task] and text==source['instruction'][task]
        ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        assert array_hash(instruction)==ref['instruction_hash'] and array_hash(prototypes)==ref['prototypes_hash']
        readouts[task]=(instruction,prototypes)
        cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        ids=calibration[task];positions=np.searchsorted(cache['pool_indices'].numpy(),ids)
        assert np.array_equal(cache['pool_indices'][positions].numpy(),ids)
        with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as handle:
            rows,y=row_map(handle,task,'calibration')
            assert array_hash(np.asarray(y,dtype=np.int64))==coverage['audits'][task]['train_labels_hash']
            x=torch.from_numpy(selected_eeg(handle,[rows[int(i)] for i in ids]));truth=torch.from_numpy(np.asarray(y[ids],dtype=np.int64))
        assert array_hash(x)==plan['original512_eeg_hash'][task]
        assert torch.equal(truth,cache['truth'][positions])
        original=torch.load(source['experts'][task],map_location='cpu',weights_only=True)
        expert_model.load_state_dict(original,strict=True)
        native_wrapper=PrototypeClassifier(expert_model,instruction,prototypes).to(device).eval()
        native_logits=torch.cat([native_wrapper(x[i:i+plan['batch_size']].to(device)).cpu() for i in range(0,len(x),plan['batch_size'])])
        replay=compare(cache['teacher_logits'][positions],native_logits,truth)
        assert replay['max_abs_logit_error']<plan['cached_native_replay_atol'],(task,replay)
        assert all(row['margin_within_twice_error'] for row in replay['changed_rows'])
        native_replay_checks[task]=replay;forward_examples['source_equivalence']+=len(x)
        source_checks[task]={}
        for kind in plan['conditions']:
            state={**original,**transformed_experts[kind][ti]};expert_model.load_state_dict(state,strict=True)
            wrapper=PrototypeClassifier(expert_model,instruction,prototypes).to(device).eval()
            logits=torch.cat([wrapper(x[i:i+plan['batch_size']].to(device)).cpu() for i in range(0,len(x),plan['batch_size'])])
            check=compare(native_logits,logits,truth)
            assert check['max_abs_logit_error']<plan['source_logit_equivalence_atol'],(task,kind,check)
            assert all(row['margin_within_twice_error'] for row in check['changed_rows'])
            source_checks[task][kind]=check;forward_examples['source_equivalence']+=len(x)
        print('source equivalence passed',task,flush=True)
        (out/'source_equivalence_progress.json').write_text(json.dumps(source_checks,indent=2)+'\n')
    del expert_model,transformed_experts,original_experts
    (out/'source_equivalence.json').write_text(json.dumps(source_checks,indent=2)+'\n')
    # Source equivalence for all experts must finish before any candidate validation.
    metrics={k:{} for k in plan['conditions']};data_audits={}
    for task in plan['datasets']:
        with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as handle:
            rows,truth=row_map(handle,task,'validation')
            x=selected_eeg(handle,rows);truth=np.asarray(truth,dtype=np.int64)
        audit=coverage['audits'][task]
        assert array_hash(x)==audit['validation_eeg_hash'] and array_hash(truth)==audit['validation_labels_hash']
        data_audits[task]={'rows':len(x),'eeg_hash':array_hash(x),'truth_hash':array_hash(truth)}
        x=torch.from_numpy(x);truth=torch.from_numpy(truth)
        for kind,model in models.items():
            wrapper=PrototypeClassifier(model,*readouts[task]).to(device).eval()
            logits=torch.cat([wrapper(x[i:i+plan['batch_size']].to(device)).cpu() for i in range(0,len(x),plan['batch_size'])])
            metrics[kind][task]=classification_metrics(logits,truth)
            forward_examples['candidate_validation']+=len(x)
        (out/'progress.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print('validation',task,{k:metrics[k][task]['BAcc'] for k in plan['conditions']},flush=True)
    assert forward_examples==plan['forward_budget'],forward_examples
    result={'status':'complete','unix':time.time(),'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'checkpoints':checkpoints,'cached_native_replay':native_replay_checks,'source_equivalence':source_checks,'source_equivalence_sha256':sha256(out/'source_equivalence.json'),
            'original_qformer_coefficient_composition_max_error':replay_error,
            'validation':metrics,'macro_bacc':{k:float(np.mean([v['BAcc'] for v in values.values()])) for k,values in metrics.items()},
            'archived_initial_macro_bacc':initial_run['folded_val_macro_BAcc'],'dataset_audits':data_audits,
            'new_forward_examples':forward_examples,'optimizer_updates':0,'test_data_read':False,'seconds':time.time()-started,'scope':plan['scope']}
    result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(device)
    (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed',result['macro_bacc'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
