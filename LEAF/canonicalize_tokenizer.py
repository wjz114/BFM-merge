"""Exact common-temporal-basis diagnostic for LEAF's consecutive convolutions.

This is an operator-space diagnostic related to SLOA/GeoMerge, not a novelty
claim. It changes expert parameter coordinates while preserving their function.
The pretrained base itself is unchanged.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, make_loader, prepare_text, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import stratified_indices


PREFIX = 'tower.tokenizer.enc_conv.'
KEYS = [PREFIX + suffix for suffix in ['0.weight','0.bias','1.weight','1.bias']]


def factors(state):
    a = state[KEYS[0]][:,0,0,:].double()
    b = state[KEYS[2]][:,:,:,0].permute(0,2,1).flatten(0,1).double()
    bias_a, bias_b = state[KEYS[1]].double(), state[KEYS[3]].double()
    return a,b,bias_a,bias_b


def canonical_parameters(state,base):
    a0,b0,bias_a0,_ = factors(base)
    a,b,bias_a,bias_b = factors(state)
    inverse = torch.linalg.pinv(a0,rtol=1e-12)
    assert torch.linalg.matrix_rank(a0)==a0.shape[1]
    kernel,base_kernel = b@a,b0@a0
    new_b = b0+(kernel-base_kernel)@inverse
    out_channels,_,channels,_ = state[KEYS[2]].shape
    old_bias = bias_b+(b@bias_a).reshape(out_channels,channels).sum(-1)
    new_bias_b = old_bias-(new_b@bias_a0).reshape(out_channels,channels).sum(-1)
    values = {
        KEYS[0]:base[KEYS[0]].clone(),KEYS[1]:base[KEYS[1]].clone(),
        KEYS[2]:new_b.reshape(out_channels,channels,-1).permute(0,2,1).unsqueeze(-1).float(),
        KEYS[3]:new_bias_b.float(),
    }
    actual_a,actual_b,actual_ba,actual_bb = factors(values)
    actual_kernel = actual_b@actual_a
    actual_bias = actual_bb+(actual_b@actual_ba).reshape(out_channels,channels).sum(-1)
    info = {'relative_kernel_error_after_fp32_export':float((actual_kernel-kernel).norm()/kernel.norm()),
            'max_effective_bias_error':float((actual_bias-old_bias).abs().max()),
            'original_spatial_delta_norm':float((b-b0).norm()),
            'canonical_spatial_delta_norm':float((actual_b-b0).norm()),
            'temporal_base_rank':int(torch.linalg.matrix_rank(a0)),
            'temporal_base_condition':float(torch.linalg.cond(a0))}
    return values,info


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Original completed merge run supplies expert order and fixed calibration IDs')
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    source = json.loads((args.source/'run.json').read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config,Path(source['base_checkpoint'])).to(device).eval()
    base = {name:value.detach().clone() for name,value in leaf.state_dict().items()}
    # Architecture must have exactly the two consecutive affine convolutions.
    layers = leaf.tower.tokenizer.enc_conv
    assert isinstance(layers[0],torch.nn.Conv2d) and isinstance(layers[1],torch.nn.Conv2d)
    assert layers[0].groups==layers[1].groups==1
    assert layers[0].kernel_size==(1,40) and layers[1].kernel_size==(65,1)
    assert layers[0].padding==(0,20) and layers[1].padding==(0,0)
    calibration = np.load(args.source/'calibration_indices.npz')
    records = {}
    for index,(task,path) in enumerate(source['experts'].items()):
        expert_started = time.time()
        original = torch.load(path,map_location=device,weights_only=True)
        values,geometry = canonical_parameters(original,base)
        state = {name:value.clone() for name,value in original.items()}
        state.update(values)
        assert all(torch.equal(state[name],original[name]) for name in state if name not in KEYS)
        assert all(torch.isfinite(state[name]).all() for name in KEYS)
        assert geometry['relative_kernel_error_after_fp32_export']<1e-6,geometry
        labels,_,instruction,prototypes = prepare_text(config,task)
        wrapper = PrototypeClassifier(leaf,instruction,prototypes).to(device).eval()
        _,train_x,train_y,val_x,val_y,test_x,test_y = load_dataset(task)
        val_indices = stratified_indices(val_y,512,np.random.default_rng([0,index,713]))
        x = np.concatenate([train_x[calibration[task]],val_x[val_indices]])
        y = np.concatenate([train_y[calibration[task]],val_y[val_indices]])
        del train_x,train_y,val_x,val_y,test_x,test_y
        loader = make_loader(x,y,config,len(labels),128,False)
        before,after,truth = [],[],[]
        leaf.load_state_dict(original,strict=True)
        for xx,yy in loader:
            before.append(wrapper(xx.to(device)).cpu());truth.append(yy)
        leaf.load_state_dict(state,strict=True)
        for xx,_ in loader:
            after.append(wrapper(xx.to(device)).cpu())
        checks = compare(torch.cat(before),torch.cat(after),torch.cat(truth))
        assert checks['max_abs_logit_error']<=1e-5,checks
        assert all(row['margin_within_twice_error'] for row in checks['changed_rows'])
        folder = args.outdir/task
        folder.mkdir()
        torch.save({name:value.cpu() for name,value in state.items()},folder/'best.ckpt')
        metadata = json.loads(Path(path).with_name('run.json').read_text())
        provenance = {'source_expert':str(path),'source_sha256':sha256(Path(path)),
            'canonical_sha256':sha256(folder/'best.ckpt'),'changed_tensors':KEYS,
            'geometry':geometry,'function_check':checks,'calibration_examples':len(calibration[task]),
            'validation_examples':len(val_indices),'validation_indices':val_indices.tolist(),
            'seconds':time.time()-expert_started}
        metadata['operator_canonicalization'] = provenance
        metadata['original_training_metrics_note'] = 'Original expert training metrics retained as provenance. New function-preservation checks are in operator_canonicalization.'
        (folder/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        records[task] = provenance
        (args.outdir/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(task,'logit error',checks['max_abs_logit_error'],flush=True)
    result = {'diagnostic_only':True,'source_run':str(args.source),'records':records,
        'base_checkpoint_unchanged':source['base_checkpoint'],'base_sha256':sha256(Path(source['base_checkpoint'])),
        'calibration_indices_sha256':sha256(args.source/'calibration_indices.npz'),
        'code_sha256':sha256(Path(__file__)),'seconds':time.time()-start,
        'test_inference_performed':False,'mechanism_claim':'Exact local operator preservation under a common temporal basis; no claim that this explains the observed Joint gap.',
        'closest_prior_work':['https://arxiv.org/html/2610.06415v1','https://arxiv.org/html/2604.27155v1']}
    (args.outdir/'run.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
