"""Real-EEG chain-rule and ordinary-export checks for the augmented coefficient basis."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from gradient_basis_support import AugmentedResidualTensorMerger
from coverage_support import training_and_validation
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, sha256
from load_config import build_model_config, load_yaml
from merge_poc import stratified_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--basis-kind',choices=['gradient','random'],required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    initial = ROOT/'runs/paper_merge_v2/seed0/original_fp32_tensor_balanced_avg/merged.ckpt'
    source_path = ROOT/'runs/paper_merge_v2/seed0/original_fp32_tensor_balanced_ce/run.json'
    source = json.loads(source_path.read_text())
    coverage = json.loads((ROOT/'runs/paper_merge_v2/inputs/full_control_coverage_data/run.json').read_text())
    leaf = load_leaf(config,initial).eval().requires_grad_(False)
    basis_path = ROOT/'runs/paper_merge_v2/seed0/local_gradient_basis/run.json'
    basis = json.loads(basis_path.read_text())
    output = basis['outputs'][args.basis_kind]
    merger = AugmentedResidualTensorMerger(leaf,source,output['path'],output['sha256'],basis['geometry'])
    zero = merger.materialize()
    assert all(torch.equal(value,zero[name]) for name,value in leaf.state_dict().items())
    indices = np.load(source['calibration_input'])
    batches,initial_checks = {},{}
    for task in ['MI_OpenBMI','EMO_FACED']:
        labels,_,instruction,prototypes = prepare_text(config,task)
        wrapper = PrototypeClassifier(leaf,instruction,prototypes).eval()
        tx,ty,vx,vy = training_and_validation(task)
        assert array_hash(tx[indices[task]]) == coverage['audits'][task]['calibration_eeg_hash']
        ids = indices[task][stratified_indices(ty[indices[task]],16,np.random.default_rng(0))]
        x,y = torch.from_numpy(tx[ids].copy()),torch.from_numpy(ty[ids].copy())
        counts = torch.bincount(torch.from_numpy(ty),minlength=len(labels))
        weights = len(ty)/(len(labels)*counts.float())
        with torch.no_grad():
            direct = wrapper(x)
            functional = merger.logits(wrapper,x)
        assert torch.equal(direct,functional)
        initial_checks[task] = {'exact_logits':True,'eeg_shape':list(x.shape),'class_counts':torch.bincount(y,minlength=len(labels)).tolist()}
        batches[task] = (wrapper,x,y,weights)
        del tx,ty,vx,vy
    optimizer = torch.optim.Adam([merger.beta],lr=.001,weight_decay=0)
    states,betas,gradient_checks = [],[],{}
    for task,(wrapper,x,y,weights) in batches.items():
        params = merger.compose()
        logits = merger.logits(wrapper,x,params)
        loss = (F.cross_entropy(logits,y,reduction='none')*weights[y]).mean()
        gradients = torch.autograd.grad(loss,[merger.beta,*[params[name] for name in merger.active]])
        actual = gradients[0]
        expected = torch.stack([torch.tensordot(merger.deltas[name],g,dims=(tuple(range(1,g.ndim+1)),tuple(range(g.ndim))))
                                for name,g in zip(merger.active,gradients[1:])],dim=1)
        maximum = float((actual-expected).abs().max())
        relative = float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
        assert torch.allclose(actual,expected,rtol=5e-5,atol=1e-6),(task,maximum,relative)
        assert torch.isfinite(actual).all() and actual.norm()>0
        assert actual[:16].norm()>0 and actual[16:].norm()>0
        gradient_checks[task] = {'loss':float(loss.detach()),'gradient_norm':float(actual.norm()),
                                'chain_rule_max_abs_error':maximum,'chain_rule_relative_l2_error':relative,
                                'original_basis_gradient_norm':float(actual[:16].norm()),
                                'added_basis_gradient_norm':float(actual[16:].norm()),
                                'component_gradient_norms':{p:float(actual[:,cols].norm()) for p,cols in merger.component_columns.items()}}
        assert all(value>0 for value in gradient_checks[task]['component_gradient_norms'].values())
        optimizer.zero_grad(set_to_none=True)
        merger.beta.grad = actual.detach()
        optimizer.step()
        assert torch.isfinite(merger.beta).all()
        states.append(merger.materialize())
        betas.append(merger.beta.detach().clone())
        print('checked',task,gradient_checks[task],flush=True)
        del params,logits,loss,gradients,actual,expected
    assert all(torch.equal(value,merger.initial[name]) for name,value in leaf.state_dict().items())
    average = {name:torch.stack([state[name].double() for state in states]).mean(0).float()
               if name in merger.active else value.clone() for name,value in states[0].items()}
    mean_beta = torch.stack([value.double() for value in betas]).mean(0).float()
    beta_fold = merger.materialize(mean_beta)
    rounding = max(float((average[name]-beta_fold[name]).abs().max()) for name in merger.active)
    assert rounding <= 1e-5
    checkpoint = args.outdir/'merged.ckpt'
    torch.save(average,checkpoint)
    reloaded = load_leaf(config,checkpoint).eval().requires_grad_(False)
    in_memory = copy.deepcopy(leaf)
    in_memory.load_state_dict(average,strict=True)
    export_checks,rounding_checks = {},{}
    with torch.no_grad():
        final_params = merger.compose(mean_beta)
        for task,(wrapper,x,y,_) in batches.items():
            exported = PrototypeClassifier(reloaded,*prepare_text(config,task)[2:]).eval()(x)
            folded = PrototypeClassifier(in_memory,*prepare_text(config,task)[2:]).eval()(x)
            export_checks[task] = compare(exported,folded,y)
            assert torch.equal(exported,folded)
            functional = merger.logits(wrapper,x,final_params)
            rounding_checks[task] = compare(functional,exported,y)
            assert rounding_checks[task]['max_abs_logit_error'] <= 1e-5
            assert all(row['margin_within_twice_error'] for row in rounding_checks[task]['changed_rows'])
    result = {'passed':True,'engineering_only':True,'initial_model_state_exact':True,'initial_probe_checks':initial_checks,
              'coefficient_shape':list(merger.beta.shape),'coefficient_count':merger.beta.numel(),
              'basis_kind':args.basis_kind,'basis_run_sha256':sha256(basis_path),'basis_artifact':merger.basis_info,
              'active_parameter_tensors':len(merger.active),'gradient_checks':gradient_checks,
              'fixed_state_exact':True,'functional_call_restores_template':True,
              'mean_coefficient_vs_mean_state_max_abs':rounding,'export_checks':export_checks,'rounding_checks':rounding_checks,
              'source_fit_sha256':sha256(source_path),'initial_checkpoint_sha256':sha256(initial),
              'checkpoint_sha256':sha256(checkpoint),'seconds':time.time()-started,
              'source_sha256':{p:sha256(ROOT/p) for p in ['check_gradient_basis_control.py','gradient_basis_support.py','coefficient_control_support.py','train_gradient_basis_control.py']},
              'test_data_read':False,'test_inference_performed':False,
              'scope':'CPU FP32,16 actual EEG examples each from two tasks with800/2000 time samples. Verify beta-gradient equals inner product of each original or added basis vector with the full-parameter loss gradient, then actual Adam updates, fixed-state preservation and ordinary export. Not performance evidence.'}
    (args.outdir/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Complete engineering coefficient check',result['seconds'],flush=True)


if __name__ == '__main__':
    main()
