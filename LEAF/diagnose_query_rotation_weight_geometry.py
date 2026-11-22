"""Compare response rotations with the exact query-parameter Procrustes anchor."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from qformer_residual_rotation import constrained_procrustes, rotate_qformer_state


def signature(state):
    prefix='qformer.';d=state[prefix+'base_queries'].shape[-1]
    pieces=[state[prefix+'base_queries']]
    layer=0
    while prefix+f'layers.{layer}.norm1.weight' in state:
        stem=prefix+f'layers.{layer}.'
        pieces.extend([state[stem+'self_attn.in_proj_weight'],state[stem+'cross_attn.in_proj_weight'][:d]])
        for branch in ['self_attn.out_proj','cross_attn.out_proj','ff.2']:
            pieces.extend([state[stem+branch+'.weight'].T,state[stem+branch+'.bias'][None,:]])
        pieces.append(state[stem+'ff.0.weight']);layer+=1
    pieces.append(state[prefix+'output_embed.2.weight'].reshape(-1,d))
    return torch.cat(pieces)


def distance(a,b):return sum(float((a[k]-b[k]).square().sum()) for k in a)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    output=Path(plan['output']);matrix_output=Path(plan['rotation_output'])
    assert not output.exists() and not matrix_output.exists()
    started=time.time();torch.set_num_threads(4)
    source=json.loads(Path(plan['source_fit']).read_text());targets=json.loads(Path(plan['target_manifest']).read_text())
    response=torch.load(plan['response_rotations'],map_location='cpu',weights_only=True)
    schedule=np.load(plan['schedule']);eye=torch.eye(256,dtype=torch.float64)
    original0=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    state0={k:v.double() for k,v in original0.items() if k.startswith('qformer.')}
    target=rotate_qformer_state(state0,eye);target_signature=signature(target)
    gamma0,beta0=state0['qformer.final_norm.weight'],state0['qformer.final_norm.bias']
    records,matrices={},{}
    for ti,task in enumerate(plan['datasets']):
        p=Path(source['experts'][task]);assert sha256(p)==plan['expert_sha256'][task]
        original=torch.load(p,map_location='cpu',weights_only=True)
        state={k:v.double() for k,v in original.items() if k.startswith('qformer.')}
        canonical=rotate_qformer_state(state,eye);x=signature(canonical)
        rw,s=constrained_procrustes(x.T@target_signature);matrices[task]=rw
        rotations={'identity':eye,'parameter_fit':rw,**response[task]}
        baseline=distance(canonical,target);rows={};max_identity_error=0.
        for name,r in rotations.items():
            transformed=rotate_qformer_state(state,r)
            actual=distance(transformed,target)
            expected=baseline+float((x@r-target_signature).square().sum()-(x-target_signature).square().sum())
            error=abs(actual-expected)/max(actual,baseline,1e-300)
            max_identity_error=max(max_identity_error,error);assert error<1e-9
            rows[name]={'qformer_squared_distance_to_initial_merger':actual,
                        'parameter_distance_ratio_to_identity':(actual/baseline)**.5,
                        'rotation_rms_from_identity':float((r-eye).norm()/256**.5)}
        assert rows['parameter_fit']['qformer_squared_distance_to_initial_merger']<=baseline+1e-8
        ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        counts=stream_counts(cache,schedule,ti,ref);cw=torch.tensor(ref['class_weights'],dtype=torch.float64)
        gamma,beta=state['qformer.final_norm.weight'],state['qformer.final_norm.bias']
        cross=torch.zeros_like(eye);energy=mass=0.
        for start in range(0,len(counts),plan['batch_size']):
            sl=slice(start,start+plan['batch_size']);truth=cache['truth'][sl]
            xx=((cache['teacher_prehead'][sl].double().reshape(-1,8,256)-beta)/gamma).reshape(-1,256)
            yy=((cache['initial_prehead'][sl].double().reshape(-1,8,256)-beta0)/gamma0).reshape(-1,256)
            w=(counts[sl]*cw[truth]).repeat_interleave(8)
            cross+=xx.T@(yy*w[:,None]);energy+=float((w*(xx.square().sum(-1)+yy.square().sum(-1))).sum());mass+=float(w.sum())
        identity_mse=(energy-2*float((eye*cross).sum()))/(mass*256)
        for name,r in rotations.items():
            mse=(energy-2*float((r*cross).sum()))/(mass*256)
            rows[name]['all_fitting_normalized_query_mse']=mse
            rows[name]['response_error_reduction_from_identity']=1-mse/identity_mse
        records[task]={'conditions':rows,'signature_shape':list(x.shape),
                       'maximum_exact_parameter_distance_identity_relative_error':max_identity_error,
                       'parameter_cross_covariance_relative_minimum_singular_value':float(s[-1]/s[0])}
        print(task,'response-fit parameter ratio',rows['fit_all']['parameter_distance_ratio_to_identity'],
              'parameter-fit response reduction',rows['parameter_fit']['response_error_reduction_from_identity'],flush=True)
    torch.save(matrices,matrix_output)
    result={'unix':time.time(),'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'records':records,'rotation_path':str(matrix_output),'rotation_sha256':sha256(matrix_output),
            'response_fit_increases_parameter_distance_tasks':sum(v['conditions']['fit_all']['parameter_distance_ratio_to_identity']>1 for v in records.values()),
            'parameter_fit_improves_response_tasks':sum(v['conditions']['parameter_fit']['response_error_reduction_from_identity']>0 for v in records.values()),
            'mean_parameter_fit_response_error_reduction':float(np.mean([v['conditions']['parameter_fit']['response_error_reduction_from_identity'] for v in records.values()])),
            'seconds':time.time()-started,'new_eeg_reads':0,'new_model_forward_examples':0,'optimizer_updates':0,'scope':plan['scope']}
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['response_fit_increases_parameter_distance_tasks','parameter_fit_improves_response_tasks','mean_parameter_fit_response_error_reduction','seconds']}),flush=True)


if __name__=='__main__':main()
