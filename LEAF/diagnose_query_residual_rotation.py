"""Cross-fit LayerNorm-compatible query rotations using existing training caches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from qformer_residual_rotation import constrained_procrustes, toy_checks


def cost(stats,rotation):
    squared=stats['energy']-2*float((rotation*stats['cross']).sum())
    return squared/stats['mass']/rotation.shape[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    output=Path(plan['output']);assert not output.exists()
    matrices_path=Path(plan['rotation_output']);assert not matrices_path.exists()
    started=time.time();torch.set_num_threads(4);toy=toy_checks()
    targets=json.loads(Path(plan['target_manifest']).read_text())
    source=json.loads(Path(plan['source_fit']).read_text())
    schedule=np.load(plan['schedule'])
    initial=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    gamma0=initial['qformer.final_norm.weight'].double();beta0=initial['qformer.final_norm.bias'].double()
    assert float(gamma0.abs().min())>1e-6
    records,rotations={},{}
    d=plan['query_dimension'];nq=plan['query_count'];eye=torch.eye(d,dtype=torch.float64)
    for ti,task in enumerate(plan['datasets']):
        ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        expert_path=Path(source['experts'][task]);assert sha256(expert_path)==plan['expert_sha256'][task]
        expert=torch.load(expert_path,map_location='cpu',weights_only=True)
        gamma=expert['qformer.final_norm.weight'].double();beta=expert['qformer.final_norm.bias'].double()
        assert float(gamma.abs().min())>1e-6
        counts=stream_counts(cache,schedule,ti,ref)
        weights=torch.tensor(ref['class_weights'],dtype=torch.float64)
        halves=cache['pool_indices'].remainder(2)
        stats=[{'cross':torch.zeros(d,d,dtype=torch.float64),'energy':0.,'mass':0.,'unique_rows':0,'exposures':0,'class_counts':torch.zeros(len(weights),dtype=torch.int64)} for _ in range(2)]
        maximum_row_mean=0.
        for start in range(0,len(counts),plan['batch_size']):
            sl=slice(start,start+plan['batch_size']);truth=cache['truth'][sl]
            x=(cache['teacher_prehead'][sl].double().reshape(-1,nq,d)-beta)/gamma
            y=(cache['initial_prehead'][sl].double().reshape(-1,nq,d)-beta0)/gamma0
            maximum_row_mean=max(maximum_row_mean,float(x.mean(-1).abs().max()),float(y.mean(-1).abs().max()))
            mass=counts[sl]*weights[truth]
            for half in range(2):
                mask=(halves[sl]==half)&(counts[sl]>0)
                xx,yy=x[mask].reshape(-1,d),y[mask].reshape(-1,d)
                w=mass[mask].repeat_interleave(nq)
                stats[half]['cross']+=xx.T@(yy*w[:,None])
                stats[half]['energy']+=float((w*(xx.square().sum(-1)+yy.square().sum(-1))).sum())
                stats[half]['mass']+=float(w.sum())
                stats[half]['unique_rows']+=int(mask.sum())
                stats[half]['exposures']+=int(counts[sl][mask].sum())
                stats[half]['class_counts']+=torch.bincount(truth[mask],minlength=len(weights))
        assert maximum_row_mean<plan['normalized_row_mean_tolerance']
        fits={};task_matrices={}
        for fit_half in range(2):
            heldout=1-fit_half;r,s=constrained_procrustes(stats[fit_half]['cross'])
            assert float((r.T@r-eye).abs().max())<1e-11
            fit_id,fit_rot=cost(stats[fit_half],eye),cost(stats[fit_half],r)
            test_id,test_rot=cost(stats[heldout],eye),cost(stats[heldout],r)
            assert fit_rot<=fit_id+1e-10
            fits[str(fit_half)]={'fit_identity_mse':fit_id,'fit_rotated_mse':fit_rot,
                                 'heldout_identity_mse':test_id,'heldout_rotated_mse':test_rot,
                                 'heldout_relative_error_reduction':1-test_rot/test_id,
                                 'rotation_rms_distance_from_identity':float((r-eye).norm()/d**.5),
                                 'cross_covariance_relative_minimum_singular_value':float(s[-1]/s[0]),
                                 'fit_unique_rows':stats[fit_half]['unique_rows'],'heldout_unique_rows':stats[heldout]['unique_rows'],
                                 'fit_class_counts':stats[fit_half]['class_counts'].tolist(),'heldout_class_counts':stats[heldout]['class_counts'].tolist()}
            task_matrices[f'fit_half_{fit_half}']=r
        total={'cross':sum(t['cross'] for t in stats),'energy':sum(t['energy'] for t in stats),'mass':sum(t['mass'] for t in stats)}
        full,s=constrained_procrustes(total['cross']);task_matrices['fit_all']=full
        records[task]={'folds':fits,'all_identity_mse':cost(total,eye),'all_rotated_mse':cost(total,full),
                       'fold_rotation_rms_difference':float((task_matrices['fit_half_0']-task_matrices['fit_half_1']).norm()/d**.5),
                       'maximum_recovered_normalized_row_mean':maximum_row_mean,
                       'expert_checkpoint_sha256':plan['expert_sha256'][task],
                       'target_sha256':ref['target_sha256'],
                       'fitting_exposures':sum(t['exposures'] for t in stats),
                       'unique_fitting_rows':sum(t['unique_rows'] for t in stats)}
        rotations[task]=task_matrices
        print(task,[fits[str(h)]['heldout_relative_error_reduction'] for h in range(2)],flush=True)
    torch.save(rotations,matrices_path)
    reductions=[fold['heldout_relative_error_reduction'] for r in records.values() for fold in r['folds'].values()]
    result={'unix':time.time(),'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'toy_checks':toy,'records':records,'rotation_path':str(matrices_path),'rotation_sha256':sha256(matrices_path),
            'heldout_positive_folds':sum(x>0 for x in reductions),'fold_count':len(reductions),
            'mean_heldout_relative_error_reduction':float(np.mean(reductions)),
            'minimum_heldout_relative_error_reduction':min(reductions),'seconds':time.time()-started,
            'new_eeg_reads':0,'new_eeg_forward_examples':0,'model_optimizer_updates':0,'scope':plan['scope']}
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['heldout_positive_folds','fold_count','mean_heldout_relative_error_reduction','minimum_heldout_relative_error_reduction','seconds']}),flush=True)


if __name__=='__main__':main()
