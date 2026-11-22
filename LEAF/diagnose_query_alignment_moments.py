"""Cross-fit decomposition of query alignment into class means and paired residuals."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from qformer_residual_rotation import mean_zero_basis


def nearest_identity_procrustes(cross):
    d=len(cross);b=mean_zero_basis(d);u,s,vh=torch.linalg.svd(b.T@cross@b,full_matrices=True)
    rank=int((s>s[0]*torch.finfo(s.dtype).eps*(d-1)).sum())
    r=u[:,:rank]@vh[:rank]
    if rank<d-1:
        u0,v0=u[:,rank:],vh[rank:].T
        a,_,bh=torch.linalg.svd(u0.T@v0,full_matrices=False)
        r=r+u0@(a@bh)@v0.T
    result=torch.ones(d,d,dtype=torch.float64)/d+b@r@b.T
    assert float((result.T@result-torch.eye(d,dtype=torch.float64)).abs().max())<1e-11
    optimum=float(s.sum())+float(cross.sum()/d)
    assert abs(float((result*cross).sum())-optimum)<1e-9*max(1,abs(optimum))
    return result,rank


def decompose(stats):
    mass=stats['class_mass'];assert (mass>0).all()
    mx=stats['class_x']/mass[:,None,None];my=stats['class_y']/mass[:,None,None]
    mean_cross=torch.einsum('cqd,cqe,c->de',mx,my,mass)
    mean_energy=float(((mx.square()+my.square()).sum((1,2))*mass).sum())
    return {'total':{'cross':stats['cross'],'energy':stats['energy']},
            'means':{'cross':mean_cross,'energy':mean_energy},
            'within':{'cross':stats['cross']-mean_cross,'energy':stats['energy']-mean_energy},
            'denominator':float(mass.sum())*mx.shape[1]*mx.shape[2]}


def costs(stats,r):
    values={k:(v['energy']-2*float((r*v['cross']).sum()))/stats['denominator']
            for k,v in stats.items() if k!='denominator'}
    assert abs(values['total']-values['means']-values['within'])<1e-12
    assert min(values.values())>-1e-12
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    output=Path(plan['output']);assert not output.exists();start=time.time();torch.set_num_threads(4)
    manifest=json.loads(Path(plan['target_manifest']).read_text());source=json.loads(Path(plan['source_fit']).read_text())
    schedule=np.load(plan['schedule']);initial=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    previous=json.loads(Path(plan['response_diagnostic']).read_text());eye=torch.eye(256,dtype=torch.float64)
    g0=initial['qformer.final_norm.weight'].double();b0=initial['qformer.final_norm.bias'].double()
    records={};matrices={}
    for ti,task in enumerate(plan['datasets']):
        ref=manifest['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        cp=Path(source['experts'][task]);assert sha256(cp)==plan['expert_sha256'][task]
        expert=torch.load(cp,map_location='cpu',weights_only=True);gamma=expert['qformer.final_norm.weight'].double();beta=expert['qformer.final_norm.bias'].double()
        counts=stream_counts(cache,schedule,ti,ref);weights=torch.tensor(ref['class_weights'],dtype=torch.float64);k=len(weights)
        stats=[{'cross':torch.zeros(256,256,dtype=torch.float64),'energy':0.,'class_mass':torch.zeros(k,dtype=torch.float64),
                'class_x':torch.zeros(k,8,256,dtype=torch.float64),'class_y':torch.zeros(k,8,256,dtype=torch.float64)} for _ in range(2)]
        for startrow in range(0,len(counts),plan['batch_size']):
            sl=slice(startrow,startrow+plan['batch_size']);truth=cache['truth'][sl];halves=cache['pool_indices'][sl].remainder(2)
            x=(cache['teacher_prehead'][sl].double().reshape(-1,8,256)-beta)/gamma
            y=(cache['initial_prehead'][sl].double().reshape(-1,8,256)-b0)/g0
            mass=counts[sl]*weights[truth]
            for half in range(2):
                mask=(halves==half)&(mass>0);xx,yy=x[mask],y[mask];w=mass[mask];labels=truth[mask];s=stats[half]
                s['cross']+=xx.reshape(-1,256).T@(yy*w[:,None,None]).reshape(-1,256)
                s['energy']+=float(((xx.square()+yy.square()).sum((1,2))*w).sum())
                s['class_mass'].index_add_(0,labels,w)
                s['class_x'].index_add_(0,labels,xx*w[:,None,None]);s['class_y'].index_add_(0,labels,yy*w[:,None,None])
        decomposed=[decompose(s) for s in stats];folds={};task_matrices={}
        for fit_half in range(2):
            train,test=decomposed[fit_half],decomposed[1-fit_half];identity=costs(test,eye);conditions={}
            for mode in ['total','means','within']:
                r,rank=nearest_identity_procrustes(train[mode]['cross']);actual=costs(test,r)
                assert costs(train,r)[mode]<=costs(train,eye)[mode]+1e-10
                if mode=='total':
                    old=previous['records'][task]['folds'][str(fit_half)]
                    assert abs(actual['total']-old['heldout_rotated_mse'])<1e-8
                conditions[mode]={'heldout_mse':actual,'heldout_gain':{k:identity[k]-actual[k] for k in identity},
                                  'heldout_relative_error_reduction':{k:1-actual[k]/identity[k] for k in identity},
                                  'fit_mean_zero_cross_rank':rank,'rotation_rms_distance_from_identity':float((r-eye).norm()/16)}
                task_matrices[f'{mode}_fit_half_{fit_half}']=r
            folds[str(fit_half)]={'identity_heldout_mse':identity,'conditions':conditions}
        records[task]={'folds':folds,'target_sha256':ref['target_sha256'],'class_masses':[s['class_mass'].tolist() for s in stats]}
        matrices[task]=task_matrices
        print(task,[folds[str(h)]['conditions']['total']['heldout_relative_error_reduction']['within'] for h in range(2)],flush=True)
    summary={}
    for mode in ['total','means','within']:
        folds=[f['conditions'][mode] for t in records.values() for f in t['folds'].values()]
        summary[mode]={k:{'positive_folds':sum(f['heldout_gain'][k]>0 for f in folds),
                          'mean_relative_error_reduction':float(np.mean([f['heldout_relative_error_reduction'][k] for f in folds])),
                          'mean_absolute_error_reduction':float(np.mean([f['heldout_gain'][k] for f in folds]))} for k in ['total','means','within']}
    summary['total_fit_mean_gain_fraction_from_class_means']=summary['total']['means']['mean_absolute_error_reduction']/summary['total']['total']['mean_absolute_error_reduction']
    summary['total_vs_means_fit_within_mse_better_folds']=sum(f['conditions']['total']['heldout_mse']['within']<f['conditions']['means']['heldout_mse']['within'] for t in records.values() for f in t['folds'].values())
    rotations=Path(plan['rotation_output']);assert not rotations.exists();torch.save(matrices,rotations)
    result={'unix':time.time(),'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'records':records,'summary':summary,'rotation_path':str(rotations),'rotation_sha256':sha256(rotations),
            'seconds':time.time()-start,'new_eeg_reads':0,'new_model_forward_examples':0,'optimizer_updates':0,'scope':plan['scope']}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
