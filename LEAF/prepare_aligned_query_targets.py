"""Re-express frozen teacher query targets in the initial student's affine frame."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from prehead_distillation_support import make_target_controls


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--design',type=Path,required=True);args=parser.parse_args()
    design=json.loads(args.design.read_text())
    for p,h in {**design['source_sha256'],**design['input_sha256']}.items():assert sha256(Path(p))==h,p
    start=time.time();torch.set_num_threads(4);out=Path(design['target_directory']);out.mkdir(exist_ok=False)
    old=json.loads(Path(design['target_manifest']).read_text());source=json.loads(Path(design['source_fit']).read_text())
    scales=json.loads(Path(design['scales']).read_text());schedule=np.load(design['schedule'])
    rotations=torch.load(design['response_rotations'],map_location='cpu',weights_only=True)
    initial=torch.load(design['initial_checkpoint'],map_location='cpu',weights_only=True)
    g0=initial['qformer.final_norm.weight'].double();b0=initial['qformer.final_norm.bias'].double();assert g0.abs().min()>1e-6
    records={};audits={}
    for ti,task in enumerate(design['datasets']):
        ref=old['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        data=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        cp=Path(source['experts'][task]);assert sha256(cp)==design['expert_sha256'][task]
        expert=torch.load(cp,map_location='cpu',weights_only=True)
        gamma=expert['qformer.final_norm.weight'].double();beta=expert['qformer.final_norm.bias'].double();assert gamma.abs().min()>1e-6
        w=expert['qformer.output_embed.2.weight'].double();bias=expert['qformer.output_embed.2.bias'].double();blocks=w.reshape(768,8,256)
        r=rotations[task]['fit_all'];assert (r.T@r-torch.eye(256,dtype=torch.float64)).abs().max()<1e-11
        mapped_w=(((blocks*gamma)@r)/g0).reshape_as(w)
        mapped_b=bias+torch.einsum('oqd,d->o',blocks,beta)-torch.einsum('oqd,d->o',mapped_w.reshape(768,8,256),b0)
        counts=stream_counts(data,schedule,ti,ref);weights=torch.tensor(ref['class_weights'],dtype=torch.float64)
        mapped=torch.empty_like(data['teacher_prehead']);exact_error=0.;rounded_error=0.;changed=0;native_cache_error=0.
        energy={'original':0.,'aligned':0.};mse={'original':0.,'aligned':0.}
        for startrow in range(0,len(counts),design['batch_size']):
            sl=slice(startrow,startrow+design['batch_size']);teacher=data['teacher_prehead'][sl].double()
            target=((((teacher.reshape(-1,8,256)-beta)/gamma)@r)*g0+b0).reshape_as(teacher)
            mapped[sl]=target.float();native=F.linear(teacher,w,bias);reexpressed=F.linear(target,mapped_w,mapped_b)
            exact_error=max(exact_error,float((native-reexpressed).abs().max()));assert exact_error<1e-10
            proto=data['prototypes'].double();before=F.normalize(native,dim=-1)@proto.T
            after=F.normalize(F.linear(mapped[sl].double(),mapped_w,mapped_b),dim=-1)@proto.T
            err=(after-before).abs().amax(-1);rounded_error=max(rounded_error,float(err.max()))
            mask=after.argmax(-1)!=before.argmax(-1);changed+=int(mask.sum())
            margin=before.topk(2).values;assert ((margin[:,0]-margin[:,1])[mask]<=2*err[mask]).all()
            native_cache_error=max(native_cache_error,float((before-data['teacher_logits'][sl].double()).abs().max()))
            truth=data['truth'][sl];cw=weights[truth];h=data['initial_prehead'][sl].double();count=counts[sl]
            for kind,t in [('original',teacher),('aligned',mapped[sl].double())]:
                diff=h-t;grad=2*cw[:,None]*diff/h.shape[-1]
                energy[kind]+=float((grad.square().sum(-1)*count).sum())
                mse[kind]+=float((diff.square().mean(-1)*cw*count).sum())
        assert rounded_error<1e-5 and native_cache_error<2e-5
        positions=torch.searchsorted(data['pool_indices'],torch.from_numpy(schedule['local_indices'][schedule['task']==ti].reshape(-1)))
        means,shuffled,shuffle_audit=make_target_controls(mapped,data['truth'],positions,len(weights),[0,ti,20261118])
        assert torch.equal(shuffled,data['shuffled_pool_positions'])
        updated={**data,'teacher_prehead':mapped,'teacher_class_means':means,
                 'aligned_teacher_weight_fp64':mapped_w,'aligned_teacher_bias_fp64':mapped_b}
        path=out/f'{task}.pt';torch.save(updated,path)
        records[task]={**ref,'target_path':str(path),'target_sha256':sha256(path),'shuffle':shuffle_audit,
                       'original_target_path':ref['target_path'],'original_target_sha256':ref['target_sha256'],
                       'transformation':'((teacher_prehead-beta_teacher)/gamma_teacher @ R_task) * gamma_M0 + beta_M0; same R for all8queries'}
        mass=float(counts.sum());old_rms=(energy['original']/mass)**.5;new_rms=(energy['aligned']/mass)**.5
        assert abs(old_rms-scales['per_task'][task]['conditions']['ce_prehead_paired']['auxiliary_prehead_gradient_rms'])<1e-10
        audits[task]={'exact_affine_raw_output_max_error':exact_error,'rounded_target_cosine_max_error':rounded_error,
                      'rounded_target_prediction_changes':changed,'native_head_vs_cache_max_error':native_cache_error,
                      'original_aux_prehead_gradient_rms':old_rms,'aligned_aux_prehead_gradient_rms':new_rms,
                      'aligned_to_original_initial_aux_gradient_rms':new_rms/old_rms,
                      'weighted_mse':{k:v/mass for k,v in mse.items()},'fixed_scale':scales['per_task'][task]['scale'],
                      'unique_cached_rows':len(counts),'fitting_exposures':int(mass)}
        print(task,'initial auxiliary gradient ratio',new_rms/old_rms,flush=True)
    manifest={**old,'records':records,'design':str(args.design),'design_sha256':sha256(args.design),
              'original_target_manifest':design['target_manifest'],'original_target_manifest_sha256':sha256(Path(design['target_manifest'])),
              'source_sha256':{str(args.design):sha256(args.design)},'target_coordinate_change':design['target_rule']}
    path=Path(design['aligned_manifest']);assert not path.exists();path.write_text(json.dumps(manifest,indent=2)+'\n')
    copied=copy.deepcopy(scales);copied.update(target_manifest=str(path),target_manifest_sha256=sha256(path),
        scope='Exact original paired-target per-task scales, copied unchanged. Original calibration statistics retained as historical values, not relabeled as aligned-target statistics. New aligned gradient magnitudes appear in the target preparation audit.',
        original_scales=design['scales'],original_scales_sha256=sha256(Path(design['scales'])))
    scale_path=Path(design['aligned_scales']);assert not scale_path.exists();scale_path.write_text(json.dumps(copied,indent=2)+'\n')
    result={'unix':time.time(),'design_sha256':sha256(args.design),'code_sha256':sha256(Path(__file__)),
            'manifest_sha256':sha256(path),'scales_sha256':sha256(scale_path),'records':audits,'seconds':time.time()-start,
            'new_eeg_reads':0,'new_leaf_forward_examples':0,'cached_linear_head_rows_checked':sum(v['unique_cached_rows'] for v in audits.values()),
            'optimizer_updates':0,'validation_used_in_statistics':False,'test_data_read':False}
    audit=Path(design['preparation_audit']);assert not audit.exists();audit.write_text(json.dumps(result,indent=2)+'\n')
    print('Completed preparation',result['seconds'],flush=True)


if __name__=='__main__':main()
