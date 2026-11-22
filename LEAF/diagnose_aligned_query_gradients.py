"""Initial cached prehead gradients, with fixed training weights and target controls."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from prehead_distillation_support import raw_ce_gradient,target_rows


def main():
    root=Path('runs/paper_merge_v2');plan_path=root/'inputs/aligned_query_gradient_plan.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['input_sha256'].items():assert sha256(Path(p))==h,p
    assert sha256(Path(__file__))==plan['code_sha256'];output=Path(plan['output']);assert not output.exists()
    start=time.time();torch.set_num_threads(4)
    original=json.loads(Path(plan['original_manifest']).read_text());aligned=json.loads(Path(plan['aligned_manifest']).read_text())
    scales=json.loads(Path(plan['scales']).read_text());prepared=json.loads(Path(plan['preparation']).read_text());schedule=np.load(plan['schedule'])
    records={}
    for ti,task in enumerate(original['datasets']):
        caches={}
        for kind,manifest in [('original',original),('aligned',aligned)]:
            ref=manifest['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
            caches[kind]=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        old=caches['original'];counts=stream_counts(old,schedule,ti,original['records'][task]);weights=torch.tensor(original['records'][task]['class_weights'],dtype=torch.float64)
        scale=scales['per_task'][task]['scale'];head=old['initial_weight'].double()
        accum={f'{frame}_{kind}':{'aux_energy':0.,'ce_energy':0.,'dot':0.,'negative_exposures':0.,'combined_energy':0.}
               for frame in caches for kind in ['paired','mean','shuffled']}
        for startrow in range(0,len(counts),512):
            pos=torch.arange(startrow,min(startrow+512,len(counts)));truth=old['truth'][pos];mass=counts[pos]
            h=old['initial_prehead'][pos].double();ce=raw_ce_gradient(old['initial_raw'][pos].double(),truth,old['prototypes'].double(),weights)@head
            for frame,cache in caches.items():
                for kind in ['paired','mean','shuffled']:
                    target=target_rows(cache,pos,'ce_prehead_'+kind).double();g=2*weights[truth,None]*(h-target)/h.shape[-1]
                    dot=(g*ce).sum(-1);a=accum[frame+'_'+kind]
                    a['aux_energy']+=float((g.square().sum(-1)*mass).sum());a['ce_energy']+=float((ce.square().sum(-1)*mass).sum())
                    a['dot']+=float((dot*mass).sum());a['negative_exposures']+=float(mass[dot<0].sum())
                    a['combined_energy']+=float(((ce+scale*g).square().sum(-1)*mass).sum())
        mass=float(counts.sum());stats={}
        for kind,a in accum.items():
            stats[kind]={'aux_gradient_rms':(a['aux_energy']/mass)**.5,'ce_gradient_rms':(a['ce_energy']/mass)**.5,
                         'flattened_cosine_with_ce':a['dot']/(a['aux_energy']*a['ce_energy'])**.5,
                         'negative_example_fraction':a['negative_exposures']/mass,
                         'combined_gradient_rms':(a['combined_energy']/mass)**.5,
                         'scaled_aux_ce_dot_over_ce_energy':scale*a['dot']/a['ce_energy'],
                         'fixed_aux_scale':scale}
        for kind in ['paired','mean','shuffled']:
            prior=scales['per_task'][task]['conditions']['ce_prehead_'+kind]
            assert abs(stats['original_'+kind]['aux_gradient_rms']-prior['auxiliary_prehead_gradient_rms'])<1e-10
            assert abs(stats['original_'+kind]['flattened_cosine_with_ce']-prior['flattened_gradient_cosine_with_ce'])<1e-10
        assert abs(stats['aligned_paired']['aux_gradient_rms']-prepared['records'][task]['aligned_aux_prehead_gradient_rms'])<1e-10
        records[task]=stats
    summary={kind:{metric:float(np.mean([v[kind][metric] for v in records.values()])) for metric in ['flattened_cosine_with_ce','negative_example_fraction','scaled_aux_ce_dot_over_ce_energy']} for kind in next(iter(records.values()))}
    summary['aligned_paired_ce_cosine_better_tasks']=sum(v['aligned_paired']['flattened_cosine_with_ce']>v['original_paired']['flattened_cosine_with_ce'] for v in records.values())
    summary['aligned_paired_ce_cosine_better_than_aligned_mean_tasks']=sum(v['aligned_paired']['flattened_cosine_with_ce']>v['aligned_mean']['flattened_cosine_with_ce'] for v in records.values())
    result={'unix':time.time(),'plan_sha256':sha256(plan_path),'code_sha256':sha256(Path(__file__)),
            'records':records,'summary':summary,'seconds':time.time()-start,'new_eeg_reads':0,'new_leaf_forward_examples':0,
            'optimizer_updates':0,'scope':plan['scope']}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
