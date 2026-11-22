"""Separate global-null calibration from other-task decision components of Adam."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from diagnose_adam_decision_null import stream_counts
from diagnose_contrast_normalizer_gradients import geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    output=Path(plan['output']);assert not output.exists()
    torch.set_num_threads(4);started=time.time()
    targets=json.loads(Path(plan['target_manifest']).read_text())
    schedule=np.load(plan['schedule'])
    bases=torch.load(plan['bases'],map_location='cpu',weights_only=True)
    _,s,vh=torch.linalg.svd(torch.cat(list(bases.values())),full_matrices=False)
    union=vh[s>s[0]*1e-8]
    gradients=[];updates={k:[] for k in ['full','task','global','selective']};records={}
    for ti,task in enumerate(plan['datasets']):
        ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
        cache=torch.load(ref['target_path'],map_location='cpu',weights_only=True)
        p=cache['prototypes'].double();basis=bases[task]
        counts=stream_counts(cache,schedule,ti,ref)
        weights=torch.tensor(ref['class_weights'],dtype=torch.float64)
        ghead=torch.zeros(768,2049,dtype=torch.float64)
        for start in range(0,len(counts),plan['batch_size']):
            sl=slice(start,start+plan['batch_size']);y=cache['truth'][sl]
            g,*_=geometry(cache['initial_raw'][sl].double(),p,y,basis)
            weighted=g*(counts[sl]*weights[y])[:,None]
            ghead[:,:-1]+=weighted.T@cache['initial_prehead'][sl].double()
            ghead[:,-1]+=weighted.sum(0)
        ghead/=counts.sum();gradients.append(ghead.flatten())
        initial=torch.cat([cache['initial_weight'],cache['initial_bias'][:,None]],dim=1)
        param=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([param],**plan['adam'])
        param.grad=ghead.float();opt.step()
        u=param.detach().double()-initial.double()
        pt=basis.T@(basis@u);pu=union.T@(union@u)
        ideal={'full':u,'task':pt,'global':pu,'selective':u-pu+pt}
        applied={k:(initial.double()+v).float().double()-initial.double() for k,v in ideal.items()}
        for k,v in applied.items():updates[k].append(v.flatten())
        norm2=float(u.square().sum())
        records[task]={'task_contrast_energy_fraction':float(pt.square().sum())/norm2,
                       'other_task_contrast_energy_fraction':float((pu-pt).square().sum())/norm2,
                       'global_null_energy_fraction':float((u-pu).square().sum())/norm2,
                       'global_projection_exact_all_contrast_relative_error':float((union@(pu-u)).norm()/(union@u).norm()),
                       'selective_vs_task_exact_all_contrast_relative_error':float((union@(ideal['selective']-pt)).norm()/(union@pt).norm()),
                       'selective_source_contrast_rounding_relative_error':float((basis@(applied['selective']-u)).norm()/(basis@u).norm())}
        assert records[task]['global_projection_exact_all_contrast_relative_error']<1e-11
        assert records[task]['selective_vs_task_exact_all_contrast_relative_error']<1e-11
        print(task,records[task]['global_null_energy_fraction'],flush=True)
    g=torch.stack(gradients);off=~torch.eye(len(g),dtype=torch.bool)
    matrices={k:g@torch.stack(v).T for k,v in updates.items()}
    summary={}
    for k,matrix in matrices.items():
        macro=matrix.mean(0);full=matrices['full'].mean(0);task=matrices['task'].mean(0)
        summary[k]={'macro_ce_first_order_change_mean':float(macro.mean()),
                    'macro_ce_first_order_change_by_source':dict(zip(plan['datasets'],macro.tolist())),
                    'negative_transfer_pairs_original_ce':int((matrix[off]>0).sum()),
                    'source_tasks_better_macro_ce_than_full':int((macro<full).sum()),
                    'source_tasks_better_macro_ce_than_task_projection':int((macro<task).sum()),
                    'source_own_ce_first_order_changes':dict(zip(plan['datasets'],matrix.diagonal().tolist()))}
    result={'unix':time.time(),'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'union_rank':len(union),'records':records,'original_ce_first_order_matrices':{k:v.tolist() for k,v in matrices.items()},
            'matrix_convention':'row recipient original CE gradient; column source applied first-Adam head update. Positive means loss increase.',
            'summary':summary,'seconds':time.time()-started,'model_optimizer_updates':0,'disposable_first_adam_head_proposals':16,
            'new_eeg_reads':0,'new_model_forward_examples':0,'scope':plan['scope']}
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:{x:v for x,v in row.items() if x not in ['macro_ce_first_order_change_by_source','source_own_ce_first_order_changes']} for k,row in summary.items()},indent=2),flush=True)


if __name__=='__main__':main()
