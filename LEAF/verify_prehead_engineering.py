"""Verify the new prehead loss path without extrapolating deterministic replay."""
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def main():
    root=Path('runs/paper_merge_v2')
    reference=root/'seed0/bank_ce_deterministic_replay'
    candidate=root/'seed0/prehead_ce_deterministic_replay'
    paired=root/'seed0/prehead_paired_smoke'
    old,new,aux=[json.loads((p/'run.json').read_text()) for p in [reference,candidate,paired]]
    assert all(r['engineering_smoke'] and r['updates']==32 for r in [old,new,aux])
    assert old['datasets']==new['datasets']==aux['datasets']
    assert old['schedule_sha256']==new['schedule_sha256']==aux['schedule_sha256']
    assert all(row['auxiliary_scale']==0 for row in new['distillation_audits'].values())
    assert all(row['auxiliary_scale']>0 for row in aux['distillation_audits'].values())
    assert all(new['numerical_flags'].values()) and not any(aux['numerical_flags'].values())
    checks={}
    for name in ['step_16.ckpt','step_32.ckpt','selected.ckpt','final.ckpt','merged.ckpt']:
        a,b=[torch.load(p/name,map_location='cpu',weights_only=True,mmap=True) for p in [reference,candidate]]
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),name
        checks[name]={'all_tensors_exact':True,'tensor_count':len(a)}
    for key in ['initial_validation','selected_validation','final_validation','average_validation',
                'average_calibration','dataset_audits','component_gradient_max']:
        assert old[key]==new[key],key
    assert json.loads((reference/'history.json').read_text())==json.loads((candidate/'history.json').read_text())
    for r in [new,aux]:
        assert r['fixed_tensors_exact'] and r['trainable_parameters']==25751552
        assert all(v>0 for v in r['component_gradient_max'].values())
        assert not r['test_data_read'] and not r['test_inference_performed']
        assert max(v['max_abs_logit_error'] for v in r['export_checks'].values())<=1e-5
        for name,digest in r['plan']['source_sha256'].items():assert sha256(Path(name))==digest,name
    scale_path=Path(aux['plan']['scales'])
    scales=json.loads(scale_path.read_text())
    assert sha256(scale_path)==aux['scales_sha256']
    result={'unix':time.time(),'code_sha256':sha256(Path(__file__)),
        'sources':{str(p/'run.json'):sha256(p/'run.json') for p in [reference,candidate,paired]},
        'scale_checks':{'path':str(scale_path),'sha256':sha256(scale_path)},
        'state_checks':checks,'all_zero_auxiliary_metrics_histories_data_gradients_exact':True,
        'paired_smoke_finite_active_gradients_and_native_export':True,'engineering_optimizer_updates':64,
        'scope':'Deterministic32-step CE engineering replay plus native32-step paired-feature gradient/export smoke. All three target derivatives verified in the scale artifact. This does not require scientific native trajectories to match historical states bitwise and is not efficacy evidence.'}
    out=root/'analysis/prehead_engineering_verification.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS prehead engineering',sha256(out))


if __name__=='__main__':main()
