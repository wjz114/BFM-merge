"""Freeze one prehead-gradient scale per task, shared by all target interventions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from prehead_distillation_support import CONDITIONS, objective, raw_ce_gradient, target_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    for path, digest in {**design['source_sha256'], **design['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    started = time.time()
    torch.set_num_threads(4)
    records, sources = {}, {}
    for spec in design['cache_shards'].values():
        path = Path(spec['outdir'])/'run.json'
        run = json.loads(path.read_text())
        assert run['all_model_states_unchanged'] and not run['validation_used_in_statistics']
        assert run['plan']['design_sha256'] == sha256(args.design)
        records.update(run['records']); sources[str(path)] = sha256(path)
    assert set(records) == set(design['datasets'])
    merged = {'datasets': design['datasets'], 'records': records, 'source_sha256': sources,
              'validation_used_in_statistics': False, 'design': str(args.design), 'design_sha256': sha256(args.design)}
    with Path(design['target_manifest']).open('x') as f:json.dump(merged,f,indent=2);f.write('\n')
    schedule = np.load(design['schedule'])
    per_task, checks = {}, {}
    for ti, task in enumerate(design['datasets']):
        row = records[task]
        assert sha256(Path(row['target_path'])) == row['target_sha256']
        data = torch.load(row['target_path'],map_location='cpu',weights_only=True)
        ids = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert array_hash(ids) == row['stream_indices_hash']
        pos = torch.searchsorted(data['pool_indices'], torch.from_numpy(ids))
        assert torch.equal(data['pool_indices'][pos], torch.from_numpy(ids))
        weights = torch.tensor(row['class_weights'],dtype=torch.float64)
        truth, h = data['truth'][pos], data['initial_prehead'][pos].double()
        ce_raw = raw_ce_gradient(data['initial_raw'][pos].double(),truth,data['prototypes'].double(),weights)
        ce = ce_raw @ data['initial_weight'].double()
        ce_rms = float(ce.square().sum(-1).mean().sqrt())
        aux = {}
        for condition in CONDITIONS[1:]:
            target = target_rows(data,pos,condition).double()
            grad = 2 * weights[truth,None] * (h-target) / h.shape[1]
            rms = float(grad.square().sum(-1).mean().sqrt())
            assert rms > 0 and np.isfinite(rms)
            aux[condition] = {'auxiliary_prehead_gradient_rms': rms,
                'weighted_auxiliary_mean': float(((h-target).square().mean(-1)*weights[truth]).mean()),
                'flattened_gradient_cosine_with_ce': float((grad*ce).sum()/(grad.norm()*ce.norm()))}
        scale = ce_rms / aux['ce_prehead_paired']['auxiliary_prehead_gradient_rms']
        assert scale>0 and np.isfinite(scale)
        for values in aux.values():
            values['shared_scale'] = scale
            values['scaled_aux_to_ce_prehead_rms_ratio'] = scale*values['auxiliary_prehead_gradient_rms']/ce_rms
        per_task[task] = {'scale': scale, 'ce_prehead_gradient_rms': ce_rms, 'conditions': aux}
        # Real native head and double normalization: analytic CE/MSE gradients.
        subset = pos[:32]
        probe = data['initial_prehead'][subset].double().requires_grad_(True)
        raw = F.linear(probe, data['initial_weight'].double(), data['initial_bias'].double())
        logits = F.normalize(F.normalize(raw,dim=-1),dim=-1) @ data['prototypes'].double().T
        labels = data['truth'][subset]
        reference = (F.cross_entropy(logits,labels,reduction='none')*weights[labels]).mean()
        actual, = torch.autograd.grad(reference,probe,retain_graph=True)
        expected = raw_ce_gradient(raw.detach(),labels,data['prototypes'].double(),weights) @ data['initial_weight'].double()/len(probe)
        error = float((actual-expected).norm()/actual.norm())
        assert error < 1e-10, (task,error)
        rows = {}
        for condition in CONDITIONS[1:]:
            target = target_rows(data,subset,condition).double()
            zero = objective(logits,probe,labels,weights,target,0.)
            zg, = torch.autograd.grad(zero,probe,retain_graph=True)
            assert torch.equal(zero,reference) and torch.equal(zg,actual)
            value = objective(logits,probe,labels,weights,target,scale)
            g, = torch.autograd.grad(value,probe,retain_graph=True)
            expected_total = actual + scale*2*weights[labels,None]*(probe.detach()-target)/probe.numel()
            relative = float((g-expected_total).norm()/g.norm())
            assert relative < 1e-10 and torch.isfinite(g).all()
            rows[condition] = {'total_gradient_relative_error': relative, 'zero_auxiliary_exact': True}
        checks[task] = {'ce_gradient_relative_error': error, 'conditions': rows}
        print(task,'fixed scale',scale,'ratios',{k:v['scaled_aux_to_ce_prehead_rms_ratio'] for k,v in aux.items()},flush=True)
    result = {'unix':time.time(),'design':str(args.design),'design_sha256':sha256(args.design),
        'code_sha256':sha256(Path(__file__)),'support_sha256':sha256(Path('prehead_distillation_support.py')),
        'target_manifest':design['target_manifest'],'target_manifest_sha256':sha256(Path(design['target_manifest'])),
        'per_task':per_task,'objective_checks':checks,'seconds':time.time()-started,
        'validation_used':False,'test_data_read':False,
        'scope':'One initial fitting-stream gradient-RMS scale at the2048 prehead, calibrated on PAIRED targets only and shared unchanged by mean/shuffled controls. No parameter-gradient matching, adaptive scaling, validation tuning, or claim that this matches earlier768-output objective strengths.'}
    with Path(design['scales']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Completed scale calibration',result['seconds'],flush=True)


if __name__ == '__main__':
    main()
