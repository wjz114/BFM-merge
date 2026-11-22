"""Audit all-task contrast-null output components using cached training rows."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256
from distillation_control_support import readout


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    cp = root/'seed0/full_model_distillation_targets/run.json'
    bp = root/'seed0/prototype_head_geometry/contrast_bases.pt'
    cache = json.loads(cp.read_text())
    union = torch.load(bp, map_location='cpu', weights_only=True)['union_basis'].double()
    assert union.shape == (22, 768)
    assert torch.allclose(union@union.T, torch.eye(22, dtype=torch.float64), atol=1e-12, rtol=0)
    schedule = np.load(cache['plan']['schedule'])
    rows, hashes = {}, {str(p): sha256(p) for p in [cp, bp]}
    for ti, task in enumerate(cache['datasets']):
        record = cache['records'][task]
        path = Path(record['target_path'])
        assert sha256(path) == record['target_sha256']
        hashes[str(path)] = record['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64))
        pos = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][pos], ids)
        raw = data['initial_raw'][pos].double()
        p = data['prototypes'].double()
        y = data['truth'][pos]
        w = torch.tensor(record['class_weights'], dtype=torch.float64)[y]
        projected = (raw@union.T)@union
        null = raw-projected
        centered = p-p.mean(0)
        null_contrast_error = float((centered-(centered@union.T)@union).norm()/centered.norm())
        assert null_contrast_error < 1e-12
        logits = readout(raw, p)
        plogits = readout(projected, p)
        assert torch.equal(logits.argmax(-1), plogits.argmax(-1))
        scale = raw.norm(dim=-1)/projected.norm(dim=-1)
        expected_centered = (logits-logits.mean(-1, keepdim=True))*scale[:, None]
        relation_error = float((expected_centered-(plogits-plogits.mean(-1, keepdim=True))).abs().max())
        assert relation_error < 1e-12
        unit = F.normalize(raw, dim=-1)
        v = (logits.softmax(-1)-F.one_hot(y, len(p)))@p
        gradient = w[:, None]*(v-unit*(v*unit).sum(-1, keepdim=True))/raw.norm(dim=-1, keepdim=True)
        gn = gradient-(gradient@union.T)@union
        # Independent autograd check on a real training batch.
        z = raw[:128].clone().requires_grad_(True)
        autograd, = torch.autograd.grad((w[:128]*F.cross_entropy(readout(z, p), y[:128], reduction='none')).sum(), z)
        relative = float((autograd-gradient[:128]).norm()/autograd.norm())
        assert relative < 1e-10
        correct = logits.argmax(-1) == y
        en, total = gn.square().sum(-1), gradient.square().sum(-1)
        null_norm_increases = (gn*null).sum(-1) < 0
        row = {'native_argmax_changes_after_global_projection': 0,
               'prototype_contrast_null_relative_error': null_contrast_error,
               'centered_logit_scaling_max_abs_error': relation_error,
               'analytic_ce_gradient_relative_error': relative,
               'weighted_output_global_null_energy_fraction': float((w*null.square().sum(-1)/raw.square().sum(-1)).sum()/w.sum()),
               'ce_output_gradient_global_null_energy_fraction': float(en.sum()/total.sum()),
               'projection_logit_scale': {'weighted_mean': float((w*scale).sum()/w.sum()), 'min': float(scale.min()), 'max': float(scale.max())},
               'weighted_native_ce': float((w*F.cross_entropy(logits, y, reduction='none')).mean()),
               'weighted_projected_ce': float((w*F.cross_entropy(plogits, y, reduction='none')).mean()),
               'groups': {}}
        for name, mask in [('initial_correct', correct), ('initial_wrong', ~correct)]:
            mass = float(w[mask].sum())
            row['groups'][name] = {'weighted_mass_fraction': mass/float(w.sum()),
                'ce_null_gradient_energy_fraction_within_group': float(en[mask].sum()/total[mask].sum()),
                'weighted_null_norm_increase_fraction_within_group': float(w[mask & null_norm_increases].sum())/mass}
        rows[task] = row
    macro = {k: float(np.mean([row[k] for row in rows.values()])) for k in [
        'weighted_output_global_null_energy_fraction', 'ce_output_gradient_global_null_energy_fraction',
        'weighted_native_ce', 'weighted_projected_ce']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'per_task': rows, 'macro': macro, 'seconds': time.time()-started,
              'new_model_inference': False, 'fitting_performed': False, 'validation_used_in_statistics': False,
              'test_data_read': False,
              'scope': 'Exact fitting multiset at frozen merger initialization. Global projection preserves all centered raw-score contrasts and argmax; after normalization it raises centered logits by a sample-dependent positive factor. Output-space CE null components can change confidence without changing current decisions, but can affect future learning and other samples through the model Jacobian. Neither null energy nor projected fitting CE establishes harmful interference, a candidate generalization gain, or novelty. No proposed intervention is fitted/evaluated here.'}
    output = root/'analysis/readout_global_null_diagnostic.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
