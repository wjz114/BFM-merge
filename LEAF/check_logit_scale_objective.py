"""Check the two scale objectives and inspect their local calibration-gradient effects."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256
from train_bank_ce_control import objective as reference_objective
from train_logit_scale_control import objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cache = json.loads(args.feature_cache.read_text())
    torch.set_num_threads(4)
    rows = {}
    for task, record in cache['records'].items():
        assert sha256(Path(record['path'])) == record['sha256']
        data = torch.load(record['path'], map_location='cpu', weights_only=True)
        n = data['n_calibration']
        truth, weights = data['truth'][:n].long(), data['class_weights'].double()
        logits = data['initial_logits'][:n].double().requires_grad_(True)
        native = reference_objective(logits, truth, weights, None, 'balanced_ce')
        base_gradient = torch.autograd.grad(native, logits)[0]
        rows[task] = {}
        for condition in ['scaled_logits', 'scaled_loss']:
            identity = objective(logits, truth, weights, None, condition, 1.)
            identity_gradient = torch.autograd.grad(identity, logits)[0]
            assert torch.equal(identity, native)
            assert torch.equal(identity_gradient, base_gradient)
            scaled = objective(logits, truth, weights, None, condition, 10.)
            gradient = torch.autograd.grad(scaled, logits)[0]
            if condition == 'scaled_loss':
                analytical = 10 * base_gradient
            else:
                analytical = 10 * ((10*logits).softmax(-1)-F.one_hot(truth, logits.shape[1])) * weights[truth, None] / n
            maximum_error = float((gradient-analytical).abs().max())
            assert maximum_error < 1e-12
            p = data['prototypes'].double()
            raw = data['raw_output'][:n].double()
            norm = raw.norm(dim=-1, keepdim=True)
            unit = raw / norm
            unit_grad = gradient @ p
            raw_grad = (unit_grad-(unit_grad*unit).sum(-1, keepdim=True)*unit)/norm
            reference_unit = base_gradient @ p
            reference_raw = (reference_unit-(reference_unit*unit).sum(-1, keepdim=True)*unit)/norm
            head_gradient = raw_grad.T @ data['features'][:n].double()
            reference_head = reference_raw.T @ data['features'][:n].double()
            correct = logits.argmax(-1) == truth
            energy = raw_grad.square().sum(-1)
            rows[task][condition] = {
                'objective': float(scaled.detach()), 'scale_one_identical_to_reference': True,
                'analytical_gradient_max_error': maximum_error,
                'raw_gradient_energy_correct_fraction': float(energy[correct].sum()/energy.sum()),
                'head_gradient_norm_ratio_to_baseline': float(head_gradient.norm()/reference_head.norm()),
                'head_gradient_cosine_to_baseline': float(F.cosine_similarity(head_gradient.reshape(1,-1), reference_head.reshape(1,-1))),
            }
    result = {
        'unix': time.time(), 'source_sha256': {str(p): sha256(p) for p in [Path(__file__), Path('train_logit_scale_control.py'), Path('train_bank_ce_control.py'), args.feature_cache]},
        'records': rows, 'model_inference_performed': False, 'optimizer_updates': 0,
        'validation_used_in_statistics': False, 'test_data_read': False,
        'scope': 'Actual original512 cached logits; scale1 loss and gradient exactly match the frozen reference for both branches. '
                 'Scale10 analytic gradient checked in FP64. Calibration head-gradient effects are descriptive; no model training or efficacy inference.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Both objectives pass on all16 tasks; scale1 identity exact; max analytical gradient error',
          max(v['analytical_gradient_max_error'] for row in rows.values() for v in row.values()))


if __name__ == '__main__':
    main()
