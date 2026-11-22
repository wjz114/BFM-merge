"""Investigate the failed input-Jacobian smoke gate; training rows only, no fitting."""
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, sha256
from diagnose_input_responses_v2 import contrast_basis
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


def main():
    started = time.time()
    root = Path('runs/paper_merge_v2')
    target = root/'analysis/input_response_numerics.json'
    assert not target.exists()
    plan_path = root/'inputs/input_response_v2_shard0_plan.json'
    plan = json.loads(plan_path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    device = configure(0, 0)
    source = json.loads(Path(plan['source_fit']).read_text())
    task = 'MI_OpenBMI'
    old = np.load(plan['calibration'])[task]
    with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
        rows, truth = row_map(h, task, 'calibration')
        labels512 = np.asarray(truth, dtype=np.int64)[old]
        rng = np.random.default_rng([0, source['datasets'].index(task), 20261116])
        positions = np.concatenate([rng.choice(np.flatnonzero(labels512 == k), 2, replace=False)
                                    for k in range(2)])[:2]
        selected = [rows[int(i)] for i in old[positions]]
        memory = {key: h[key][:] for key in dict.fromkeys(row[0] for row in selected)}
    x32 = torch.from_numpy(read_rows(memory, selected)).to(device)
    del memory
    covariance = np.load(plan['sensor_covariances'][task]['path'])
    support = torch.eye(65, dtype=torch.float64)-torch.from_numpy(covariance['null_projector'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    _, _, instruction, prototypes = prepare_text(config, task)
    results = {}
    for label, spec in [('teacher', plan['experts'][task]), ('initial', plan['models']['initial'])]:
        assert sha256(Path(spec['checkpoint'])) == spec['sha256']
        model = load_leaf(config, Path(spec['checkpoint'])).to(device).eval()
        model.requires_grad_(False)
        original = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
        state = {k: p.detach().cpu().clone() for k, p in original.state_dict().items()}
        model_results = {}
        direction32 = None
        gradient32 = None
        for dtype in [torch.float32, torch.float64]:
            wrapper = original if dtype == torch.float32 else copy.deepcopy(original).double()
            x = x32.to(dtype).detach().requires_grad_(True)
            basis = contrast_basis(2).to(device=device, dtype=dtype)
            captured = []

            def hook(module, arguments):
                _, indices = F.max_pool2d(arguments[0], module.kernel_size, module.stride,
                                         module.padding, module.dilation, module.ceil_mode, True)
                captured.append(indices.detach().cpu())

            handle = wrapper.leaf.tower.tokenizer.enc_conv[4].register_forward_pre_hook(hook)

            def forward(value):
                if dtype == torch.float32:
                    return wrapper(value)
                # Same operations as the native classifier, retaining float64 in its last normalization.
                embedding, _ = wrapper.leaf(value, wrapper.instruction[None].expand(len(value), -1))
                return F.normalize(embedding, dim=-1)@wrapper.prototypes.T

            logits = forward(x)
            baseline_pool = captured.pop()
            gradient = torch.autograd.grad((logits@basis.T)[:, 0].sum(), x)[0].detach().cpu().double()
            if dtype == torch.float32:
                gradient32 = gradient.clone()
                direction = support@gradient
                direction *= (x.detach().cpu().double().flatten(1).norm(dim=1)
                              /direction.flatten(1).norm(dim=1))[:, None, None]
                direction32 = direction.float()
            direction = direction32.to(device=device, dtype=dtype)
            reference = (gradient*direction.detach().cpu().double()).sum((1, 2))
            rows = []
            for step in [1e-4, 5e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                # Keep the autograd-enabled execution path on both sides of the finite difference.
                positive = (forward((x.detach()+step*direction).requires_grad_(True))@basis.T)[:, 0].detach().cpu().double()
                positive_pool = captured.pop()
                negative = (forward((x.detach()-step*direction).requires_grad_(True))@basis.T)[:, 0].detach().cpu().double()
                negative_pool = captured.pop()
                finite = (positive-negative)/(2*step)
                rows.append({'relative_input_step': step, 'central_difference': finite.tolist(),
                             'relative_l2_error': float((finite-reference).norm()/reference.norm()),
                             'positive_pool_switches': int((positive_pool != baseline_pool).sum()),
                             'negative_pool_switches': int((negative_pool != baseline_pool).sum()),
                             'pool_outputs': baseline_pool.numel()})
            handle.remove()
            model_results[str(dtype)] = {'logits': logits.detach().cpu().tolist(),
                                         'directional_derivative': reference.tolist(),
                                         'gradient_norm_per_example': gradient.flatten(1).norm(dim=1).tolist(),
                                         'gradient_relative_difference_from_fp32': float((gradient-gradient32).norm()/gradient32.norm()),
                                         'finite_differences': rows}
            print(label, dtype, model_results[str(dtype)], flush=True)
            if dtype == torch.float64:
                del wrapper
        assert all(torch.equal(p.detach().cpu(), state[k]) for k, p in original.state_dict().items())
        assert all(p.grad is None for p in original.parameters())
        results[label] = model_results
        del original, model, state
    result = {'source_plan': str(plan_path), 'source_plan_sha256': sha256(plan_path),
              'source_sha256': sha256(Path(__file__)), 'task': task,
              'positions_in512': positions.tolist(), 'training_ids': old[positions].tolist(),
              'input_norm': x32.flatten(1).norm(dim=1).cpu().tolist(), 'records': results,
              'optimizer_updates': 0, 'all_model_states_unchanged': True,
              'validation_inference_performed': False, 'test_data_read': False,
              'scope': 'Numerical investigation of the first two original-training smoke rows only. Float64 is a mathematical-operation reference, not a native candidate. No performance or finite-perturbation claim.',
              'seconds': time.time()-started}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target), flush=True)


if __name__ == '__main__':
    main()
