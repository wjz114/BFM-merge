"""Measure CE gradients hidden by shared Block coefficients; no fitting.

Expands the selected Block coefficients to one coefficient per parameter
tensor, then measures gradients on the original 512-example calibration
pools. The tied subspace is the original 33-group parameterization.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from d_downstream_qformer_ce import ROOT, load_leaf, make_loader, prepare_text, prototype_logits, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES, group_for_name


def projected_gradient(gradient, group_ids, group_count):
    projected = torch.empty_like(gradient)
    for group in range(group_count):
        mask = group_ids == group
        projected[:, mask] = gradient[:, mask].mean(1, keepdim=True)
    return projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/aligned_merge_v1/seed0/block_ce')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'runs/aligned_merge_v1/seed0/analysis/tensor_gradient_diagnostic')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    run = json.loads((args.run / 'run.json').read_text())
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(run['base_checkpoint'])).to(device).eval().requires_grad_(False)
    base = {name: value.detach() for name, value in leaf.named_parameters()}
    buffers = dict(leaf.named_buffers())
    selected = [name for name in base if not name.startswith(UNUSED_PREFIXES)]
    groups = run['groups']
    group_ids = torch.tensor([groups.index(group_for_name(name)) for name in selected], device=device)
    deltas = {name: [] for name in selected}
    for ds in run['datasets']:
        state = torch.load(run['experts'][ds], map_location=device, weights_only=True)
        for name in selected:
            deltas[name].append(state[name] - base[name])
    deltas = {name: torch.stack(values) for name, values in deltas.items()}
    alpha = torch.tensor(run['best_alpha'], device=device)[:, group_ids].clone().requires_grad_(True)

    def compose():
        params = dict(base)
        for column, name in enumerate(selected):
            params[name] = base[name] + torch.tensordot(alpha[:, column], deltas[name], dims=1)
        return params

    reference = torch.load(args.run / 'merged.ckpt', map_location=device, weights_only=True)
    with torch.no_grad():
        state_error = max((value - reference[name]).abs().max().item() for name, value in compose().items())
    if state_error > 1e-6:
        raise RuntimeError(f'Expanded coefficients do not reproduce selected Block: {state_error}')
    del reference, state
    indices = np.load(args.run / 'calibration_indices.npz')
    gradients, losses = [], {}
    for ds in run['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, ds)
        instruction, prototypes = instruction.to(device), prototypes.to(device)
        _, x, y, vx, vy, tx, ty = load_dataset(ds)
        index = indices[ds]
        loader = make_loader(x[index], y[index], config, len(labels), 32, False)
        del x, y, vx, vy, tx, ty
        alpha.grad = None
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            embedding, _ = functional_call(leaf, (compose(), buffers),
                                           (x, instruction[None].expand(len(x), -1)))
            loss = F.cross_entropy(prototype_logits(embedding, prototypes), y)
            (loss * (len(x) / len(index))).backward()
            loss_sum += loss.item() * len(x) / len(index)
        if alpha.grad is None or not torch.isfinite(alpha.grad).all():
            raise RuntimeError(f'{ds}: missing/nonfinite gradient')
        gradients.append(alpha.grad.detach().cpu().clone())
        losses[ds] = loss_sum
        print(f'{ds}: calibration CE={loss_sum:.6f}', flush=True)
    gradient_stack = torch.stack(gradients)
    group_ids = group_ids.cpu()
    summaries, group_rows = {}, []
    for name, gradient in [*zip(run['datasets'], gradients), ('macro_mean', gradient_stack.mean(0))]:
        projected = projected_gradient(gradient, group_ids, len(groups))
        remainder = gradient - projected
        summaries[name] = {'fine_gradient_l2': gradient.norm().item(),
                           'tied_projection_l2': projected.norm().item(),
                           'untied_fraction_squared_l2': (remainder.square().sum() / gradient.square().sum()).item()}
        for group, label in enumerate(groups):
            mask = group_ids == group
            group_rows.append({'task': name, 'group': label, 'parameter_tensors': int(mask.sum()),
                'fine_gradient_l2': gradient[:, mask].norm().item(),
                'tied_projection_l2': projected[:, mask].norm().item(),
                'untied_gradient_l2': remainder[:, mask].norm().item()})
    np.savez(args.outdir / 'gradients.npz', gradients=gradient_stack.numpy(),
             datasets=run['datasets'], parameters=selected, group_ids=group_ids.numpy(), groups=groups)
    with (args.outdir / 'by_group.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    (args.outdir / 'summary.json').write_text(json.dumps({
        'meaning': 'Euclidean coefficient-gradient projection in expanded tensor coordinates; calibration CE only, no fitting or validation gain claim',
        'checkpoint': str(args.run / 'merged.ckpt'), 'checkpoint_sha256': sha256(args.run / 'merged.ckpt'),
        'calibration_sha256': sha256(args.run / 'calibration_indices.npz'),
        'precision': 'FP32 forward/loss/composition, TF32 off', 'state_reconstruction_max_abs': state_error,
        'groups': len(groups), 'parameter_tensors': len(selected), 'coefficients': alpha.numel(),
        'calibration_ce': losses, 'gradients': summaries, 'seconds': time.time() - start,
        'code_sha256': sha256(Path(__file__))}, indent=2) + '\n')
    print(json.dumps(summaries['macro_mean']), flush=True)


if __name__ == '__main__':
    main()
