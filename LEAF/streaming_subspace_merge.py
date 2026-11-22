"""Split incoming updates using observed old-task linear-layer inputs.

This is a calibrated merge operator, not a null-space preservation guarantee.
Attention maps called functionally and convolutional weights retain the incoming
row baseline. No extra branch survives materialization.
"""
import time

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_residual_merge_v2 import StreamingMerger as IncomingRowMerger


def observable_linears(leaf, active):
    attention = [n for n, m in leaf.named_modules() if isinstance(m, torch.nn.MultiheadAttention)]
    return {n + '.weight': m for n, m in leaf.named_modules()
            if isinstance(m, torch.nn.Linear) and n + '.weight' in active
            and not any(n.startswith(a + '.') for a in attention)}


@torch.no_grad()
def collect_split(leaf, base, previous, incoming, data, old_tasks, plan, stage, condition, device):
    """Task-balanced activation sketches; identical row IDs across conditions."""
    started = time.monotonic()
    leaf.load_state_dict(previous, strict=True)
    layers = observable_linears(leaf, plan['active'])
    assert list(layers) == plan['split_layers']
    samples = {n: [] for n in layers}
    selected_ids, counts = {}, {n: [] for n in layers}
    handles = []
    for name, module in layers.items():
        def capture(module, args, name=name):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            take = min(plan['tokens_per_task_layer'], len(x))
            ids = torch.linspace(0, len(x) - 1, take, device=x.device).round().long()
            # Each old task contributes equal total weight, despite sequence lengths.
            samples[name].append((x[ids] / np.sqrt(take)).cpu())
            counts[name].append(take)
        handles.append(module.register_forward_pre_hook(capture))
    try:
        for task in old_tasks:
            rng = np.random.default_rng([plan['seed'], stage, plan['tasks'].index(task), 881])
            ids = rng.permutation(len(data.train[task][0]))[:plan['subspace_examples_per_task']]
            selected_ids[task] = ids.tolist()
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            wrapper(data.train[task][0][ids].to(device))
    finally:
        for handle in handles:
            handle.remove()
    projections, bases, scales, records = {}, {}, {}, {}
    for j, name in enumerate(layers):
        assert len(samples[name]) == len(old_tasks), name
        x = torch.cat(samples.pop(name)).to(device)
        q = min(plan['subspace_rank'] + plan['svd_oversampling'], *x.shape)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(plan['seed'] + stage * 1000 + j)
            _, singular, v = torch.svd_lowrank(x, q=q, niter=plan['svd_power_iterations'])
        rank = min(plan['subspace_rank'], int((singular > singular[0] * 1e-6).sum()))
        assert rank > 0, name
        real = v[:, :rank]
        ortho_error = float((real.T @ real - torch.eye(rank, device=device)).abs().max())
        assert ortho_error < 1e-4, (name, ortho_error)
        delta = (incoming[name] - base[name]).to(device)
        aligned = (delta @ real) @ real.T
        real_norm = aligned.norm(dim=1)
        generator = torch.Generator(device=device).manual_seed(plan['seed'] + stage * 2000 + j + 19017)
        random_basis = torch.linalg.qr(torch.randn(real.shape, generator=generator, device=device), mode='reduced').Q
        random_direction = (delta @ random_basis) @ random_basis.T
        scale = real_norm / random_direction.norm(dim=1).clamp_min(1e-30)
        random_direction.mul_(scale[:, None])
        if condition == 'random_split':
            basis, direction, row_scale = random_basis, random_direction, scale
        else:
            basis, direction, row_scale = real, aligned, torch.ones_like(real_norm)
        projections[name] = direction
        bases[name], scales[name] = basis.cpu(), row_scale.cpu()
        records[name] = {'rank': rank, 'input_dim': x.shape[1], 'sampled_tokens': sum(counts[name]),
                         'basis_orthogonality_max_abs': ortho_error,
                         'captured_activity_fraction': float((x @ real).square().sum() / x.square().sum()),
                         'projected_delta_energy_fraction': float(aligned.square().sum() / delta.square().sum().clamp_min(1e-30)),
                         'random_row_norm_max_abs_error': float((random_direction.norm(dim=1) - real_norm).abs().max())}
        del x, delta, aligned, random_direction
    audit = {'old_tasks': old_tasks, 'selected_local_ids': selected_ids,
             'selected_ids_hash': array_hash(np.asarray(list(selected_ids.values()), dtype=np.int64)),
             'extra_forward_examples': len(old_tasks) * plan['subspace_examples_per_task'],
             'layers': records, 'seconds': time.monotonic() - started,
             'scope': 'Old model, old training calibration only. Activation energy is not downstream task importance. Random direction matches real-direction per-row norms.'}
    return projections, {'basis': bases, 'row_scale': scales}, audit


class StreamingMerger(IncomingRowMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition, projections):
        assert condition in ('incoming_rows', 'activation_split', 'random_split')
        super().__init__(leaf, base, previous, incoming, history_sum, count, active, 'incoming_rows')
        self.condition = condition
        self.row_coefficient_count = self.coefficients.numel()
        self.projected = {} if condition == 'incoming_rows' else projections
        self.split_slices = {}
        offset = self.row_coefficient_count
        pieces = [self.coefficients.detach()]
        for name, value in self.projected.items():
            self.split_slices[name] = slice(offset, offset + len(value))
            pieces.append(torch.zeros(len(value), device=self.coefficients.device))
            offset += len(value)
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.row_coefficient_count])
        for name, direction in self.projected.items():
            result[name] = result[name] + c[self.split_slices[name]][:, None] * direction
        return result
