"""Restricted updates of the existing shared semantic projection.

This is a repair experiment, not a prediction-preserving CPSC transform.
Updates can use the full output space, the union of task-prototype contrast
spaces, or a random subspace of the same rank. All start at zero and export
back to the original ordinary Linear layer.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def prototype_union_basis(prototypes):
    rows = []
    for p in prototypes:
        p = p.detach().cpu().double().numpy()
        rows.append(p - p.mean(axis=0, keepdims=True))
    _, singular, vt = np.linalg.svd(np.concatenate(rows), full_matrices=False)
    rank = int((singular > max(1e-12, 1e-10 * singular[0])).sum())
    return torch.from_numpy(vt[:rank].T.copy()).float(), singular.tolist()


class ProjectionDelta(nn.Module):
    """W = W_init + Q A, b = b_init + Q a; Q=None gives full output repair."""
    def __init__(self, original, basis=None):
        super().__init__()
        self.register_buffer('weight', original.weight.detach().clone())
        self.register_buffer('bias', original.bias.detach().clone())
        self.register_buffer('basis', basis.to(original.weight) if basis is not None else None)
        rank = original.out_features if basis is None else basis.shape[1]
        self.delta_weight = nn.Parameter(original.weight.new_zeros(rank, original.in_features))
        self.delta_bias = nn.Parameter(original.bias.new_zeros(rank))

    def effective_parameters(self):
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            dw = self.delta_weight if self.basis is None else self.basis @ self.delta_weight
            db = self.delta_bias if self.basis is None else self.basis @ self.delta_bias
            return self.weight + dw, self.bias + db

    def forward(self, features):
        return F.linear(features, *self.effective_parameters())


def plain_state(leaf):
    state = dict(leaf.state_dict())
    projection = leaf.qformer.output_embed[-1]
    if isinstance(projection, ProjectionDelta):
        prefix = 'qformer.output_embed.2.'
        for name in ['basis', 'delta_weight', 'delta_bias']:
            state.pop(prefix + name, None)
        w, b = projection.effective_parameters()
        state[prefix + 'weight'], state[prefix + 'bias'] = w.detach(), b.detach()
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


@torch.no_grad()
def projection_features(leaf, loader, instruction, device):
    """One deterministic frozen forward per trial, preserving loader order."""
    captured, labels = [], []
    projection = leaf.qformer.output_embed[-1]
    handle = projection.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu()))
    try:
        # Even an unshuffled DataLoader consumes a CPU RNG seed on iteration.
        # Feature preparation must not change the later calibration schedule.
        with torch.random.fork_rng(devices=[]):
            for x, y in loader:
                leaf(x.to(device), instruction[None].expand(len(x), -1))
                labels.append(y)
    finally:
        handle.remove()
    return torch.cat(captured), torch.cat(labels)
