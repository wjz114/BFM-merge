"""Matrix adaptive calibration in changing old/new expert row planes."""
import torch
from torch.func import functional_call

from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_projected_sketch_v2 import orthonormal_rows


def matrix_inverse_root_action(covariance, vector, epsilon):
    """Apply (sqrt(V)+eps*I)^-1 via the two spectral projectors."""
    a, b, c = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    gap = torch.sqrt((a - c).square() + 4 * b.square())
    high = ((a + c + gap) / 2).clamp_min(0)
    low = ((a + c - gap) / 2).clamp_min(0)
    invgap = torch.where(gap > 0, 1 / gap.clamp_min(torch.finfo(gap.dtype).tiny), 0.)
    difference_action = torch.stack([(a - c) * vector[:, 0] + 2 * b * vector[:, 1],
                                     2 * b * vector[:, 0] + (c - a) * vector[:, 1]], dim=1)
    plus = .5 * (vector + invgap[:, None] * difference_action)
    minus = vector - plus
    return plus / (high.sqrt() + epsilon)[:, None] + minus / (low.sqrt() + epsilon)[:, None]


@torch.no_grad()
def transport_state(old, basis, scale, active):
    transforms = []
    for name in active:
        transforms.append(basis[name].double() @ old['basis'][name].double().transpose(1, 2))
    overlap = torch.cat(transforms)
    old_scale = old['scale'].double()
    new_scale = scale.double()
    ratio = torch.where(old_scale > 0, new_scale / old_scale.clamp_min(1e-300), 0.)
    transform = ratio[:, None, None] * overlap
    m = (transform @ old['m'].double().unsqueeze(-1)).squeeze(-1)
    v = transform @ old['v'].double() @ transform.transpose(1, 2)
    v = (v + v.transpose(1, 2)) / 2
    assert torch.isfinite(m).all() and torch.isfinite(v).all()
    audit = {'previous_step': old['step'], 'old_first_moment_squared_norm': float(old['m'].square().sum()),
             'new_first_moment_squared_norm': float(m.square().sum()),
             'old_second_moment_trace': float(old['v'].diagonal(dim1=1, dim2=2).sum()),
             'new_second_moment_trace': float(v.diagonal(dim1=1, dim2=2).sum()),
             'mean_squared_basis_overlap': float(overlap.square().sum((1, 2)).mean()),
             'scale_ratio_max': float(ratio.max()), 'scale_ratio_min': float(ratio.min())}
    return {'m': m, 'v': v, 'step': int(old['step'])}, audit


class RowMatrixAdam:
    def __init__(self, parameter, condition, history=None, lr=.01, betas=(.9, .999), epsilon=1e-8):
        assert condition in ('diagonal_reset', 'matrix_reset', 'matrix_transport')
        assert parameter.ndim == 2 and parameter.shape[1] == 2
        self.parameter, self.condition = parameter, condition
        self.lr, self.betas, self.epsilon = lr, betas, epsilon
        device = parameter.device
        if history is None:
            self.m = torch.zeros_like(parameter, dtype=torch.float64)
            self.v = torch.zeros(len(parameter), 2, 2, device=device, dtype=torch.float64)
            self.steps = 0
        else:
            assert condition == 'matrix_transport'
            self.m = history['m'].to(device).clone()
            self.v = history['v'].to(device).clone()
            self.steps = int(history['step'])
        self.initial_steps = self.steps

    def zero_grad(self, set_to_none=True):
        assert set_to_none
        self.parameter.grad = None

    @torch.no_grad()
    def step(self):
        gradient = self.parameter.grad.double()
        assert torch.isfinite(gradient).all()
        b1, b2 = self.betas
        self.steps += 1
        self.m.mul_(b1).add_(gradient, alpha=1 - b1)
        self.v.mul_(b2).add_(gradient.unsqueeze(-1) * gradient.unsqueeze(-2), alpha=1 - b2)
        if self.condition == 'diagonal_reset':
            self.v[:, 0, 1] = 0
            self.v[:, 1, 0] = 0
        mean = self.m / (1 - b1 ** self.steps)
        covariance = self.v / (1 - b2 ** self.steps)
        if self.condition == 'diagonal_reset':
            update = mean / (covariance.diagonal(dim1=1, dim2=2).sqrt() + self.epsilon)
        else:
            update = matrix_inverse_root_action(covariance, mean, self.epsilon)
        assert torch.isfinite(update).all()
        self.parameter.add_(update.to(self.parameter.dtype), alpha=-self.lr)

    def state(self):
        return {'m': self.m.detach().cpu().clone(), 'v': self.v.detach().cpu().clone(), 'step': self.steps}


class StreamingMerger:
    def __init__(self, leaf, base, previous, incoming, count, active):
        self.leaf, self.base, self.active = leaf.eval().requires_grad_(False), base, active
        device = next(leaf.parameters()).device
        reference = BilateralMerger(leaf, base, previous, incoming, {}, count, active, 'old_rows', {})
        self.initial_native = reference.materialize()
        del reference
        self.parameter_names = [name for name, _ in leaf.named_parameters()]
        self.anchor = {name: self.initial_native[name].to(device) for name in self.parameter_names}
        self.basis, self.slices, scales = {}, {}, []
        offset, ranks, max_error = 0, {}, 0.
        for name in active:
            old, new = previous[name] - base[name], incoming[name] - base[name]
            rows = old.shape[0] if old.ndim else 1
            directions = torch.stack([old.reshape(rows, -1), new.reshape(rows, -1)], dim=1)
            q = orthonormal_rows(directions)
            scale = directions.double().square().sum(-1).mean(1).sqrt().float()
            support = q.square().sum(-1) > .5
            error = float((q @ q.transpose(1, 2) - torch.diag_embed(support.float())).abs().max())
            max_error = max(max_error, error)
            assert error < 2e-5 and torch.isfinite(scale).all()
            unique, counts = torch.unique(support.sum(1), return_counts=True)
            for rank, n in zip(unique.tolist(), counts.tolist()):
                ranks[str(rank)] = ranks.get(str(rank), 0) + n
            self.basis[name] = q
            self.slices[name] = slice(offset, offset + rows)
            offset += rows
            scales.append(scale)
        self.scale = torch.cat(scales)
        self.device_basis = {name: value.to(device) for name, value in self.basis.items()}
        self.device_scale = self.scale.to(device)
        self.coefficients = torch.nn.Parameter(torch.zeros(offset, 2, device=device))
        self.geometry = {'per_row_rank_counts': ranks, 'orthogonality_max_error': max_error,
                         'row_count': offset, 'coefficient_count': offset * 2,
                         'basis_bytes': sum(v.numel() * v.element_size() for v in self.basis.values()),
                         'scale_bytes': self.scale.numel() * self.scale.element_size(),
                         'scale_max': float(self.scale.max()), 'scale_min': float(self.scale.min())}

    def compose(self, coefficients=None):
        b = self.coefficients if coefficients is None else coefficients
        result = self.anchor.copy()
        for name in self.active:
            sl = self.slices[name]
            delta = (b[sl, :, None] * self.device_basis[name]).sum(1) * self.device_scale[sl, None]
            result[name] = self.anchor[name] + delta.reshape(self.anchor[name].shape)
        return result

    def logits(self, wrapper, x, parameters=None):
        parameters = self.compose() if parameters is None else parameters
        return functional_call(wrapper, {'leaf.' + k: v for k, v in parameters.items()}, (x,))

    @torch.no_grad()
    def materialize(self, coefficients=None):
        parameters = self.compose(coefficients)
        result = {name: parameters.get(name, value).detach().cpu().clone() for name, value in self.base.items()}
        assert all(torch.isfinite(result[name]).all() for name in self.active)
        return result
