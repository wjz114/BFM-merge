"""Learn coupled attention-operator residuals with native rank and local gauge."""
import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger


CONDITIONS = ('operator_zero', 'operator_covariance', 'operator_random')


class FactorSumProjection(torch.autograd.Function):
    """Rank-k projection of X.T@Y, differentiated for product-invariant losses.

    Inputs have 2k rows. The first k rows supply the local factor gauge. Backward
    uses only the singular-value gap across the retained/discarded boundary;
    repeated singular values within either block need no divided difference.
    This is not a derivative for arbitrary losses on the returned factors.
    """
    @staticmethod
    def forward(ctx, x, y, rank):
        dtype = x.dtype
        x, y = x.double(), y.double()
        qx, rx = torch.linalg.qr(x.transpose(-1, -2), mode='reduced')
        qy, ry = torch.linalg.qr(y.transpose(-1, -2), mode='reduced')
        # The 64x64 CPU decomposition is substantially faster than CUDA's
        # batched Jacobi kernel here; QR and the model remain on the GPU.
        core = rx @ ry.transpose(-1, -2)
        uc, s, vhc = [t.to(x.device) for t in torch.linalg.svd(core.cpu(), full_matrices=False)]
        u, v = qx @ uc, qy @ vhc.transpose(-1, -2)
        gap = s[..., rank - 1].square() - s[..., rank].square()
        assert bool((gap > 1e-12 * s[..., 0].square()).all()), 'rank-boundary singular-value gap collapsed'
        u1, v1 = u[..., :rank], v[..., :rank]
        gauge = x[..., :rank, :] @ u1
        left = gauge @ u1.transpose(-1, -2)
        right = torch.linalg.solve(gauge.transpose(-1, -2),
            s[..., :rank, None] * v1.transpose(-1, -2))
        # At an exactly zero correction, preserve the ordinary factors bitwise.
        zero = (y[..., rank:, :] == 0).all(dim=(-2, -1))
        left = torch.where(zero[..., None, None], x[..., :rank, :], left)
        right = torch.where(zero[..., None, None], y[..., :rank, :], right)
        ctx.save_for_backward(x, y, u, v, s, gauge)
        ctx.rank, ctx.dtype = rank, dtype
        return left.to(dtype), right.to(dtype)

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        x, y, u, v, s, gauge = ctx.saved_tensors
        k = ctx.rank
        u1, u2, v1, v2 = u[..., :k], u[..., k:], v[..., :k], v[..., k:]
        # Recover the product gradient's components touching either kept space.
        lfull = (grad_left.double().transpose(-1, -2) @ gauge) / s[..., None, :k]
        rfull = torch.linalg.solve(gauge, grad_right.double())
        left, right = u.transpose(-1, -2) @ lfull, rfull @ v
        e11 = (left[..., :k, :] + right[..., :, :k]) * .5
        keep, rest = s[..., :k, None], s[..., None, k:]
        denominator = keep.square() - rest.square()
        a, b = keep.square() / denominator, keep * rest / denominator
        d12 = a * right[..., :, k:] + b * left[..., k:, :].transpose(-1, -2)
        d21 = a.transpose(-1, -2) * left[..., k:, :] + b.transpose(-1, -2) * right[..., :, k:].transpose(-1, -2)
        product_gradient = (u1 @ e11 @ v1.transpose(-1, -2)
            + u1 @ d12 @ v2.transpose(-1, -2)
            + u2 @ d21 @ v1.transpose(-1, -2)
            + (lfull - u @ left) @ v1.transpose(-1, -2)
            + u1 @ (rfull - right @ v.transpose(-1, -2)))
        return (y @ product_gradient.transpose(-1, -2)).to(ctx.dtype), (x @ product_gradient).to(ctx.dtype), None


def factor_pairs(parameters, names):
    left, right = [], []
    for name in names:
        q, k, v = parameters[name + '.in_proj_weight'].reshape(3, 8, 32, 256)
        qb, _, vb = parameters[name + '.in_proj_bias'].reshape(3, 8, 32)
        ot = parameters[name + '.out_proj.weight'].reshape(256, 8, 32).permute(1, 2, 0)
        left += [torch.cat([q, qb[..., None]], -1), torch.cat([v, vb[..., None]], -1)]
        right += [k, ot]
    return torch.stack(left).flatten(0, 1), torch.stack(right).flatten(0, 1)


def restore_factors(parameters, names, left, right):
    result = dict(parameters)
    left = left.reshape(len(names), 2, 8, 32, 257)
    right = right.reshape(len(names), 2, 8, 32, 256)
    for index, name in enumerate(names):
        q, v = left[index, 0], left[index, 1]
        k, ot = right[index, 0], right[index, 1]
        result[name + '.in_proj_weight'] = torch.cat([q[..., :256].reshape(256, 256),
            k.reshape(256, 256), v[..., :256].reshape(256, 256)])
        # Key bias contributes a per-query constant to the attention logits.
        # Keeping its original value also preserves all tensors at zero gates.
        result[name + '.in_proj_bias'] = torch.cat([q[..., 256].reshape(256),
            parameters[name + '.in_proj_bias'][256:512], v[..., 256].reshape(256)])
        result[name + '.out_proj.weight'] = ot.permute(2, 0, 1).reshape(256, 256)
    return result


@torch.no_grad()
def random_spectrum_factors(left, right, seed):
    ql, rl = torch.linalg.qr(left.transpose(-1, -2).double(), mode='reduced')
    qr, rr = torch.linalg.qr(right.transpose(-1, -2).double(), mode='reduced')
    singular = torch.linalg.svdvals(rl @ rr.transpose(-1, -2))
    generator = torch.Generator(device=left.device).manual_seed(seed)
    a = torch.randn(ql.shape, device=left.device, dtype=torch.float64, generator=generator)
    b = torch.randn(qr.shape, device=left.device, dtype=torch.float64, generator=generator)
    a = torch.linalg.qr(a, mode='reduced')[0]
    b = torch.linalg.qr(b, mode='reduced')[0]
    root = singular.sqrt()[..., :, None]
    return (root * a.transpose(-1, -2)).to(left.dtype), (root * b.transpose(-1, -2)).to(right.dtype), singular


class OperatorMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, count, active, condition, seed):
        assert condition in CONDITIONS
        super().__init__(leaf, base, previous, incoming, None, count, active, 'old_rows', None)
        self.operator_condition = condition
        self.native_coefficient_count = self.coefficients.numel()
        self.attention_names = [n.removesuffix('.out_proj.weight') for n in active if n.endswith('.out_proj.weight')]
        device = self.coefficients.device
        with torch.no_grad():
            old_f, old_g = factor_pairs({n: previous[n].to(device) for n in active}, self.attention_names)
            new_f, new_g = factor_pairs({n: incoming[n].to(device) for n in active}, self.attention_names)
            scale = (count / (count + 1) ** 2) ** .5
            self.residual_left, self.residual_right = (old_f - new_f) * scale, (old_g - new_g) * scale
            random_left, random_right, singular = random_spectrum_factors(self.residual_left,
                self.residual_right, seed + 1009 * count)
            self.residual_singular_values = singular.cpu()
            if condition == 'operator_random':
                self.residual_left, self.residual_right = random_left, random_right
        self.gate_count = len(self.attention_names) * 16
        assert self.gate_count == 512
        self.coefficients = torch.nn.Parameter(torch.cat([self.coefficients.detach(),
            torch.zeros(self.gate_count, device=device)]))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        parameters = super().compose(c[:self.native_coefficient_count])
        left, right = factor_pairs(parameters, self.attention_names)
        gates = c[self.native_coefficient_count:]
        if self.operator_condition == 'operator_zero':
            gates = gates * 0
        x = torch.cat([left, self.residual_left], dim=-2)
        y = torch.cat([right, gates[:, None, None] * self.residual_right], dim=-2)
        if self.operator_condition == 'operator_zero':
            with torch.no_grad():
                a, b = FactorSumProjection.apply(x, y, 32)
                assert torch.equal(a, left) and torch.equal(b, right)
            # With gates fixed to zero, the exact projection map is identity.
            return parameters
        fitted_left, fitted_right = FactorSumProjection.apply(x, y, 32)
        return restore_factors(parameters, self.attention_names, fitted_left, fitted_right)
