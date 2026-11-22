"""Simultaneous bilateral coefficients and shared native calibration residuals."""
import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_projected_sketch_v2 import orthonormal_rows


def project_complement(value, basis):
    rows = value.reshape(basis.shape[0], -1)
    result = rows
    # Two passes reduce the effect of casting the FP64 basis to native FP32.
    for _ in range(2):
        coordinates = (result[:, None, :] * basis).sum(-1)
        result = result - (coordinates[:, :, None] * basis).sum(1)
    return result.reshape_as(value)


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, count, active, condition, projections):
        assert condition in ('complement_hybrid', 'unprojected_hybrid', 'native_only')
        super().__init__(leaf, base, previous, incoming, {}, count, active, 'old_rows', projections)
        self.condition = condition
        self.coefficients.requires_grad_(condition != 'native_only')
        self.residual = {n: torch.nn.Parameter(torch.zeros_like(self.anchor[n])) for n in active}
        self.bases, self.hooks = {}, []
        self.basis_audit = {'padded_basis_bytes': 0, 'maximum_orthogonality_error': 0.,
                            'per_row_effective_rank_counts': {}, 'represented_span_dimensions': 0}
        if condition == 'complement_hybrid':
            for n in active:
                old, new = previous[n] - base[n], incoming[n] - base[n]
                rows = old.shape[0] if old.ndim else 1
                directions = torch.stack([old.reshape(rows, -1), new.reshape(rows, -1)], dim=1)
                q = orthonormal_rows(directions)
                nonzero = q.square().sum(-1) > .5
                error = float((q @ q.transpose(1, 2) - torch.diag_embed(nonzero.float())).abs().max())
                assert error < 2e-5
                self.basis_audit['maximum_orthogonality_error'] = max(self.basis_audit['maximum_orthogonality_error'], error)
                self.basis_audit['padded_basis_bytes'] += q.numel() * q.element_size()
                self.basis_audit['represented_span_dimensions'] += int(nonzero.sum())
                ranks, counts = torch.unique(nonzero.sum(1), return_counts=True)
                for r, c in zip(ranks.tolist(), counts.tolist()):
                    d = self.basis_audit['per_row_effective_rank_counts']
                    d[str(r)] = d.get(str(r), 0) + c
                q = q.to(self.coefficients.device)
                self.bases[n] = q
                self.hooks.append(self.residual[n].register_hook(lambda grad, basis=q: project_complement(grad, basis)))
        assert sum(r.numel() for r in self.residual.values()) == 25751552

    def compose(self, coefficients=None, residual=None):
        parameters = super().compose(coefficients)
        residual = self.residual if residual is None else residual
        for n in self.active:
            parameters[n] = parameters[n] + residual[n]
        return parameters

    @torch.no_grad()
    def constrain_residual(self):
        for name, q in self.bases.items():
            self.residual[name].copy_(project_complement(self.residual[name], q))

    @torch.no_grad()
    def residual_audit(self, residual=None):
        values = self.residual if residual is None else residual
        maximum = 0.
        for name, q in self.bases.items():
            rows = values[name].to(q.device).reshape(q.shape[0], -1)
            dot = (rows[:, None, :] * q).sum(-1)
            scale = rows.norm(dim=1, keepdim=True).clamp_min(1e-30)
            maximum = max(maximum, float((dot.abs() / scale).max()))
        return {'max_relative_row_projection': maximum,
                'squared_norm': sum(float(r.detach().double().square().sum()) for r in values.values())}

    @torch.no_grad()
    def materialize(self, coefficients=None, residual=None):
        parameters = self.compose(coefficients, residual)
        result = {name: parameters.get(name, value).detach().cpu().clone()
                  for name, value in self.base.items()}
        assert all(torch.isfinite(result[name]).all() for name in self.active)
        return result
