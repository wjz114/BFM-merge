"""Bilateral calibration with an original-expert Fisher consensus complement."""
import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger


@torch.no_grad()
def residual_row_direction(old, new, center):
    """Project in FP64, then scale surviving rows to the incoming row norm."""
    shape = old.shape
    rows = shape[0] if len(shape) else 1
    a, b, f = [v.double().reshape(rows, -1) for v in (old, new, center)]
    na, nb = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True)
    qa = a / na.clamp_min(1e-300)
    rb = b - (b * qa).sum(1, keepdim=True) * qa
    rb = rb - (rb * qa).sum(1, keepdim=True) * qa
    nr = rb.norm(dim=1, keepdim=True)
    qb = torch.where(nr > 1e-6 * nb, rb / nr.clamp_min(1e-300), 0.)
    result = f.clone()
    for _ in range(2):
        for q in (qa, qb):
            result.sub_((result * q).sum(1, keepdim=True) * q)
    norm = result.norm(dim=1, keepdim=True)
    keep = (norm > 1e-6 * f.norm(dim=1, keepdim=True)) & (nb > 0)
    # A width<=2 row has no complement to a generic bilateral row plane.
    # Its possible rank-deficient complement remains valid under this rule.
    result = torch.where(keep, result * (nb / norm.clamp_min(1e-300)), 0.)
    return result.reshape(shape).float()


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, moments, count, active,
                 condition, projections):
        assert condition in ('fisher_residual', 'shifted_fisher_residual')
        super().__init__(leaf, base, previous, incoming, {}, count, active,
                         'old_rows', projections)
        self.condition = condition
        self.bilateral_count = self.coefficients.numel()
        self.residual_slices, self.residual_directions = {}, {}
        pieces = [self.coefficients.detach()]
        offset = self.bilateral_count
        nonzero, total, max_cos, max_norm_error = 0, 0, 0., 0.
        for name in active:
            precision, product, uniform_sum = moments[name]
            mean = uniform_sum / (count + 1)
            center = torch.where(precision > 0,
                                 product / torch.where(precision > 0, precision, 1.), mean)
            old, new = previous[name] - base[name], incoming[name] - base[name]
            direction = residual_row_direction(old, new, center)
            rows = old.shape[0] if old.ndim else 1
            self.residual_slices[name] = slice(offset, offset + rows)
            offset += rows
            pieces.append(torch.zeros(rows, device=self.coefficients.device))
            self.residual_directions[name] = direction.to(self.coefficients.device)
            x = direction.double().reshape(rows, -1)
            nx = x.norm(dim=1)
            nonzero += int((nx > 0).sum())
            total += rows
            for value in (old, new):
                v = value.double().reshape(rows, -1)
                cosine = (x * v).sum(1).abs() / (nx * v.norm(dim=1)).clamp_min(1e-300)
                max_cos = max(max_cos, float(cosine.max()))
            target = new.double().reshape(rows, -1).norm(dim=1)
            relative = torch.where(nx > 0, (nx - target).abs() / target.clamp_min(1e-300), 0.)
            max_norm_error = max(max_norm_error, float(relative.max()))
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))
        self.residual_audit = {'nonzero_rows': nonzero, 'total_rows': total,
                               'max_abs_cosine_to_old_or_new': max_cos,
                               'max_relative_target_norm_error': max_norm_error}

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.bilateral_count])
        for name, direction in self.residual_directions.items():
            shape = (-1,) + (1,) * (direction.ndim - 1) if direction.ndim else ()
            result[name] = result[name] + c[self.residual_slices[name]].view(shape) * direction
        return result
