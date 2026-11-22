"""Shared input-column corrections with a matched raw and row-complement form."""
import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger


def eligible(name, value):
    return name.endswith('weight') and value.ndim == 2 and min(value.shape) > 2


@torch.no_grad()
def row_plane(old, new):
    qa = old / old.norm(dim=1, keepdim=True).clamp_min(1e-30)
    residual = new - (new * qa).sum(1, keepdim=True) * qa
    norm = residual.norm(dim=1, keepdim=True)
    qb = residual / norm.clamp_min(1e-30)
    qb = torch.where(norm > 1e-6 * new.norm(dim=1, keepdim=True), qb, 0.)
    return qa, qb


def column_correction(old, new, coefficients, basis=None):
    centered = coefficients - coefficients.mean(dim=1, keepdim=True)
    result = old * centered[0] + new * centered[1]
    if basis is not None:
        for _ in range(2):
            for q in basis:
                result = result - (result * q).sum(1, keepdim=True) * q
    return result


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, history, count, active, condition, projections):
        assert condition in ('raw_columns', 'projected_columns')
        super().__init__(leaf, base, previous, incoming, {}, count, active, 'old_rows', projections)
        self.condition = condition
        self.bilateral_count = self.coefficients.numel()
        self.column_slices, self.column_bases = {}, {}
        pieces = [self.coefficients.detach()]
        offset = self.bilateral_count
        for name in active:
            if not eligible(name, base[name]):
                continue
            old, new = self.directions[name][:2]
            width = old.shape[1]
            self.column_slices[name] = slice(offset, offset + 2 * width)
            self.column_bases[name] = row_plane(old, new) if condition == 'projected_columns' else None
            pieces.append(torch.zeros(2 * width, device=old.device))
            offset += 2 * width
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.bilateral_count])
        for name, section in self.column_slices.items():
            old, new = self.directions[name][:2]
            correction = column_correction(old, new, c[section].view(2, -1), self.column_bases[name])
            result[name] = result[name] + correction
        return result
