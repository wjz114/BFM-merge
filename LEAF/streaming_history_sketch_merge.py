"""Bounded row-wise task-vector sketches expose lost historical directions.

The sketch is a rank-two incremental approximation across original experts,
not an SVD truncation of the current merged weight matrix. It is training-only
state. Replay calibration learns coefficients and folds one ordinary model.
"""
import math

import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_bilateral_merge import collect_split


RANK = 2


@torch.no_grad()
def update_sketch(history, incoming, base, active):
    """Append the arriving original vector, truncate each row's task covariance.

    H has two vector slots per parameter row. H.T @ H approximates the sum
    of outer products of original expert rows. The update never uses a merged
    checkpoint and does not revisit earlier experts.
    """
    retained, dropped = 0., 0.
    for name in active:
        delta = incoming[name] - base[name]
        rows = delta.shape[0] if delta.ndim else 1
        old = history[name].reshape(RANK, rows, -1)
        appended = torch.cat([old, delta.reshape(1, rows, -1)], dim=0).transpose(0, 1)
        gram = appended @ appended.transpose(1, 2)
        values, vectors = torch.linalg.eigh(gram.double())
        rotation = vectors[:, :, -RANK:].flip(-1).transpose(1, 2).float()
        kept = rotation @ appended
        history[name].copy_(kept.transpose(0, 1).reshape_as(history[name]))
        retained += float(values[:, -RANK:].clamp_min(0).sum())
        dropped += float(values[:, :-RANK].clamp_min(0).sum())
    return {'retained_squared_norm': retained, 'discarded_squared_norm_this_update': dropped,
            'state_bytes': sum(x.numel() * x.element_size() for x in history.values())}


@torch.no_grad()
def residual_directions(sketch, old, new, count):
    """Remove the old/new row span, retaining at most two historical directions."""
    rows = old.shape[0] if old.ndim else 1
    a, b = old.reshape(rows, -1), new.reshape(rows, -1)
    qa = a / a.norm(dim=1, keepdim=True).clamp_min(1e-30)
    rb = b - (b * qa).sum(1, keepdim=True) * qa
    rb_norm = rb.norm(dim=1, keepdim=True)
    qb = rb / rb_norm.clamp_min(1e-30)
    qb = torch.where(rb_norm > 1e-6 * b.norm(dim=1, keepdim=True), qb, 0.)
    h = sketch.reshape(RANK, rows, -1).transpose(0, 1) / math.sqrt(count)
    original_norm = h.norm(dim=2, keepdim=True)
    for q in (qa, qb):
        h = h - (h * q[:, None, :]).sum(2, keepdim=True) * q[:, None, :]
    h = torch.where(h.norm(dim=2, keepdim=True) > 1e-6 * original_norm, h, 0.)
    # Recombine the two residual slots into orthogonal row vectors. This has
    # the same span but gives a meaningful norm-matched orientation control.
    gram = h @ h.transpose(1, 2)
    values, vectors = torch.linalg.eigh(gram.double())
    rotation = vectors.flip(-1).transpose(1, 2).float()
    h = rotation @ h
    values = values.flip(-1).clamp_min(0)
    h = torch.where((values > 1e-12 * values[:, :1]).unsqueeze(-1), h, 0.)
    return h, (qa, qb)


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition, projections):
        assert condition in ('history_sketch', 'random_sketch')
        super().__init__(leaf, base, previous, incoming, history_sum, count, active, 'old_rows', projections)
        self.condition = condition
        self.bilateral_count = self.coefficients.numel()
        self.sketch_slices, self.sketch_directions = {}, {}
        pieces = [self.coefficients.detach()]
        offset = self.bilateral_count
        for j, name in enumerate(active):
            old, new = self.directions[name][:2]
            h, plane = residual_directions(history_sum[name].to(old.device), old, new, count)
            rows = h.shape[0]
            norms = h.norm(dim=2, keepdim=True)
            if condition == 'random_sketch':
                generator = torch.Generator(device=old.device).manual_seed(27143 + count * 1000 + j)
                z = torch.randn(h.shape, device=old.device, generator=generator)
                for q in plane:
                    z = z - (z * q[:, None, :]).sum(2, keepdim=True) * q[:, None, :]
                q0 = z[:, 0] / z[:, 0].norm(dim=1, keepdim=True).clamp_min(1e-30)
                z[:, 1] -= (z[:, 1] * q0).sum(1, keepdim=True) * q0
                h = z * (norms / z.norm(dim=2, keepdim=True).clamp_min(1e-30))
            self.sketch_directions[name] = h.transpose(0, 1).reshape(RANK, *old.shape)
            self.sketch_slices[name] = slice(offset, offset + RANK * rows)
            offset += RANK * rows
            pieces.append(torch.zeros(RANK * rows, device=old.device))
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.bilateral_count])
        for name, direction in self.sketch_directions.items():
            shape = (RANK, -1) + (1,) * (direction.ndim - 2) if direction.ndim > 1 else (RANK,)
            result[name] = result[name] + (c[self.sketch_slices[name]].view(shape) * direction).sum(0)
        return result
