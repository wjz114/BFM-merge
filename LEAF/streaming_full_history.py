"""Causal row-history reference retaining every arrived original expert delta.

State and calibration coefficient counts grow with the number of seen tasks.
Only the materialized ordinary model is needed for inference.
"""
import math

import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger


def initialize_history(base, active):
    return {name: torch.empty((0, *base[name].shape), dtype=torch.float32) for name in active}


@torch.no_grad()
def update_memory(history, base, previous, incoming, current, count, active, condition):
    assert condition == 'full_history' and count >= 0
    assert all(value.shape[0] == count for value in history.values())
    for name in active:
        delta = incoming[name] - base[name]
        history[name] = torch.cat([history[name], delta[None]], dim=0)
    return {'history_count': count + 1, 'appended_original_only': True,
            'state_bytes': sum(value.numel() * value.element_size() for value in history.values())}


@torch.no_grad()
def residual_directions(sketch, old, new, count):
    """The frozen two-slot arithmetic generalized to every retained slot."""
    slots = sketch.shape[0]
    assert slots == count and count > 0
    rows = old.shape[0] if old.ndim else 1
    a, b = old.reshape(rows, -1), new.reshape(rows, -1)
    qa = a / a.norm(dim=1, keepdim=True).clamp_min(1e-30)
    rb = b - (b * qa).sum(1, keepdim=True) * qa
    rb_norm = rb.norm(dim=1, keepdim=True)
    qb = rb / rb_norm.clamp_min(1e-30)
    qb = torch.where(rb_norm > 1e-6 * b.norm(dim=1, keepdim=True), qb, 0.)
    h = sketch.reshape(slots, rows, -1).transpose(0, 1) / math.sqrt(count)
    original_norm = h.norm(dim=2, keepdim=True)
    for q in (qa, qb):
        h = h - (h * q[:, None, :]).sum(2, keepdim=True) * q[:, None, :]
    h = torch.where(h.norm(dim=2, keepdim=True) > 1e-6 * original_norm, h, 0.)
    gram = h @ h.transpose(1, 2)
    values, vectors = torch.linalg.eigh(gram.double())
    rotation = vectors.flip(-1).transpose(1, 2).float()
    h = rotation @ h
    values = values.flip(-1).clamp_min(0)
    h = torch.where((values > 1e-12 * values[:, :1]).unsqueeze(-1), h, 0.)
    return h, (qa, qb)


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, history, count, active, condition, projections):
        assert condition == 'full_history'
        assert all(value.shape[0] == count for value in history.values())
        super().__init__(leaf, base, previous, incoming, history, count, active, 'old_rows', projections)
        self.condition, self.slots = condition, count
        self.bilateral_count = self.coefficients.numel()
        self.sketch_slices, self.sketch_directions = {}, {}
        pieces = [self.coefficients.detach()]
        offset = self.bilateral_count
        for name in active:
            old, new = self.directions[name][:2]
            h, _ = residual_directions(history[name].to(old.device), old, new, count)
            rows = h.shape[0]
            self.sketch_directions[name] = h.transpose(0, 1).reshape(count, *old.shape)
            self.sketch_slices[name] = slice(offset, offset + count * rows)
            offset += count * rows
            pieces.append(torch.zeros(count * rows, device=old.device))
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.bilateral_count])
        for name, direction in self.sketch_directions.items():
            shape = (self.slots, -1) + (1,) * (direction.ndim - 2) if direction.ndim > 1 else (self.slots,)
            result[name] = result[name] + (c[self.sketch_slices[name]].view(shape) * direction).sum(0)
        return result
