"""Compress a bounded history jointly with the calibrated current model.

Each parameter row retains two scaled memory vectors. The current model is
an additional retained anchor, not a reconstruction of original expert means.
The sqrt(count) anchor weight is a fixed heuristic, not a covariance identity.
"""
import math

import torch

RANK = 2
CONDITIONS = ('joint_memory', 'complement_memory')
RELATIVE_ENERGY_FLOOR = 1e-12


@torch.no_grad()
def compress_rows(dictionary, anchor, condition):
    """FP64 projection and rank-two compression; return FP32 scaled vectors.

    dictionary: rows x 4 x width; anchor: rows x width. The energy floor uses
    the original dictionary energy and is invariant to orthogonal slot mixing.
    """
    assert condition in CONDITIONS
    assert dictionary.dtype == anchor.dtype == torch.float64
    assert dictionary.ndim == 3 and dictionary.shape[1] == 4
    assert anchor.shape == (dictionary.shape[0], dictionary.shape[2])
    energy = dictionary.square().sum((1, 2))
    norm = anchor.norm(dim=1, keepdim=True)
    q = anchor / norm.clamp_min(torch.finfo(torch.float64).tiny)
    residual = dictionary
    if condition == 'complement_memory':
        for _ in range(2):
            residual = residual - (residual * q[:, None]).sum(2, keepdim=True) * q[:, None]
    gram = residual @ residual.transpose(1, 2)
    values, vectors = torch.linalg.eigh(gram)
    top_values = values[:, -RANK:].flip(-1).clamp_min(0)
    keep = top_values > RELATIVE_ENERGY_FLOOR * energy[:, None]
    rotation = vectors[:, :, -RANK:].flip(-1).transpose(1, 2)
    kept = torch.where(keep[:, :, None], rotation @ residual, 0.).float()
    stored = kept.double()
    alignment = (stored * q[:, None]).sum(2).abs() / stored.norm(dim=2).clamp_min(1e-300)
    projected_energy = residual.square().sum((1, 2))
    stats = {
        'dictionary_squared_norm': float(energy.sum()),
        'projected_dictionary_squared_norm': float(projected_energy.sum()),
        'anchor_removed_squared_norm': float((energy - projected_energy).sum()),
        'retained_squared_norm': float(stored.square().sum()),
        'discarded_eigenvalue_sum': float(values.clamp_min(0).sum() - torch.where(keep, top_values, 0.).sum()),
        'max_stored_anchor_cosine_abs': float(alignment.max()),
        'retained_rank_counts': torch.bincount(keep.sum(1), minlength=3).tolist(),
    }
    return kept, stats


@torch.no_grad()
def update_memory(history, base, previous, incoming, current, count, active, condition):
    """Update only after calibration; never reload previous original experts."""
    assert condition in CONDITIONS and count >= 0
    state_bytes = sum(x.numel() * x.element_size() for x in history.values())
    if count == 0:
        assert previous is None
        assert all(torch.count_nonzero(x) == 0 for x in history.values())
        return {'first_expert_memory_zero': True, 'state_bytes': state_bytes}
    total = {'dictionary_squared_norm': 0., 'projected_dictionary_squared_norm': 0.,
             'anchor_removed_squared_norm': 0., 'retained_squared_norm': 0.,
             'discarded_eigenvalue_sum': 0., 'max_stored_anchor_cosine_abs': 0.,
             'retained_rank_counts': [0, 0, 0], 'state_bytes': state_bytes}
    for name in active:
        # Match the operative FP32 native delta before doing geometry in FP64.
        old = (previous[name] - base[name]).double()
        new = (incoming[name] - base[name]).double()
        anchor = (current[name] - base[name]).double()
        rows = old.shape[0] if old.ndim else 1
        dictionary = torch.cat([
            math.sqrt(count) * old.reshape(rows, 1, -1),
            history[name].reshape(RANK, rows, -1).transpose(0, 1).double(),
            new.reshape(rows, 1, -1)], dim=1)
        kept, stats = compress_rows(dictionary, anchor.reshape(rows, -1), condition)
        history[name].copy_(kept.transpose(0, 1).reshape_as(history[name]))
        for key in total:
            if key == 'state_bytes':
                continue
            if key == 'retained_rank_counts':
                total[key] = [a + b for a, b in zip(total[key], stats[key])]
            elif key == 'max_stored_anchor_cosine_abs':
                total[key] = max(total[key], stats[key])
            else:
                total[key] += stats[key]
    assert all(torch.isfinite(x).all() for x in history.values())
    return total
