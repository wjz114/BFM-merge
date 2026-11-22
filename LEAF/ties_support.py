"""TIES task-vector preprocessing for the fixed EEG baseline comparison.

Reference: prateeky2806/ties-merging commit
44e7891fc84f3de7e4caa52664cd864ca3715e91, topk_values_mask,
resolve_sign(mass), and disjoint_merge(mean). Source and BSD-3-Clause license
are archived in runs/aligned_merge_v1/inputs/external/TIES_44e7891f.
"""
import time

import torch


@torch.no_grad()
def ties_vectors(flat, density):
    """Return disjoint expert vectors scaled so their mean equals TIES.

    Threshold comparison and exact-zero sign handling follow the reference,
    including the kth-value boundary and its global majority tie resolution.
    """
    if flat.ndim != 2 or not 0 < density < 1:
        raise ValueError('Expected K x D task vectors and density in (0,1)')
    experts, width = flat.shape
    kth = width - int(width * density)
    thresholds = torch.stack([row.abs().kthvalue(kth).values for row in flat])
    trimmed = flat * (flat.abs() >= thresholds[:, None])
    signs = trimmed.sum(0).sign()
    zero_count = int((signs == 0).sum())
    majority = signs.sum().sign()
    signs[signs == 0] = majority
    kept = trimmed * torch.where(signs[None] > 0, trimmed > 0, trimmed < 0)
    counts = (kept != 0).sum(0).clamp_min(1)
    expected_mean = kept.sum(0) / counts
    result = kept * (experts / counts.float())[None]
    error = (result.mean(0) - expected_mean).abs().max().item()
    info = {'requested_density': density, 'active_coordinates_per_expert': width,
            'thresholds': thresholds.tolist(),
            'trimmed_nonzero_fraction': (trimmed != 0).float().mean(1).tolist(),
            'sign_consistent_nonzero_fraction': (kept != 0).float().mean(1).tolist(),
            'zero_consensus_coordinates_before_tie_resolution': zero_count,
            'global_majority_sign': majority.item(), 'uniform_equals_disjoint_mean_max_abs': error,
            'original_vector_l2': flat.norm(dim=1).tolist(),
            'transformed_vector_l2': result.norm(dim=1).tolist()}
    return result, info


@torch.no_grad()
def transform_task_vectors(deltas, density):
    started = time.time()
    names = list(deltas)
    experts = deltas[names[0]].shape[0]
    # One global threshold per expert, across every active model parameter.
    flat = torch.cat([deltas[name].detach().cpu().reshape(experts, -1) for name in names], dim=1)
    transformed, info = ties_vectors(flat, density)
    result, offset = {}, 0
    for name in names:
        original = deltas[name]
        width = original[0].numel()
        result[name] = transformed[:, offset:offset + width].reshape(original.shape).to(original.device)
        offset += width
    info.update(reference_commit='44e7891fc84f3de7e4caa52664cd864ca3715e91',
                scope='all classification-active parameters, one threshold per expert; buffers/unused heads excluded',
                definition='topk magnitude, mass sign election with global-majority ties, disjoint mean, fixed scale 1',
                coefficient_definition='transformed delta_k = K * sign-consistent trimmed delta_k / nonzero contributor count per coordinate',
                seconds=time.time() - started)
    return result, info
