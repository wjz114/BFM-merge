"""Transport original-expert covariance in current-model anchored row bases.

Two full vector slots are supplemented by a small FP64 coordinate covariance.
The current native model supplies the third basis direction. Its variance and
cross terms survive reanchoring, although calibration uses only its complement.
"""
import torch

RANK = 2
RELATIVE_ENERGY_FLOOR = 1e-12


def initialize_memory(base, active):
    return {
        'basis': {n: torch.zeros((2, *base[n].shape), dtype=torch.float32) for n in active},
        'covariance': {n: torch.zeros((base[n].shape[0] if base[n].ndim else 1, 3, 3),
                                     dtype=torch.float64) for n in active}}


def state_bytes(memory):
    return {name + '_bytes': sum(x.numel() * x.element_size() for x in values.values())
            for name, values in memory.items()}


def unit_rows(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def covariance_factor(covariance, basis):
    values, vectors = torch.linalg.eigh(covariance)
    factor = values.clamp_min(0).sqrt().unsqueeze(-1) * vectors.transpose(1, 2)
    return factor @ basis


@torch.no_grad()
def compress_anchored(dictionary, anchor):
    assert dictionary.dtype == anchor.dtype == torch.float64
    assert dictionary.shape[1] == 4
    energy = dictionary.square().sum((1, 2))
    q = unit_rows(anchor)
    residual = dictionary
    for _ in range(2):
        residual = residual - (residual * q[:, None]).sum(2, keepdim=True) * q[:, None]
    values, vectors = torch.linalg.eigh(residual @ residual.transpose(1, 2))
    top = values[:, -2:].flip(-1).clamp_min(0)
    keep = top > RELATIVE_ENERGY_FLOOR * energy[:, None]
    directions = vectors[:, :, -2:].flip(-1).transpose(1, 2) @ residual
    stored = torch.where(keep[:, :, None], unit_rows(directions), 0.).float()
    basis = torch.cat([q[:, None], stored.double()], dim=1)
    gram = basis @ basis.transpose(1, 2)
    gvalues, gvectors = torch.linalg.eigh(gram)
    inverse_values = torch.where(gvalues > 1e-12 * gvalues[:, -1:],
                                 1. / gvalues.clamp_min(1e-300), 0.)
    inverse = (gvectors * inverse_values[:, None]) @ gvectors.transpose(1, 2)
    coordinates = inverse @ (basis @ dictionary.transpose(1, 2))
    covariance = coordinates @ coordinates.transpose(1, 2)
    projected = coordinates.transpose(1, 2) @ basis
    retained = projected.square().sum((1, 2))
    unanchored_values = torch.linalg.eigvalsh(dictionary @ dictionary.transpose(1, 2))
    reference = unanchored_values[:, -2:].clamp_min(0).sum(1)
    margin = (retained - reference) / energy.clamp_min(1e-300)
    assert float(margin.min()) >= -1e-7
    alignment = (stored.double() * q[:, None]).sum(2).abs()
    stats = {'dictionary_squared_norm': float(energy.sum()),
             'retained_squared_norm': float(retained.sum()),
             'discarded_squared_norm': float((energy - retained).sum()),
             'rank2_reference_retained_squared_norm': float(reference.sum()),
             'min_row_retention_margin_over_rank2_relative': float(margin.min()),
             'max_stored_anchor_cosine_abs': float(alignment.max()),
             'residual_rank_counts': torch.bincount(keep.sum(1), minlength=3).tolist()}
    return stored, covariance, stats


@torch.no_grad()
def decode_history(memory):
    history = {}
    for name, stored in memory['basis'].items():
        covariance = memory['covariance'][name][:, 1:, 1:]
        rows = covariance.shape[0]
        basis = stored.reshape(2, rows, -1).transpose(0, 1).double()
        factor = covariance_factor(covariance, basis)
        history[name] = factor.transpose(0, 1).reshape_as(stored).float()
    return history


@torch.no_grad()
def update_memory(memory, base, previous, incoming, current, count, active, condition):
    assert condition == 'anchored_covariance' and count >= 0
    sizes = state_bytes(memory)
    if count == 0:
        assert previous is None
        for name in active:
            assert torch.count_nonzero(memory['basis'][name]) == 0
            delta = (incoming[name] - base[name]).double()
            rows = delta.shape[0] if delta.ndim else 1
            memory['covariance'][name][:, 0, 0] = delta.reshape(rows, -1).square().sum(1)
        return {**sizes, 'state_bytes': sum(sizes.values()), 'first_expert_basis_zero': True,
                'first_expert_covariance_encoded_in_anchor': True}
    total = {'dictionary_squared_norm': 0., 'retained_squared_norm': 0.,
             'discarded_squared_norm': 0., 'rank2_reference_retained_squared_norm': 0.,
             'min_row_retention_margin_over_rank2_relative': float('inf'),
             'max_stored_anchor_cosine_abs': 0., 'residual_rank_counts': [0, 0, 0]}
    for name in active:
        old = (previous[name] - base[name]).double()
        new = (incoming[name] - base[name]).double()
        anchor = (current[name] - base[name]).double()
        rows = old.shape[0] if old.ndim else 1
        stored = memory['basis'][name].reshape(2, rows, -1).transpose(0, 1).double()
        previous_basis = torch.cat([unit_rows(old.reshape(rows, -1))[:, None], stored], dim=1)
        previous_factor = covariance_factor(memory['covariance'][name], previous_basis)
        dictionary = torch.cat([previous_factor, new.reshape(rows, 1, -1)], dim=1)
        directions, covariance, stats = compress_anchored(dictionary, anchor.reshape(rows, -1))
        memory['basis'][name].copy_(directions.transpose(0, 1).reshape_as(memory['basis'][name]))
        memory['covariance'][name].copy_(covariance)
        for key in total:
            if key == 'residual_rank_counts':
                total[key] = [a + b for a, b in zip(total[key], stats[key])]
            elif key == 'max_stored_anchor_cosine_abs':
                total[key] = max(total[key], stats[key])
            elif key == 'min_row_retention_margin_over_rank2_relative':
                total[key] = min(total[key], stats[key])
            else:
                total[key] += stats[key]
    assert all(torch.isfinite(x).all() for group in memory.values() for x in group.values())
    total.update(**sizes, state_bytes=sum(sizes.values()))
    return total
