"""Row-wise native Adam spaces with a parameter-rounding floor for history."""
import torch

from streaming_history_sketch_merge import RANK, residual_directions, update_sketch


@torch.no_grad()
def orthonormal_rows(directions, minimum_norm=None):
    vectors = directions.double()
    basis = []
    for index, vector in enumerate(vectors.unbind(1)):
        original_norm = vector.norm(dim=-1, keepdim=True)
        residual = vector.clone()
        for _ in range(2):
            for q in basis:
                residual -= (residual * q).sum(-1, keepdim=True) * q
        norm = residual.norm(dim=-1, keepdim=True)
        threshold = 1e-6 * original_norm
        if minimum_norm is not None:
            threshold = torch.maximum(threshold, minimum_norm[:, index:index + 1].double())
        keep = (norm > threshold) & (norm > 0)
        basis.append(torch.where(keep, residual / norm.clamp_min(1e-30), 0.))
    return torch.stack(basis, dim=1).float()


@torch.no_grad()
def build_bases(base, previous, incoming, history, count, active, condition):
    assert condition in ('bilateral', 'history_sketch', 'random_sketch')
    records, total_bytes, ranks, maximum_error = {}, 0, {}, 0.
    kept_history_slots = 0
    for j, name in enumerate(active):
        old, new = previous[name] - base[name], incoming[name] - base[name]
        rows = old.shape[0] if old.ndim else 1
        directions = torch.stack([old.reshape(rows, -1), new.reshape(rows, -1)], dim=1)
        if condition == 'bilateral':
            q = orthonormal_rows(directions)
        else:
            h, _ = residual_directions(history[name], old, new, count)
            # A unit basis must not promote parameter-rounding noise to a free
            # direction. The extra slots must exceed this absolute row scale.
            reference_norm = sum(v[name].double().reshape(rows, -1).norm(dim=1)
                                 for v in (base, previous, incoming))
            floor = 8 * torch.finfo(torch.float32).eps * reference_norm
            minimum = torch.stack([torch.zeros_like(floor), torch.zeros_like(floor), floor, floor], dim=1)
            real_q = orthonormal_rows(torch.cat([directions, h], dim=1), minimum)
            support = real_q[:, 2:].square().sum(-1) > .5
            kept_history_slots += int(support.sum())
            if condition == 'history_sketch':
                q = real_q
            else:
                generator = torch.Generator(device='cpu').manual_seed(27143 + count * 1000 + j)
                z = torch.randn(h.shape, generator=generator)
                z *= support.unsqueeze(-1)
                q = orthonormal_rows(torch.cat([directions, z], dim=1))
                assert torch.equal(q[:, 2:].square().sum(-1) > .5, support)
        nonzero = q.square().sum(-1) > .5
        gram = q @ q.transpose(1, 2)
        error = float((gram - torch.diag_embed(nonzero.float())).abs().max())
        assert error <= 2e-5
        maximum_error = max(maximum_error, error)
        rank, counts = torch.unique(nonzero.sum(-1), return_counts=True)
        for a, b in zip(rank.tolist(), counts.tolist()):
            ranks[str(a)] = ranks.get(str(a), 0) + b
        records[name] = {'q': q, 'shape': list(old.shape), 'permutation': None}
        total_bytes += q.numel() * q.element_size()
    audit = {'orthogonality_max_abs_error': maximum_error, 'per_row_effective_rank_counts': ranks,
             'padded_basis_bytes': total_bytes, 'condition': condition,
             'retained_history_coordinate_slots': kept_history_slots,
             'history_precision_floor': '8*eps_FP32*(row_norm(base)+row_norm(previous)+row_norm(incoming))',
             'scope': 'Reorthogonalized FP64 row directions exported in FP32. Historical residuals below '
                      'a native-parameter rounding scale are discarded before normalization. Random '
                      'orientation has exactly the same historical rank/support for the current model. '
                      'Models evolve differently across arms; no cross-arm rank-equality claim. Fixed CPU RNG.'}
    return records, audit
