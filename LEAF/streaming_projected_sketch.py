"""Fixed per-row expert/history spaces for native-coordinate Adam proposals."""
import torch

from streaming_history_sketch_merge import RANK, residual_directions, update_sketch


@torch.no_grad()
def orthonormal_rows(directions):
    """Reorthogonalized FP64 Gram-Schmidt; FP32 rows with zero padding."""
    vectors = directions.double()
    basis = []
    for vector in vectors.unbind(1):
        original_norm = vector.norm(dim=-1, keepdim=True)
        residual = vector.clone()
        for _ in range(2):
            for q in basis:
                residual -= (residual * q).sum(-1, keepdim=True) * q
        norm = residual.norm(dim=-1, keepdim=True)
        keep = (norm > 1e-6 * original_norm) & (norm > 0)
        basis.append(torch.where(keep, residual / norm.clamp_min(1e-30), 0.))
    return torch.stack(basis, dim=1).float()


@torch.no_grad()
def build_bases(base, previous, incoming, history, count, active, condition):
    assert condition in ('bilateral', 'history_sketch', 'random_sketch')
    records, total_bytes, ranks, maximum_error = {}, 0, {}, 0.
    for j, name in enumerate(active):
        old, new = previous[name] - base[name], incoming[name] - base[name]
        rows = old.shape[0] if old.ndim else 1
        directions = torch.stack([old.reshape(rows, -1), new.reshape(rows, -1)], dim=1)
        if condition != 'bilateral':
            h, plane = residual_directions(history[name], old, new, count)
            if condition == 'random_sketch':
                generator = torch.Generator(device='cpu').manual_seed(27143 + count * 1000 + j)
                z = torch.randn(h.shape, generator=generator)
                for q in plane:
                    z -= (z * q[:, None, :]).sum(2, keepdim=True) * q[:, None, :]
                q0 = z[:, 0] / z[:, 0].norm(dim=1, keepdim=True).clamp_min(1e-30)
                z[:, 1] -= (z[:, 1] * q0).sum(1, keepdim=True) * q0
                h = z * (h.norm(dim=2, keepdim=True) / z.norm(dim=2, keepdim=True).clamp_min(1e-30))
            directions = torch.cat([directions, h], dim=1)
        q = orthonormal_rows(directions)
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
             'scope': 'FP64 reorthogonalization of native old/new rows and optional historical complement; '
                      'FP32 padded basis. Native Adam acts before projection. Random orientation uses a '
                      'fixed CPU generator; this is not bitwise identical to the earlier GPU coefficient control.'}
    return records, audit
