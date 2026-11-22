"""CPU dense-matrix oracle independent of the low-rank training projection."""
import torch


def pairs(state, names):
    f, g = [], []
    for name in names:
        weight = state[name + '.in_proj_weight'].double()
        bias = state[name + '.in_proj_bias'].double()
        for h in range(8):
            a = slice(32 * h, 32 * (h + 1))
            q = torch.cat([weight[a], bias[a, None]], 1)
            k = weight[256 + a.start:256 + a.stop]
            f.append(q); g.append(k)
        for h in range(8):
            a = slice(512 + 32 * h, 512 + 32 * (h + 1))
            value = torch.cat([weight[a], bias[a, None]], 1)
            output = state[name + '.out_proj.weight'][:, 32 * h:32 * (h + 1)].double().T
            f.append(value); g.append(output)
    return torch.stack(f), torch.stack(g)


def independent_raw(base, previous, incoming, mean, active):
    n = len(active)
    rows_total = sum(base[k].shape[0] if base[k].ndim else 1 for k in active)
    out = {k: v.clone() for k, v in base.items()}
    offset = 2 * n
    for j, name in enumerate(active):
        rows = base[name].shape[0] if base[name].ndim else 1
        shape = (-1,) + (1,) * (base[name].ndim - 1) if base[name].ndim else ()
        old, new = previous[name] - base[name], incoming[name] - base[name]
        value = base[name] + mean[j] * old + mean[n + j] * new
        value = value + mean[offset:offset + rows].reshape(shape) * new
        value = value + mean[offset + rows_total:offset + rows_total + rows].reshape(shape) * old
        out[name] = value
        offset += rows
    assert offset + rows_total == 347666
    return out


@torch.no_grad()
def audit(base, previous, incoming, native, payload, active, count, condition):
    c = payload['mean']
    raw = independent_raw(base, previous, incoming, c, active)
    names = [n.removesuffix('.out_proj.weight') for n in active if n.endswith('.out_proj.weight')]
    changed = {n + suffix for n in names for suffix in ('.in_proj_weight', '.in_proj_bias', '.out_proj.weight')}
    rest_error = max(float((raw[n] - native[n]).abs().max()) for n in active if n not in changed)
    assert rest_error < 2e-6
    f, g = pairs(raw, names)
    old_f, old_g = pairs(previous, names)
    new_f, new_g = pairs(incoming, names)
    rf, rg = payload['residual_left'].double(), payload['residual_right'].double()
    residual = rf.transpose(-1, -2) @ rg
    expected_residual = count / (count + 1) ** 2 * (old_f - new_f).transpose(-1, -2) @ (old_g - new_g)
    target_norm = expected_residual.flatten(1).norm(dim=1)
    if condition != 'operator_random':
        residual_error = (residual - expected_residual).flatten(1).norm(dim=1)
        assert bool((residual_error <= 2e-6 * target_norm + 1e-12).all())
    else:
        # Equal nonzero singular spectra; this control changes orientation only.
        expected_s = torch.linalg.svdvals(expected_residual)
        actual_s = torch.linalg.svdvals(residual)
        residual_error = (actual_s - expected_s).norm(dim=1)
        assert bool((residual_error <= 2e-6 * target_norm + 1e-12).all())
    nf, ng = pairs(native, names)
    actual = nf.transpose(-1, -2) @ ng
    gates = c[347666:].double()
    if condition == 'operator_zero':
        assert torch.count_nonzero(gates) == 0
        expected = f.transpose(-1, -2) @ g
        factor_error = max(float((raw[n] - native[n]).abs().max()) for n in active)
        assert factor_error < 2e-6
    else:
        unprojected = f.transpose(-1, -2) @ g + gates[:, None, None] * residual
        u, s, vh = torch.linalg.svd(unprojected, full_matrices=False)
        expected = (u[..., :32] * s[..., None, :32]) @ vh[..., :32, :]
        gauge = f @ u[..., :32]
        expected_f = gauge @ u[..., :32].transpose(-1, -2)
        expected_g = torch.linalg.solve(gauge.transpose(-1, -2), s[..., :32, None] * vh[..., :32, :])
        factor_error = max(float((expected_f - nf).abs().max()), float((expected_g - ng).abs().max()))
        assert factor_error < 5e-5, factor_error
    operator_error = float((actual - expected).abs().max())
    relative = float((actual - expected).norm() / expected.norm())
    assert operator_error < 2e-5 and relative < 1e-5, (operator_error, relative)
    return dict(nonattention_max_error=rest_error, operator_max_abs_error=operator_error,
        operator_relative_error=relative, native_factor_max_abs_error=factor_error,
        residual_or_spectrum_max_relative_error=float((residual_error / target_norm.clamp_min(1e-20)).max()))
