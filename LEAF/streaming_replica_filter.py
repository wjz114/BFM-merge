"""Filter a new-task innovation using centered same-task source replicas."""
import torch


CONDITIONS = ('innovation_filter', 'shifted_filter', 'source_soup')


def shift_rows(centered, tensor_index, task_index):
    rows, _, width = centered.shape
    if width == 1:
        return centered
    shifts = (torch.arange(rows) + 104729 * tensor_index + 8191 * task_index) % (width - 1) + 1
    indices = (torch.arange(width)[None] - shifts[:, None]) % width
    return centered.gather(2, indices[:, None].expand(-1, 3, -1))


def filter_rows(innovation, centered):
    """Apply (I + E.T E / tau)^-1, tau=||E||_F^2/2, independently per row."""
    assert innovation.dtype == centered.dtype == torch.float64
    assert centered.shape[1] == 3
    tau = centered.square().sum((1, 2)) / 2
    denominator = torch.where(tau > 0, tau.sqrt(), torch.ones_like(tau))
    normalized = centered / denominator[:, None, None]
    gram = normalized @ normalized.transpose(1, 2)
    rhs = (normalized @ innovation.unsqueeze(-1))
    solved = torch.linalg.solve(gram + torch.eye(3, dtype=torch.float64), rhs)
    correction = (normalized.transpose(1, 2) @ solved).squeeze(-1)
    result = innovation - correction
    assert torch.isfinite(result).all()
    return result


def prepare_incoming(base, previous, replicas, active, condition, task_index):
    assert condition in CONDITIONS and len(replicas) == 3
    mean, effective, audits = base.copy(), base.copy(), {}
    active_set = set(active)
    for model in replicas:
        assert model.keys() == base.keys()
        assert all(torch.equal(model[n], base[n]) for n in base if n not in active_set)
    for j, name in enumerate(active):
        shape = base[name].shape
        rows = shape[0] if len(shape) else 1
        values = torch.stack([model[name].double().reshape(rows, -1) for model in replicas], 1)
        center = values.mean(1)
        mean[name] = center.reshape(shape).float()
        if previous is None or condition == 'source_soup':
            effective[name] = mean[name]
            continue
        # Innovation starts from the exported FP32 soup, common to all arms.
        innovation = mean[name].double().reshape(rows, -1) - previous[name].double().reshape(rows, -1)
        centered = values - center[:, None]
        if condition == 'shifted_filter':
            centered = shift_rows(centered, j, task_index)
        filtered = filter_rows(innovation, centered)
        effective[name] = (previous[name].double().reshape(rows, -1) + filtered).reshape(shape).float()
        audits[name] = {'innovation_squared_norm': float(innovation.square().sum()),
                       'filtered_squared_norm': float(filtered.square().sum()),
                       'removed_squared_norm': float((innovation - filtered).square().sum()),
                       'replica_scatter_trace': float(centered.square().sum()),
                       'zero_scatter_rows': int((centered.square().sum((1, 2)) == 0).sum())}
    return mean, effective, audits
