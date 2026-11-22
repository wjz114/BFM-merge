"""Fixed grouped expert spaces and projection of native-coordinate Adam proposals."""
import json
from pathlib import Path

import torch


class GroupedBasis:
    def __init__(self, q, shape, permutation=None):
        self.q, self.shape, self.permutation = q, tuple(shape), permutation
        self.groups, self.rank, self.width = q.shape
        assert q.numel() == self.rank*torch.tensor(shape).prod().item()

    def encode(self, value):
        flat = value.flatten()
        if self.permutation is not None:
            flat = flat[self.permutation]
        return (self.q @ flat.reshape(self.groups, self.width, 1)).squeeze(-1)

    def decode(self, coefficients):
        flat = (coefficients.unsqueeze(1) @ self.q).flatten()
        if self.permutation is not None:
            result = torch.empty_like(flat)
            result[self.permutation] = flat
            flat = result
        return flat.reshape(self.shape)


class ProjectedAdam:
    """Full Adam moments; fixed-space, norm-restored proposals; accumulated coordinates.

    Coefficients only store displacement. Adam never operates in their basis.
    This is not a memory-saving optimizer or a convergence guarantee.
    """
    def __init__(self, named_parameters, records, outdir, snapshot_steps, **adam_args):
        self.names = [n for n, _ in named_parameters]
        self.params = [p for _, p in named_parameters]
        assert self.names == list(records)
        self.optimizer = torch.optim.Adam(self.params, **adam_args)
        self.initial = [p.detach().clone() for p in self.params]
        self.bases, self.coordinates = [], []
        for (_, parameter), record in zip(named_parameters, records.values()):
            q = record['q'].to(device=parameter.device, dtype=parameter.dtype)
            perm = record['permutation']
            perm = None if perm is None else perm.to(parameter.device)
            assert tuple(record['shape']) == tuple(parameter.shape)
            self.bases.append(GroupedBasis(q, record['shape'], perm))
            self.coordinates.append(torch.zeros(q.shape[:2], device=parameter.device, dtype=parameter.dtype))
        self.outdir, self.snapshot_steps = Path(outdir), set(snapshot_steps)
        self.steps, self.statistics = 0, []

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        before = [p.detach().clone() for p in self.params]
        self.optimizer.step()
        coefficients, proposed_squares, projected_squares = [], [], []
        for p, old, basis in zip(self.params, before, self.bases):
            proposal = p-old
            encoded = basis.encode(proposal)
            projected = basis.decode(encoded)
            coefficients.append(encoded)
            proposed_squares.append(proposal.double().square().sum())
            projected_squares.append(projected.double().square().sum())
        proposed = torch.stack(proposed_squares).sum().sqrt()
        projected = torch.stack(projected_squares).sum().sqrt()
        # The zero-projection case stays at the current coordinates and is logged.
        scale = float(proposed/projected) if projected > 0 else 0.
        assert torch.isfinite(proposed) and torch.isfinite(projected) and scale >= 0
        actual_squares = []
        for p, old, initial, basis, beta, update in zip(
                self.params, before, self.initial, self.bases, self.coordinates, coefficients):
            beta.add_(update, alpha=scale)
            p.copy_(initial+basis.decode(beta))
            actual_squares.append((p.double()-old.double()).square().sum())
        actual = torch.stack(actual_squares).sum().sqrt()
        self.steps += 1
        row = {'step': self.steps, 'adam_proposal_norm': float(proposed),
               'projected_proposal_norm': float(projected), 'restoration_scale': scale,
               'actual_parameter_step_norm': float(actual),
               'actual_to_proposal_norm_ratio': float(actual/proposed) if proposed > 0 else None,
               'zero_projection': bool(projected == 0)}
        assert torch.isfinite(actual)
        self.statistics.append(row)
        with (self.outdir/'projection_dynamics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if self.steps in self.snapshot_steps:
            torch.save({n: b.detach().cpu().clone() for n, b in zip(self.names, self.coordinates)},
                       self.outdir/f'projection_coordinates_{self.steps}.pt')

    @torch.no_grad()
    def audit_average(self, snapshot_steps):
        """Compare the native weight mean with the mean of its stored coordinates."""
        snapshots = [torch.load(self.outdir/f'projection_coordinates_{s}.pt', map_location='cpu', weights_only=True)
                     for s in snapshot_steps]
        maximum, squared, displacement_squared = 0., 0., 0.
        for name, p, initial, basis in zip(self.names, self.params, self.initial, self.bases):
            beta = torch.stack([d[name].double() for d in snapshots]).mean(0).to(p)
            expected = initial+basis.decode(beta)
            difference = p.double()-expected.double()
            maximum = max(maximum, float(difference.abs().max()))
            squared += float(difference.square().sum())
            displacement_squared += float((p.double()-initial.double()).square().sum())
        return {'max_abs_weight_mean_vs_coordinate_mean': maximum,
                'squared_difference': squared,
                'relative_l2_difference': (squared/max(displacement_squared, 1e-30))**.5,
                'scope': 'Averaging rounded weights and averaging FP32 coordinates need not be bitwise identical. This measures the discrepancy; the exported primary state is the unchanged native six-checkpoint weight mean.'}


def toy_checks(directory):
    """Independent projector, basis rotation and retained-step-norm checks."""
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261120)
    q = torch.linalg.qr(torch.randn(3, 7, 3, generator=generator, dtype=torch.float64)).Q.transpose(-1, -2)
    permutation = torch.randperm(21, generator=generator)
    x = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    errors = []
    for perm in [None, permutation]:
        basis = GroupedBasis(q, x.shape, perm)
        projected = basis.decode(basis.encode(x))
        xp = x.flatten() if perm is None else x.flatten()[perm]
        independent = torch.stack([qq.T @ torch.linalg.lstsq(qq.T, xx, driver='gelsd').solution
                                   for qq, xx in zip(q, xp.reshape(3, 7))]).flatten()
        if perm is not None:
            restored = torch.empty_like(independent); restored[perm] = independent; independent = restored
        errors.append(float((projected.flatten()-independent).abs().max()))
        assert errors[-1] < 1e-12
    rotation = torch.linalg.qr(torch.randn(3, 3, 3, generator=generator, dtype=torch.float64)).Q
    initial = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    parameters = [torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.clone())]
    optimizers = []
    for i, qb in enumerate([q, rotation @ q]):
        out = Path(directory)/str(i); out.mkdir(parents=True, exist_ok=False)
        records = {'w': {'q': qb, 'shape': list(initial.shape), 'permutation': permutation}}
        optimizers.append(ProjectedAdam([('w', parameters[i])], records, out, [4], lr=1e-3, weight_decay=0))
    for _ in range(4):
        grad = torch.randn(3, 7, generator=generator, dtype=torch.float64)
        for p, optimizer in zip(parameters, optimizers):
            p.grad = grad.clone(); optimizer.step()
    rotation_error = float((parameters[0]-parameters[1]).abs().max())
    assert rotation_error < 1e-12
    norm_error = max(abs(row['actual_to_proposal_norm_ratio']-1) for o in optimizers for row in o.statistics)
    assert norm_error < 1e-10
    for p, o in zip(parameters, optimizers):
        delta = p.detach()-initial
        assert torch.allclose(o.bases[0].decode(o.bases[0].encode(delta)), delta, atol=1e-12, rtol=1e-12)
    return {'independent_projector_max_abs_error': max(errors),
            'basis_rotation_trajectory_max_abs_error': rotation_error,
            'actual_step_norm_relative_error': norm_error,
            'four_step_displacement_in_fixed_span': True,
            'scope': 'FP64 toy geometry checks; no claim of bitwise native FP32 rotation invariance.'}
