"""Project the actual optimizer proposal in an old/new merge row metric.

Halfspaces are local calibration-loss constraints, not finite-step retention
guarantees. The projection is a GEM-style construction, not a novelty claim.
"""
import numpy as np
import torch
from scipy.optimize import minimize


CONDITIONS = ('simultaneous_mean_adam', 'euclidean_step_projection',
              'native_metric_step_projection')


class RowGeometry:
    def __init__(self, base, previous, incoming, active, damping=1e-4):
        self.n = len(active)
        grams, row_tensors, slices = [], [], []
        offset = 0
        for j, name in enumerate(active):
            rows = base[name].shape[0] if base[name].ndim else 1
            old = (previous[name] - base[name]).double().reshape(rows, -1)
            new = (incoming[name] - base[name]).double().reshape(rows, -1)
            gram = torch.empty(rows, 2, 2, dtype=torch.float64)
            gram[:, 0, 0] = old.square().sum(1)
            gram[:, 1, 1] = new.square().sum(1)
            gram[:, 0, 1] = gram[:, 1, 0] = (old * new).sum(1)
            grams.append(gram)
            row_tensors.extend([j] * rows)
            slices.append((name, offset, offset + rows))
            offset += rows
        self.rows = offset
        self.count = 2 * self.n + 2 * self.rows
        self.row_tensors = torch.tensor(row_tensors, dtype=torch.int64)
        self.slices = slices
        self.gram = torch.cat(grams)
        self.ridge = damping * self.gram.diagonal(dim1=1, dim2=2).sum(1) / 2
        self.zero_rows = self.ridge == 0
        self.ridge[self.zero_rows] = 1.
        self.native_metric = self.gram + self.ridge[:, None, None] * torch.eye(2, dtype=torch.float64)
        self.native_inverse = torch.linalg.inv(self.native_metric)
        self.identity = torch.eye(2, dtype=torch.float64).expand(self.rows, 2, 2)
        assert torch.isfinite(self.native_inverse).all()

    def collapse(self, step):
        step = step.detach().cpu().double()
        assert step.shape == (self.count,)
        old = step[:self.n][self.row_tensors] + step[2 * self.n + self.rows:]
        new = step[self.n:2 * self.n][self.row_tensors] + step[2 * self.n:2 * self.n + self.rows]
        return torch.stack([old, new], dim=1)

    def row_gradients(self, gradients):
        gradients = gradients.detach().cpu().double()
        assert gradients.ndim == 2 and gradients.shape[1] == self.count
        return torch.stack([gradients[:, 2 * self.n + self.rows:],
                            gradients[:, 2 * self.n:2 * self.n + self.rows]], dim=2)

    def lift(self, correction):
        assert correction.shape == (self.rows, 2)
        step = torch.zeros(self.count, dtype=torch.float64)
        step[2 * self.n:2 * self.n + self.rows] = correction[:, 1]
        step[2 * self.n + self.rows:] = correction[:, 0]
        return step

    def matrices(self, condition):
        assert condition in CONDITIONS
        if condition == 'native_metric_step_projection':
            return self.native_metric, self.native_inverse
        return self.identity, self.identity


def project_proposal(proposal, gradients, metric, inverse):
    """CPU FP64 primal projection via a task-sized nonnegative dual QP."""
    assert proposal.dtype == gradients.dtype == metric.dtype == inverse.dtype == torch.float64
    assert proposal.device.type == gradients.device.type == metric.device.type == 'cpu'
    raised = torch.einsum('rij,trj->tri', inverse, gradients)
    norms2 = (gradients * raised).sum((1, 2))
    assert torch.all(norms2 >= 0)
    keep = norms2 > 0
    before = (gradients * proposal).sum((1, 2))
    norm2 = float(torch.einsum('ri,rij,rj->', proposal, metric, proposal))
    assert norm2 >= 0
    record = dict(before_margins=before.tolist(), projected=False, solver_iterations=0,
                  zero_gradient_tasks=int((~keep).sum()), proposal_metric_norm=norm2 ** .5)
    if norm2 == 0 or not bool(keep.any()) or bool(torch.all(before >= 0)):
        record.update(after_margins=before.tolist(), correction_metric_norm=0.,
                      normalized_min_slack=0., normalized_complementarity=0.)
        return proposal.clone(), record
    norms = norms2[keep].sqrt()
    g = gradients[keep] / norms[:, None, None]
    z = raised[keep] / norms[:, None, None]
    scale = norm2 ** .5
    gram = (g.flatten(1) @ z.flatten(1).T).numpy()
    gram = (gram + gram.T) / 2
    margins = ((g * proposal).sum((1, 2)) / scale).numpy()
    assert np.allclose(np.diag(gram), 1., atol=1e-9, rtol=1e-9)
    result = minimize(lambda x: .5 * x @ gram @ x + margins @ x,
        np.zeros(len(margins)), jac=lambda x: gram @ x + margins,
        method='SLSQP', bounds=[(0., None)] * len(margins),
        options=dict(ftol=1e-12, maxiter=200, disp=False))
    dual = result.x
    slack = gram @ dual + margins
    complementarity = float(np.max(np.abs(dual * slack)))
    assert result.success, result.message
    assert dual.min() >= -1e-10 and slack.min() >= -2e-6, (dual, slack)
    assert complementarity <= 2e-6 * (1 + np.linalg.norm(dual)), complementarity
    correction = scale * torch.einsum('t,tri->ri', torch.from_numpy(dual), z)
    projected = proposal + correction
    actual_slack = ((g * projected).sum((1, 2)) / scale).numpy()
    assert np.max(np.abs(actual_slack - slack)) < 1e-8
    correction_norm = float(torch.einsum('ri,rij,rj->', correction, metric, correction)) ** .5
    record.update(projected=True, after_margins=(gradients * projected).sum((1, 2)).tolist(),
        solver_iterations=int(result.nit), normalized_min_slack=float(slack.min()),
        normalized_complementarity=complementarity, correction_metric_norm=correction_norm,
        correction_to_proposal_metric_norm=correction_norm / scale,
        dual_multipliers=dual.tolist())
    return projected, record
