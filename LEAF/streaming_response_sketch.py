"""Compress historical expert rows using their observed loss responses.

The memory contains two original-expert combinations per parameter row. Its
selection statistic comes from existing calibration backward passes; it is a
minibatch loss-gradient statistic, not a predictive Fisher or an output bound.
"""
from functools import partial

import torch

from streaming_history_sketch_merge import RANK, StreamingMerger as HistoryMerger
from streaming_history_sketch_merge import update_sketch as update_energy_sketch


@torch.no_grad()
def dictionary(history, incoming, base, name):
    delta = incoming[name] - base[name]
    rows = delta.shape[0] if delta.ndim else 1
    return torch.cat([history[name].reshape(RANK, rows, -1),
                      delta.reshape(1, rows, -1)], dim=0).transpose(0, 1)


@torch.no_grad()
def select_rows(appended, response_covariance):
    """Return U.T B for the leading response eigenvectors, row by row.

    Exactly zero response uses the established parameter-energy criterion.
    The response criterion optimizes reconstruction of B g, with orthonormal
    latent columns U; it does not minimize the parameter reconstruction error.
    """
    assert appended.ndim == 3 and appended.shape[1] == RANK + 1
    assert response_covariance.shape == (*appended.shape[:2], RANK + 1)
    covariance = response_covariance.double()
    assert torch.isfinite(covariance).all()
    energy = (appended @ appended.transpose(1, 2)).double()
    zero = covariance.diagonal(dim1=-2, dim2=-1).sum(-1) == 0
    metric = torch.where(zero[:, None, None], energy, covariance)
    values, vectors = torch.linalg.eigh(metric)
    rotation = vectors[:, :, -RANK:].flip(-1).transpose(1, 2).float()
    kept = rotation @ appended
    retained_response = torch.einsum('rki,rij,rkj->r', rotation.double(), covariance, rotation.double())
    total_response = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    return kept, {
        'zero_response_rows': int(zero.sum()),
        'rows': appended.shape[0],
        'total_response_energy': float(total_response.sum()),
        'retained_response_energy': float(retained_response.sum()),
        'retained_parameter_squared_norm': float(kept.double().square().sum()),
        'discarded_metric_eigenvalue_sum': float(values[:, :-RANK].clamp_min(0).sum()),
    }


class ResponseStatistics:
    """Observe adjoints without modifying the calibration gradient."""
    def __init__(self, history, incoming, base, active, device):
        self.active = list(active)
        self.appended = {name: dictionary(history, incoming, base, name).to(device)
                         for name in self.active}
        self.covariance = {name: torch.zeros((value.shape[0], RANK + 1, RANK + 1),
                                            dtype=torch.float64, device=device)
                           for name, value in self.appended.items()}
        self.calls = {name: 0 for name in self.active}
        self.last_response = {}
        self.enabled = False

    @torch.no_grad()
    def observe(self, name, gradient):
        appended = self.appended[name]
        response = torch.einsum('rkd,rd->rk', appended, gradient.detach().reshape(appended.shape[0], -1))
        z = response.double()
        self.covariance[name].add_(z[:, :, None] * z[:, None, :])
        self.calls[name] += 1
        self.last_response[name] = response
        # Returning None leaves autograd's incoming gradient unchanged.

    def attach(self, parameters):
        if self.enabled and torch.is_grad_enabled():
            for name in self.active:
                if parameters[name].requires_grad:
                    parameters[name].register_hook(partial(self.observe, name))
        return parameters

    def export(self):
        return {'covariance': {name: value.detach().cpu() for name, value in self.covariance.items()},
                'hook_calls': dict(self.calls),
                'appended_dictionary_bytes': sum(x.numel() * x.element_size() for x in self.appended.values()),
                'response_covariance_bytes': sum(x.numel() * x.element_size() for x in self.covariance.values())}


class StreamingMerger(HistoryMerger):
    def __init__(self, leaf, base, previous, incoming, history, count, active, condition, projections):
        assert condition in ('energy_history', 'response_history')
        super().__init__(leaf, base, previous, incoming, history, count, active, 'history_sketch', projections)
        self.condition = condition
        self.responses = ResponseStatistics(history, incoming, base, active, self.coefficients.device)

    def compose(self, coefficients=None):
        return self.responses.attach(super().compose(coefficients))


@torch.no_grad()
def update_sketch(history, incoming, base, active, condition, statistics=None):
    if statistics is None:
        # The unchanged first expert has no calibration trajectory.
        result = update_energy_sketch(history, incoming, base, active)
        result['criterion'] = 'first_expert_energy_initialization'
        return result
    assert condition in ('energy_history', 'response_history')
    if condition == 'energy_history':
        result = update_energy_sketch(history, incoming, base, active)
        result['criterion'] = 'parameter_energy'
        return result
    totals = {'zero_response_rows': 0, 'rows': 0, 'total_response_energy': 0.,
              'retained_response_energy': 0., 'retained_parameter_squared_norm': 0.,
              'discarded_metric_eigenvalue_sum': 0.}
    for name in active:
        kept, record = select_rows(dictionary(history, incoming, base, name), statistics['covariance'][name])
        history[name].copy_(kept.transpose(0, 1).reshape_as(history[name]))
        for key in totals:
            totals[key] += record[key]
    totals.update(criterion='trajectory_minibatch_loss_response',
                  state_bytes=sum(x.numel() * x.element_size() for x in history.values()))
    return totals
