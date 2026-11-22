"""Predictive-score statistics for the bounded historical expert dictionary."""
import torch
import torch.nn.functional as F

from streaming_response_sketch import StreamingMerger as ResponseMerger
from streaming_response_sketch import update_sketch as update_response_sketch


def predictive_probe(logits, signs, sample_weights):
    """An unbiased random VJP factor of class-balanced predictive Fisher.

    Independent, zero-mean signs remove cross-example and cross-class terms.
    Probabilities and sampling weights are held fixed during differentiation.
    True classes enter only through the existing sampling-balance weights.
    """
    logp = F.log_softmax(logits, dim=-1)
    scale = (logp.detach().exp() * sample_weights.detach()[:, None] / logits.shape[0]).sqrt()
    return (scale * signs * logp).sum()


def observe_predictive(merger, parameters, logits, signs, sample_weights):
    """Collect one score VJP without accumulating coefficient gradients."""
    assert not merger.responses.enabled
    assert merger.coefficients.grad is None
    names = merger.responses.active
    score = predictive_probe(logits, signs, sample_weights)
    gradients = torch.autograd.grad(score, [parameters[name] for name in names],
                                    retain_graph=True, allow_unused=True)
    observed = 0
    for name, gradient in zip(names, gradients):
        if gradient is not None:
            merger.responses.observe(name, gradient)
            observed += 1
    assert merger.coefficients.grad is None
    return observed


class StreamingMerger(ResponseMerger):
    def __init__(self, leaf, base, previous, incoming, history, count, active, condition, projections):
        assert condition in ('energy_history', 'predictive_history')
        mapped = 'energy_history' if condition == 'energy_history' else 'response_history'
        super().__init__(leaf, base, previous, incoming, history, count, active, mapped, projections)
        self.condition = condition
        # Statistics are supplied explicitly from autograd.grad. CE has no hooks.
        self.responses.enabled = False


@torch.no_grad()
def update_sketch(history, incoming, base, active, condition, statistics=None):
    assert condition in ('energy_history', 'predictive_history')
    mapped = 'energy_history' if condition == 'energy_history' else 'response_history'
    result = update_response_sketch(history, incoming, base, active, mapped, statistics)
    if statistics is not None and condition == 'predictive_history':
        result['criterion'] = 'class_balanced_predictive_score_response'
    return result
