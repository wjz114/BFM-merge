"""MLDG-style cross-task coefficient learning; an exploratory known-method adaptation.

The source-space merger remains the ordinary bilateral row merger. This module
changes its training objective, not its inference architecture. It is not a
claim that meta-learning or normalized gradient steps are novel.
"""
import torch


CONDITIONS = ('cross_task_meta', 'detached_lookahead', 'paired_ce')


def configure_attention():
    """All arms use the same attention implementation supporting double backward."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)


def normalized_step(gradient, boundaries, rate, epsilon):
    pieces = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        g = gradient[start:end]
        pieces.append(rate * g / (g.square().mean() + epsilon ** 2).sqrt())
    return torch.cat(pieces)


def objective_gradient(coefficients, support_loss, query_loss, condition,
                       boundaries, rate, epsilon):
    """Differentiate the stated two-task objective, including step normalization.

    query_loss is a callable of coefficients. Each task's batch contributes once
    per role in a round. The virtual step is discarded after the outer update.
    """
    assert condition in CONDITIONS
    second_order = condition == 'cross_task_meta'
    gradient, = torch.autograd.grad(support_loss, coefficients,
                                   create_graph=second_order, retain_graph=True)
    step = normalized_step(gradient, boundaries, rate, epsilon)
    if condition == 'paired_ce':
        virtual = coefficients
    elif second_order:
        virtual = coefficients - step
    else:
        virtual = coefficients - step.detach()
    query = query_loss(virtual)
    objective = (support_loss + query) * .5
    outer, = torch.autograd.grad(objective, coefficients)
    assert torch.isfinite(outer).all() and torch.isfinite(objective)
    record = {
        'support_ce': float(support_loss.detach()), 'query_ce': float(query.detach()),
        'objective': float(objective.detach()),
        'virtual_step_block_rms': [float(step[a:b].detach().square().mean().sqrt())
                                  for a, b in zip(boundaries[:-1], boundaries[1:])],
        'outer_gradient_norm': float(outer.detach().norm()),
    }
    return outer.detach(), record
