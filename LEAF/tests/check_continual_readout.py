"""Finite differences through the same three-step support fit used by the pilot."""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continual_readout import calibrate, cold_start, merge_tensors, query_loss, scores


def check():
    torch.manual_seed(17)
    torch.set_default_dtype(torch.float64)
    base = {'weight': torch.randn(3, 4), 'bias': torch.randn(3)}
    old = {k: torch.randn_like(v) for k,v in base.items()}
    new = {k: torch.randn_like(v) for k,v in base.items()}
    theta = torch.full((2, 2), .5, requires_grad=True)
    S, Q = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    P = F.normalize(torch.randn(2, 3, 5), dim=-1)
    labels = torch.arange(3).repeat(2)
    target = torch.randn(2, 6, 3)
    Wold = torch.randn(5, 3)
    weights = torch.full((6,), 1/12)

    def objective(coeff, detach=False, active=True):
        params = merge_tensors(base, old, new, coeff)
        def encode(x):
            return F.normalize(x @ params['weight'].T + params['bias'], dim=-1)
        W = calibrate(Wold, [(h, p, q, weights) for h,p,q in zip(encode(S), P, target)])
        if detach:
            W = W.detach()
        new_scores, old_scores = [scores(h, W, p) for h,p in zip(encode(Q), P)]
        # Force the reference to be confidently right / wrong, away from the hinge.
        reference = (10 if active else -10) * F.one_hot(labels, 3).double()
        loss, change = query_loss(new_scores, labels, old_scores, labels, reference)
        assert bool(change > 0) == active
        return loss, F.cross_entropy(new_scores, labels)

    loss, new_ce = objective(theta)
    grad, = torch.autograd.grad(loss, theta, retain_graph=True)
    new_grad, = torch.autograd.grad(new_ce, theta)
    detached, _ = objective(theta, detach=True)
    detached_grad, = torch.autograd.grad(detached, theta)
    inactive, new_ce = objective(theta, active=False)
    inactive_grad, = torch.autograd.grad(inactive, theta, retain_graph=True)
    expected, = torch.autograd.grad(new_ce, theta)
    torch.testing.assert_close(inactive_grad, expected)
    torch.testing.assert_close(loss, detached, rtol=0, atol=0)
    assert (grad - detached_grad).abs().max() > 1e-6
    assert (grad - new_grad).abs().max() > 1e-6
    finite = torch.empty_like(theta)
    for step, value in zip(torch.eye(theta.numel()).reshape(-1, *theta.shape), finite.view(-1)):
        plus, _ = objective(theta + step * 1e-5)
        minus, _ = objective(theta - step * 1e-5)
        value.copy_(((plus - minus) / 2e-5).detach())
    torch.testing.assert_close(grad, finite, rtol=1e-4, atol=1e-9)

    # An independent dense solve checks class weights, ridge scale and centering.
    h = F.normalize(torch.randn(7, 3), dim=1)
    y = torch.tensor([0, 0, 0, 0, 1, 2, 2])
    q = torch.randn(7, 3)
    fitted, _ = cold_start(h, P[0], q, y)
    A = P[0] - P[0].mean(0)
    design = (A[None,:,:,None] * h[:,None,None,:]).reshape(21, 15)
    mass = (1 / (9 * y.bincount()[y].double())).repeat_interleave(3)
    normal = design.T @ (mass[:, None] * design)
    exact = torch.linalg.solve(normal + .001 * torch.eye(15), design.T @ (mass*q.flatten()))
    torch.testing.assert_close(fitted.flatten(), exact, atol=2e-6, rtol=1e-5)
    return {'tensor_coefficients_checked': grad.numel(), 'max_abs_error': (grad-finite).abs().max().item(),
            'max_relative_error': ((grad-finite).abs()/grad.abs()).max().item(),
            'full_vs_detached_max_grad_delta': (grad-detached_grad).abs().max().item(),
            'inactive_hinge_max_grad_delta': (inactive_grad-expected).abs().max().item(),
            'active_hinge_max_grad_delta': (grad-new_grad).abs().max().item()}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
