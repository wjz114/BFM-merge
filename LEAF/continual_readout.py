"""Linear EEG–text readout and the two-level coefficient objective (design v1.2)."""
import torch
from torch.nn import functional as F


def pool_branches(tokens, bins=1):
    # LEAF batches are fixed-length within task; both branches contain only full windows.
    pieces = [part.mean(1) for branch in tokens.chunk(2, dim=1)
              for part in branch.tensor_split(bins, dim=1)]
    return F.normalize(torch.cat(pieces, dim=-1), dim=-1)


def center(x):
    return x - x.mean(-1, keepdim=True)


def scores(h, W, prototypes):
    return h @ W.T @ center(prototypes.T)


def targets(logits, labels, support):
    q = center(logits)
    rms = q[support].square().mean().sqrt().clamp_min(1e-6)
    one_hot = F.one_hot(labels, logits.shape[-1]).to(logits)
    return (q / rms + center(one_hot)) / 2, rms


def class_weights(y):
    counts = y.bincount()
    return 1 / (counts[y] * (counts > 0).sum()).to(torch.float)


def cold_start(h, P, q, y, ridge=0.001):
    # Normal equations applied without constructing the full Hessian of vec(W).
    weights = class_weights(y).to(h)[:, None] / P.shape[0]
    A = P - P.mean(0)
    rhs = A.T @ (weights * q).T @ h
    W = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm2 = residual.square().sum()
    initial = norm2.sqrt().item()
    trace = [initial]
    for _ in range(20):
        if trace[-1] <= initial * 1e-7:
            break
        Hd = A.T @ (weights * (h @ direction.T @ A.T)).T @ h + ridge * direction
        step = norm2 / (direction * Hd).sum()
        W = W + step * direction
        residual = residual - step * Hd
        new_norm2 = residual.square().sum()
        direction = residual + new_norm2 / norm2 * direction
        norm2 = new_norm2
        trace.append(norm2.sqrt().item())
    return W, trace


def calibrate(Wold, tasks, ridge=0.001):
    """tasks contains (features, prototypes, targets, sample weights) with total mass 1."""
    W = Wold.detach().requires_grad_()
    for _ in range(3):
        loss = sum((w * (scores(h, W, P) - q).square().mean(-1)).sum()
                   for h, P, q, w in tasks) + ridge * (W - Wold).square().sum()
        grad, = torch.autograd.grad(loss, W, create_graph=True)
        W = W - grad / (2 * (1 + ridge))
    return W


def query_loss(new_scores, new_y, old_scores, old_y, previous_scores, epsilon=0.):
    # Query positions are sampled class-balanced; reference uses identical positions.
    new_ce = F.cross_entropy(new_scores, new_y)
    change = F.cross_entropy(old_scores, old_y) - F.cross_entropy(previous_scores, old_y)
    return new_ce + (change - epsilon).relu(), change


def merge_tensors(base, old_delta, new_delta, theta):
    return {k: base[k] + theta[i, 0] * old_delta[k] + theta[i, 1] * new_delta[k]
            for i, k in enumerate(old_delta)}
