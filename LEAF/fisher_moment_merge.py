"""Predictive Fisher moments and calibration from additive expert summaries.

Fisher weighting is an existing method. The experimental operator calibrates
uniform/Fisher summary directions afresh, without inheriting old calibrated
weights. It is a candidate to test, not an established novel contribution.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_residual_merge_v2 import StreamingMerger as IncomingMerger
from streaming_bilateral_merge import collect_split


def fisher_probe(logits, signs):
    """A random VJP whose squared gradient estimates predictive Fisher diagonal.

    E[sign_ic sign_jd] = 1[(i,c)=(j,d)] removes cross-example/class terms.
    Probabilities are detached; this is not the gradient of predictive entropy.
    """
    logp = F.log_softmax(logits, dim=-1)
    weights = (logp.detach().exp() / logits.shape[0]).sqrt() * signs
    return (weights * logp).sum()


def estimate_fisher(leaf, data, task, plan, device, examples=128):
    active = plan['active']
    parameters = dict(leaf.named_parameters())
    leaf.eval().requires_grad_(False)
    for name in active:
        parameters[name].requires_grad_(True)
    fisher = {n: torch.zeros_like(parameters[n]) for n in active}
    index = plan['tasks'].index(task)
    ids = np.random.default_rng([plan['seed'], index, 401]).permutation(512)[:examples]
    generator = torch.Generator(device=device).manual_seed(35713 + 1000 * plan['seed'] + index)
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    bs = plan['batch_size']
    assert examples % bs == 0
    for start in range(0, examples, bs):
        leaf.zero_grad(set_to_none=True)
        logits = wrapper(data.train[task][0][ids[start:start + bs]].to(device))
        signs = torch.randint(2, logits.shape, device=device, generator=generator).float().mul_(2).sub_(1)
        fisher_probe(logits, signs).backward()
        for name in active:
            g = parameters[name].grad
            assert g is not None and torch.isfinite(g).all(), name
            fisher[name].add_(g.detach().square(), alpha=bs / examples)
    raw = {n: value.cpu() for n, value in fisher.items()}
    norm = math.sqrt(sum(float(v.double().square().sum()) for v in raw.values()))
    assert norm > 0 and math.isfinite(norm)
    normalized = {n: v / norm for n, v in raw.items()}
    leaf.zero_grad(set_to_none=True)
    leaf.requires_grad_(False)
    return normalized, {'training_local_ids': ids.tolist(), 'ids_hash': array_hash(ids.astype(np.int64)),
                        'forward_examples': examples, 'backward_probes': examples // bs,
                        'raw_global_l2': norm, 'normalization': 'Per-expert global L2, matching a Fisher-merging option.',
                        'estimator': 'Model-predictive diagonal Fisher, one Rademacher VJP per batch; true labels unused.',
                        'diagonal_hashes': {n: array_hash(v.numpy()) for n, v in normalized.items()}}


@torch.no_grad()
def add_moments(moments, fisher, incoming, base, active, task_index, seed, shuffle=False):
    for j, name in enumerate(active):
        weights = fisher[name]
        rows = weights.shape[0] if weights.ndim else 1
        if shuffle:
            flat = weights.reshape(rows, -1)
            generator = torch.Generator().manual_seed(7919 + seed * 100000 + task_index * 1000 + j)
            offsets = torch.randint(flat.shape[1], (rows, 1), generator=generator)
            columns = (torch.arange(flat.shape[1])[None, :] + offsets) % flat.shape[1]
            weights = flat.gather(1, columns).reshape_as(weights)
        delta = (incoming[name] - base[name]).double()
        moments[name][0].add_(weights.double())
        moments[name][1].add_(weights.double() * delta)
        moments[name][2].add_(delta)


@torch.no_grad()
def moment_centers(moments, count):
    uniform, weighted = {}, {}
    for name, state in moments.items():
        precision, product, total = state
        mean = total / count
        denominator = torch.where(precision > 0, precision, 1.)
        center = torch.where(precision > 0, product / denominator, mean)
        assert torch.isfinite(center).all()
        uniform[name], weighted[name] = mean.float(), center.float()
    return uniform, weighted


class MomentMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, moments, count, active, condition, projections):
        assert condition in ('moment_restart', 'shuffled_restart')
        super().__init__(leaf, base, previous, incoming, {}, count, active, 'old_rows', projections)
        self.condition = condition
        mean, fisher = moment_centers(moments, count + 1)
        device = self.coefficients.device
        for name in active:
            delta = mean[name].to(device)
            correction = (fisher[name] - mean[name]).to(device)
            self.directions[name] = torch.stack([delta, correction, correction])
            self.retention_directions[name] = self.directions[name][0]
        with torch.no_grad():
            self.coefficients.zero_()
            self.coefficients[:len(active)] = 1.


def build_merger(leaf, base, previous, incoming, moments, count, active, condition, projections):
    if condition == 'incoming_rows':
        return IncomingMerger(leaf, base, previous, incoming, {}, count, active, condition)
    return MomentMerger(leaf, base, previous, incoming, moments, count, active, condition, projections)
