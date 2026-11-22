"""Fixed preprojection targets; shared loss scale separates target interventions."""
import numpy as np
import torch
import torch.nn.functional as F

CONDITIONS = ('balanced_ce', 'ce_prehead_paired', 'ce_prehead_mean', 'ce_prehead_shuffled')


def objective(logits, prehead, truth, class_weights, target, scale):
    weights = class_weights[truth]
    ce = (F.cross_entropy(logits, truth, reduction='none') * weights).mean()
    if scale == 0:
        return ce
    return ce + scale * ((prehead - target).square().mean(-1) * weights).mean()


def target_rows(cache, positions, condition):
    if condition in ('balanced_ce', 'ce_prehead_paired'):
        return cache['teacher_prehead'][positions]
    if condition == 'ce_prehead_mean':
        return cache['teacher_class_means'][cache['truth'][positions]]
    assert condition == 'ce_prehead_shuffled'
    return cache['teacher_prehead'][cache['shuffled_pool_positions'][positions]]


def make_target_controls(teacher, truth, fit_positions, n_classes, seed):
    """Shuffle unique rows within class AND fitting multiplicity, consistently."""
    multiplicities = torch.bincount(fit_positions, minlength=len(truth))
    rng = np.random.default_rng(seed)
    shuffled = torch.arange(len(truth))
    strata = []
    for label in range(n_classes):
        for count in torch.unique(multiplicities[truth == label]).tolist():
            if count == 0:
                continue
            ids = torch.nonzero((truth == label) & (multiplicities == count)).flatten()
            order = torch.from_numpy(rng.permutation(ids.numpy()))
            if len(order) > 1:
                shuffled[order] = order.roll(1)
                assert (shuffled[order] != order).all()
            strata.append({'class': label, 'multiplicity': count, 'unique_rows': len(ids),
                           'unchanged_singleton': len(ids) == 1})
    assert torch.equal(truth, truth[shuffled])
    assert torch.equal(multiplicities, multiplicities[shuffled])
    assert torch.equal(torch.sort(shuffled[fit_positions]).values, torch.sort(fit_positions).values)
    means = torch.stack([teacher[fit_positions[truth[fit_positions] == label]].double().mean(0)
                         for label in range(n_classes)]).float()
    assert torch.isfinite(means).all()
    return means, shuffled, {'strata': strata, 'class_and_multiplicity_preserved': True,
        'fitting_target_multiset_exact': True, 'one_fixed_target_per_unique_input': True,
        'unchanged_fitting_exposures': int((shuffled[fit_positions] == fit_positions).sum()),
        'total_fitting_exposures': len(fit_positions)}


def raw_ce_gradient(raw, truth, prototypes, weights):
    unit = F.normalize(raw, dim=-1)
    logits = F.normalize(unit, dim=-1) @ prototypes.T
    error = logits.softmax(-1) - F.one_hot(truth, prototypes.shape[0])
    grad = error @ prototypes
    return weights[truth, None] * (grad - unit * (grad * unit).sum(-1, keepdim=True)) / raw.norm(dim=-1, keepdim=True)
