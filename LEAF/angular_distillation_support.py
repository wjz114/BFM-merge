"""Conventional unit-feature distillation with optional fixed readout projection."""
import torch
import torch.nn.functional as F

from distillation_control_support import objective as original_objective, readout


CONDITIONS = ('balanced_ce', 'ce_angular_full_feature', 'ce_angular_prototype_contrast')


def auxiliary(logits, raw, teacher, condition, basis):
    residual = F.normalize(raw, dim=-1)-teacher['teacher_unit']
    if condition == 'ce_angular_prototype_contrast':
        residual = residual@basis.T
    else:
        assert condition == 'ce_angular_full_feature'
    return residual.square().mean(-1)


def objective(logits, raw, truth, weights, teacher, condition, scale, basis):
    ce = original_objective(logits, raw, truth, weights, teacher, 'balanced_ce', 0., basis)
    if condition == 'balanced_ce' or scale == 0:
        return ce
    assert condition in CONDITIONS
    return ce+scale*(auxiliary(logits, raw, teacher, condition, basis)*weights[truth]).mean()


def initial_gradient_scales(data, positions, weights, basis):
    z = data['initial_raw'][positions].double().requires_grad_(True)
    logits = readout(z, data['prototypes'].double())
    truth = data['truth'][positions]
    w = weights.double()[truth]
    teacher = {'teacher_unit': data['teacher_unit'][positions].double()}
    ce_gradient, = torch.autograd.grad((w*F.cross_entropy(logits, truth, reduction='none')).sum(), z, retain_graph=True)
    ce_rms = float(ce_gradient.square().sum(-1).mean().sqrt())
    result = {}
    for condition in CONDITIONS[1:]:
        loss = auxiliary(logits, z, teacher, condition, basis.double())
        gradient, = torch.autograd.grad((w*loss).sum(), z, retain_graph=True)
        rms = float(gradient.square().sum(-1).mean().sqrt())
        assert rms > 0 and ce_rms > 0 and torch.isfinite(gradient).all()
        scale = ce_rms/rms
        result[condition] = {'scale': scale, 'ce_raw_gradient_rms': ce_rms,
                             'aux_raw_gradient_rms': rms, 'scaled_aux_raw_gradient_rms': scale*rms,
                             'flattened_raw_gradient_cosine_with_ce': float((ce_gradient*gradient).sum()/(ce_gradient.norm()*gradient.norm())),
                             'weighted_auxiliary_mean': float((w*loss).mean().detach())}
    return result
