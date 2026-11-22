"""Fixed full-model distillation objectives and initial readout-gradient scaling."""
import torch
import torch.nn.functional as F


CONDITIONS = ('balanced_ce', 'ce_teacher_kl', 'ce_full_feature', 'ce_prototype_contrast')


def auxiliary(logits, raw, teacher, condition, basis):
    if condition == 'ce_teacher_kl':
        return F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher['teacher_logits'], dim=-1), reduction='none').sum(-1)
    residual = raw/teacher['initial_norm']-teacher['teacher_unit']
    if condition == 'ce_prototype_contrast':
        residual = residual@basis.T
    else:
        assert condition == 'ce_full_feature'
    return residual.square().mean(-1)


def objective(logits, raw, truth, weights, teacher, condition, scale, basis):
    ce = (F.cross_entropy(logits, truth, reduction='none')*weights[truth]).mean()
    if condition == 'balanced_ce' or scale == 0:
        return ce
    assert condition in CONDITIONS
    return ce+scale*(auxiliary(logits, raw, teacher, condition, basis)*weights[truth]).mean()


def readout(raw, prototypes):
    # LEAF normalizes its output, then the unchanged classifier normalizes again.
    return F.normalize(F.normalize(raw, dim=-1), dim=-1)@prototypes.T


def initial_gradient_scales(data, positions, weights, basis):
    """Match weighted per-example raw-output gradient RMS, once at initialization."""
    z = data['initial_raw'][positions].double().requires_grad_(True)
    prototypes = data['prototypes'].double()
    logits = readout(z, prototypes)
    truth = data['truth'][positions]
    sample_weights = weights.double()[truth]
    teacher = {k: data[k][positions].double() for k in ['initial_norm', 'teacher_unit', 'teacher_logits']}
    loss = (sample_weights*F.cross_entropy(logits, truth, reduction='none')).sum()
    ce_grad, = torch.autograd.grad(loss, z, retain_graph=True)
    ce_rms = float(ce_grad.square().sum(-1).mean().sqrt())
    assert ce_rms > 0
    result = {}
    for condition in CONDITIONS[1:]:
        per_example = auxiliary(logits, z, teacher, condition, basis.double())
        grad, = torch.autograd.grad((sample_weights*per_example).sum(), z, retain_graph=True)
        rms = float(grad.square().sum(-1).mean().sqrt())
        assert rms > 0 and torch.isfinite(grad).all()
        scale = ce_rms/rms
        cosine = float((ce_grad*grad).sum()/(ce_grad.norm()*grad.norm()))
        result[condition] = {'scale': scale, 'ce_raw_gradient_rms': ce_rms,
                             'aux_raw_gradient_rms': rms, 'scaled_aux_raw_gradient_rms': scale*rms,
                             'flattened_raw_gradient_cosine_with_ce': cosine,
                             'weighted_auxiliary_mean': float((sample_weights*per_example).mean().detach())}
    return result
