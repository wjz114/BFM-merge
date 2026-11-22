"""Exact minibatch generalized Gauss--Newton in the existing merge coordinates.

This is an optimization candidate, not a claim that GGN or distillation is new.
The deployed model remains an ordinary affine bilateral merged checkpoint.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger
from streaming_expert_centering import source_gradient


CONDITIONS = ('adam_ce', 'adam_kl', 'ggn_ce', 'ggn_kl')


def helmert(classes, device, dtype=torch.float64):
    h = torch.zeros(classes, classes - 1, device=device, dtype=dtype)
    for j in range(classes - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        h[:j + 1, j] = 1 / scale
        h[j + 1, j] = -(j + 1) / scale
    return h


def objective(logits, target, labels, weights, kl, double=False):
    z, e = (logits.double(), target.double()) if double else (logits, target)
    losses = (F.kl_div(z.log_softmax(-1), e.log_softmax(-1), reduction='none',
                      log_target=True).sum(-1) if kl else
              F.cross_entropy(z, labels, reduction='none'))
    return (losses * weights[labels]).mean()


def contrast_jacobian(logits, coefficients):
    h = helmert(logits.shape[1], logits.device, logits.dtype)
    scores = (logits @ h).flatten()
    rows = []
    for i in range(scores.numel()):
        grad, = torch.autograd.grad(scores[i], coefficients,
                                   retain_graph=i + 1 < scores.numel())
        rows.append(grad.detach())
    return torch.stack(rows).reshape(len(logits), h.shape[1], -1), h


def dual_ggn(jacobian, logits, target, labels, weights, kl, damping_ratio):
    """Solve (J' H J + lambda I) d = -J' g, in float64 output space."""
    batch, classes = logits.shape
    h = helmert(classes, logits.device)
    p = logits.detach().double().softmax(-1)
    q = target.detach().double().softmax(-1) if kl else F.one_hot(labels, classes).double()
    scale = weights[labels].double() / batch
    covariance = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    curvature = (h.T @ covariance @ h) * scale[:, None, None]
    gradient = ((p - q) @ h) * scale[:, None]
    eigenvalues, vectors = torch.linalg.eigh(curvature)
    assert torch.isfinite(eigenvalues).all() and eigenvalues.min() > 0
    root = (vectors * eigenvalues.sqrt()[:, None, :]) @ vectors.transpose(1, 2)
    invroot = (vectors * eigenvalues.rsqrt()[:, None, :]) @ vectors.transpose(1, 2)
    b = (root @ jacobian.double()).flatten(0, 1)
    v = (invroot @ gradient[:, :, None]).flatten()
    gram = b @ b.T
    damping = damping_ratio * gram.diag().mean()
    if float(damping) == 0:
        return torch.zeros(jacobian.shape[-1], device=logits.device, dtype=torch.float64), {
            'damping': 0., 'gradient_norm': 0., 'solve_relative_residual': 0.,
            'gradient_dot_step': 0., 'jacobian_rows': b.shape[0]}
    matrix = gram + damping * torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    alpha = torch.linalg.solve(matrix, v)
    direction = -(b.T @ alpha)
    g = b.T @ v
    residual = b.T @ (b @ direction) + damping * direction + g
    relative = float(residual.norm() / g.norm().clamp_min(1e-30))
    assert relative < 1e-7 and torch.isfinite(direction).all()
    return direction, dict(damping=float(damping), gradient_norm=float(g.norm()),
        solve_relative_residual=relative, gradient_dot_step=float(g @ direction),
        jacobian_rows=b.shape[0], curvature_min_eigenvalue=float(eigenvalues.min()))


def source_logits(merger, wrapper, batch, source, labels, weights):
    # Preserve the frozen original-expert forward/backward execution path exactly.
    target, _, _ = source_gradient(merger, wrapper, batch, labels, weights, source)
    return target


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device):
    assert condition in CONDITIONS
    merger = StreamingMerger(leaf, base, previous, incoming, None, count,
                            plan['active'], 'old_rows', None)
    c = merger.coefficients
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    is_ggn, kl = condition.startswith('ggn'), condition.endswith('kl')
    optimizer = None if is_ggn else torch.optim.Adam([c], lr=plan['learning_rate'])
    bs, stage = plan['batch_size'], count + 1
    permutations = {t: np.random.default_rng([plan['seed'], stage,
        plan['tasks'].index(t), 217]).permutation(512) for t in tasks}
    rng = np.random.default_rng([plan['seed'], stage, 931])
    snapshots, rounds, schedule, diagnostics = [], [], [], []
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        losses = []
        for k in rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            batch, labels = x[ids].to(device), y[ids].to(device)
            target = source_logits(merger, wrappers[task], batch, data.experts[task], labels, data.weights[task])
            logits = merger.logits(wrappers[task], batch)
            loss = objective(logits, target, labels, data.weights[task], kl, is_ggn)
            value = float(loss.detach())
            if is_ggn:
                detached = logits.detach()
                jacobian, _ = contrast_jacobian(logits, c)
                direction, info = dual_ggn(jacobian, detached, target, labels,
                    data.weights[task], kl, plan['damping_ratio'])
                del jacobian, logits, loss
                radius = plan['coefficient_step_rms_limit'] * math.sqrt(c.numel())
                norm = float(direction.norm())
                clip = min(1., radius / max(norm, 1e-30))
                direction = direction * clip
                slope = info['gradient_dot_step'] * clip
                before = c.detach().clone()
                accepted, trial_losses, used_scale = False, [], 0.
                for trial in range(plan['line_search_trials']):
                    factor = 0.5 ** trial
                    with torch.no_grad():
                        c.copy_(before + (factor * direction).to(c.dtype))
                        trial_logits = merger.logits(wrappers[task], batch)
                        trial_loss = float(objective(trial_logits, target, labels,
                            data.weights[task], kl, True))
                    trial_losses.append(trial_loss)
                    if math.isfinite(trial_loss) and trial_loss <= value + plan['armijo'] * factor * slope:
                        accepted, used_scale = True, factor
                        break
                if not accepted:
                    with torch.no_grad():
                        c.copy_(before)
                info.update(accepted=accepted, trial_losses=trial_losses, line_search_scale=used_scale,
                    unclipped_direction_norm=norm, radius=radius, direction_clip=clip,
                    realized_coefficient_step_norm=float((c.detach() - before).norm()),
                    line_search_forward_examples=len(trial_losses) * bs)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert torch.isfinite(c.grad).all()
                info = dict(gradient_norm=float(c.grad.norm()), jacobian_rows=1,
                            line_search_forward_examples=0, accepted=True)
                optimizer.step()
            assert math.isfinite(value) and torch.isfinite(c).all()
            losses.append(value)
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            diagnostics.append(dict(task=task, round=rnd + 1, objective=value, **info))
        rounds.append(float(np.mean(losses)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    with torch.no_grad():
        params = merger.compose(mean.to(device))
        functional = {t: merger.logits(wrappers[t], data.train[t][0][:bs].to(device), params).cpu() for t in tasks}
        leaf.load_state_dict(state, strict=True)
        checks = {t: compare(functional[t], wrappers[t](data.train[t][0][:bs].to(device)).cpu(),
                             data.train[t][1][:bs]) for t in tasks}
    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0 for v in checks.values())
    record = dict(updates=len(schedule), accepted_updates=sum(d['accepted'] for d in diagnostics),
        training_examples=len(schedule) * bs, source_forward_examples=len(schedule) * bs, source_backward_calls=len(schedule),
        source_gradient_examples=len(schedule) * bs,
        student_backward_calls=sum(d['jacobian_rows'] for d in diagnostics),
        line_search_forward_examples=sum(d['line_search_forward_examples'] for d in diagnostics),
        schedule_hash=array_hash(np.asarray(schedule, dtype=np.int64)), round_student_losses=rounds,
        gradient_records=diagnostics, coefficient_count=mean.numel(), source_count=len(data.experts),
        export_checks=checks, training_seconds=time.monotonic() - started,
        coefficient_min=float(mean.min()), coefficient_max=float(mean.max()))
    return state, dict(snapshots=torch.stack(snapshots), mean=mean), record
