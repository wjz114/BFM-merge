"""Calibrate affine merge coefficients using source-centered loss gradients.

This is an exact tangent-subtracted objective, not a nonnegative divergence
for a nonlinear network. Original experts stay frozen and arrive causally.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger


CONDITIONS = ('balanced_ce', 'expert_centered', 'output_kl')


def source_gradient(merger, wrapper, x, labels, weights, source):
    """Evaluate at source weights, pull its native gradient through merge axes."""
    parameters = merger.compose()
    for name in merger.active:
        p = parameters[name]
        parameters[name] = source[name] + (p - p.detach())
    logits = merger.logits(wrapper, x, parameters)
    loss = (F.cross_entropy(logits, labels, reduction='none') * weights[labels]).mean()
    gradient, = torch.autograd.grad(loss, merger.coefficients)
    return logits.detach(), loss.detach(), gradient.detach()


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device):
    assert condition in CONDITIONS
    merger = StreamingMerger(leaf, base, previous, incoming, None, count,
                            plan['active'], 'old_rows', None)
    c = merger.coefficients
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c], lr=plan['learning_rate'])
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
            target, source_ce, anchor = source_gradient(merger, wrappers[task], batch,
                labels, data.weights[task], data.experts[task])
            logits = merger.logits(wrappers[task], batch)
            ce = (F.cross_entropy(logits, labels, reduction='none') * data.weights[task][labels]).mean()
            kl = (F.kl_div(logits.log_softmax(-1), target.log_softmax(-1),
                  reduction='none', log_target=True).sum(-1) * data.weights[task][labels]).mean()
            objective = kl if condition == 'output_kl' else ce
            gradient, = torch.autograd.grad(objective, c)
            direction = gradient - anchor if condition == 'expert_centered' else gradient
            assert torch.isfinite(objective) and torch.isfinite(direction).all()
            optimizer.zero_grad(set_to_none=True)
            c.grad = direction
            optimizer.step()
            losses.append(float(objective.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            diagnostics.append(dict(task=task, round=rnd + 1, source_ce=float(source_ce),
                student_ce=float(ce.detach()), output_kl=float(kl.detach()),
                source_gradient_norm=float(anchor.norm()), student_gradient_norm=float(gradient.norm()),
                used_gradient_norm=float(direction.norm()), source_student_gradient_dot=float(torch.dot(anchor, gradient)),
                source_gradient_hash=array_hash(anchor.cpu().numpy())))
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
    record = dict(updates=len(schedule), training_examples=len(schedule) * bs,
        source_gradient_examples=len(schedule) * bs, source_backward_calls=len(schedule),
        student_backward_calls=len(schedule), schedule_hash=array_hash(np.asarray(schedule, dtype=np.int64)),
        round_student_losses=rounds, gradient_records=diagnostics, coefficient_count=mean.numel(),
        source_count=len(data.experts), source_active_parameter_bytes=sum(v.numel() * v.element_size()
            for expert in data.experts.values() for v in expert.values()), export_checks=checks,
        training_seconds=time.monotonic() - started, coefficient_min=float(mean.min()), coefficient_max=float(mean.max()))
    return state, dict(snapshots=torch.stack(snapshots), mean=mean), record
