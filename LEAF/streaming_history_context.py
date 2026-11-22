"""Arrival-only expert refinement under centered historical merge contexts."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier


CONDITIONS = ('source_ce', 'fixed_context', 'history_context', 'isotropic_context', 'native_after')


def historical_offset(history, active, selected):
    """Scaled leave-one-out change of the original historical expert mean."""
    count = len(history)
    assert count and 0 <= selected < count
    if count == 1:
        return {n: torch.zeros_like(history[0][n]) for n in active}
    scale = count / ((count + 1) * (count - 1))
    return {n: (torch.stack([s[n].double() for s in history]).mean(0).float()
                - history[selected][n]) * scale for n in active}


def isotropic_offset(offset, seed):
    """Independent directions with each parameter tensor's exact target norm."""
    device = next(iter(offset.values())).device
    generator = torch.Generator(device=device).manual_seed(seed)
    result, maximum = {}, 0.
    for n, value in offset.items():
        target = value.norm()
        if float(target) == 0:
            result[n] = torch.zeros_like(value)
        else:
            noise = torch.randn(value.shape, device=device, dtype=value.dtype, generator=generator)
            result[n] = noise * (target / noise.norm())
            maximum = max(maximum, float((result[n].norm() - target).abs() / target))
    assert maximum < 3e-6
    return result, maximum


def flatten(parameters, active):
    return torch.cat([parameters[n].detach().reshape(-1).cpu() for n in active])


def from_flat(flat, template, active):
    result = {n: v.detach().cpu().clone() for n, v in template.items()}
    start = 0
    for n in active:
        size = template[n].numel()
        result[n] = flat[start:start + size].reshape(template[n].shape).clone()
        start += size
    assert start == flat.numel()
    return result


def adapt_stage(leaf, initial, previous, history, data, tasks, plan, condition, device):
    assert condition in CONDITIONS and len(tasks) >= 2
    count, active = len(tasks) - 1, plan['active']
    assert len(history) == count
    leaf.load_state_dict(initial, strict=True)
    parameters = dict(leaf.named_parameters())
    for n, p in parameters.items():
        p.requires_grad_(n in active)
        p.grad = None
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([parameters[n] for n in active], lr=plan['adapt_learning_rate'])
    old = {n: previous[n].to(device) for n in active}
    alpha = 1 / (count + 1)
    rng = np.random.default_rng([plan['seed'], count + 1, 83117])
    task_rng = np.random.default_rng([plan['seed'], count + 1, 36119])
    selected_order = rng.permutation(count)
    batch_rngs = {t: np.random.default_rng([plan['seed'], count + 1,
        plan['tasks'].index(t), 71933]) for t in tasks}
    bs = plan['batch_size']
    rounds = plan['adapt_rounds']
    per_round = plan['adapt_steps_per_round']
    assert per_round == 8
    schedule, losses, contexts, snapshot_hashes = [], [], [], []
    average_sum = torch.zeros(sum(parameters[n].numel() for n in active), dtype=torch.float64)
    start_time = time.monotonic()
    for rnd in range(rounds):
        selected = int(selected_order[rnd % count])
        offset = historical_offset(history, active, selected)
        iso_seed = int(np.random.SeedSequence([plan['seed'], count + 1, rnd, 68291]).generate_state(1)[0])
        isotropic, norm_error = isotropic_offset(offset, iso_seed)
        # All arms construct the same offsets; only the selected condition applies them.
        context = dict(round=rnd + 1, selected_history_task=tasks[selected],
            original_history_tasks=list(tasks[:-1]), isotropic_seed=iso_seed,
            tensor_norm_relative_error=norm_error,
            historical_norm=float(torch.stack([v.square().sum() for v in offset.values()]).sum().sqrt()),
            historical_hash=array_hash(flatten(offset, active)),
            isotropic_hash=array_hash(flatten(isotropic, active)))
        contexts.append(context)
        task_order = []
        while len(task_order) < per_round:
            task_order.extend(task_rng.permutation(len(tasks)).tolist())
        round_losses = []
        for step in range(per_round):
            task = tasks[task_order[step]] if condition == 'native_after' else tasks[-1]
            ids = batch_rngs[task].choice(len(data.train[task][1]), bs, replace=False)
            x, y = data.train[task]
            labels = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            contextual = condition in ('fixed_context', 'history_context', 'isotropic_context') and step % 4 == 3
            if contextual:
                sign = 1 if step == 3 else -1
                used_offset = offset if condition == 'history_context' else isotropic
                overrides = {'leaf.' + n: alpha * parameters[n] + (1 - alpha) * old[n]
                    + (sign * used_offset[n] if condition != 'fixed_context' else 0.) for n in active}
                logits = torch.func.functional_call(wrappers[task], overrides, (x[ids].to(device),), strict=False)
            else:
                logits = wrappers[task](x[ids].to(device))
            loss = (F.cross_entropy(logits, labels, reduction='none') * data.weights[task][labels]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            round_losses.append(float(loss.detach()))
            del logits, loss
            if contextual:
                del overrides
        losses.append(float(np.mean(round_losses)))
        if rnd + 1 in plan['adapt_average_rounds']:
            snapshot = flatten(parameters, active)
            assert torch.isfinite(snapshot).all()
            snapshot_hashes.append(array_hash(snapshot))
            average_sum += snapshot.double()
        del offset, isotropic
    average_count = len(snapshot_hashes)
    assert average_count == len(plan['adapt_average_rounds'])
    mean = (average_sum / average_count).float()
    state = from_flat(mean, initial, active)
    assert all(torch.equal(state[n], initial[n]) for n in set(initial) - set(active))
    for p in leaf.parameters():
        p.requires_grad_(False)
        p.grad = None
    leaf.load_state_dict(state, strict=True)
    record = dict(condition=condition, updates=len(schedule), training_examples=len(schedule) * bs,
        backward_calls=len(schedule), source_context_updates=(2 * rounds if condition in
            ('fixed_context', 'history_context', 'isotropic_context') else 0),
        schedule=schedule, schedule_hash=array_hash(np.asarray(schedule, dtype=np.int64)),
        round_losses=losses, contexts=contexts, snapshot_hashes=snapshot_hashes,
        average_count=average_count, mean_hash=array_hash(mean),
        native_parameter_count=mean.numel(), original_history_count=count,
        training_seconds=time.monotonic() - start_time)
    return state, dict(average_sum=average_sum, average_count=average_count), record
