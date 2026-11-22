"""Online contiguous merge tree and an equal-event sequential control."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger


CONDITIONS = ('binary_tree', 'matched_chain')


def events(stage):
    """All intervals use one-based, inclusive arrival positions."""
    result = []
    size, suffix = 2, stage & -stage
    while size <= suffix:
        result.append(((stage - size + 1, stage - size // 2),
                       (stage - size // 2 + 1, stage)))
        size *= 2
    if suffix < stage:
        result.append(((1, stage - suffix), (stage - suffix + 1, stage)))
    return result


def live_intervals(stage):
    """Current binary-counter blocks plus their calibrated cumulative prefixes."""
    blocks, start = [], 1
    for bit in range(stage.bit_length() - 1, -1, -1):
        size = 1 << bit
        if stage & size:
            blocks.append((start, start + size - 1))
            start += size
    return sorted(set(blocks + [(1, end) for _, end in blocks]))


def interval_key(interval):
    return '%02d_%02d' % tuple(interval)


def train_event(leaf, base, left, right, left_count, right_count, stage,
                data, tasks, plan, device, initial=None):
    merger = StreamingMerger(leaf, base, left, right, None, left_count,
                            plan['active'], 'old_rows', None)
    c, n = merger.coefficients, len(plan['active'])
    with torch.no_grad():
        if initial is None:
            c[:n].fill_(left_count / (left_count + right_count))
            c[n:2 * n].fill_(right_count / (left_count + right_count))
        else:
            c.copy_(initial.to(device))
    initial_saved = c.detach().cpu().clone()
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], stage,
        plan['tasks'].index(t), 217]).permutation(len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], stage, 931])
    schedule, snapshots, losses = [], [], []
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        round_losses = []
        for k in order_rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            labels = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = merger.logits(wrappers[task], x[ids].to(device), merger.compose(c))
            loss = (F.cross_entropy(logits, labels, reduction='none')
                    * data.weights[task][labels]).sum() / bs
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(c.grad).all()
            optimizer.step()
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            round_losses.append(float(loss.detach()))
        losses.append(float(np.mean(round_losses)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        functional = {t: merger.logits(wrappers[t], data.train[t][0][:bs].to(device),
                                     parameters).cpu() for t in tasks}
        leaf.load_state_dict(state, strict=True)
        checks = {}
        for t in tasks:
            native = wrappers[t](data.train[t][0][:bs].to(device)).cpu()
            check = compare(functional[t], native, data.train[t][1][:bs])
            assert check['max_abs_logit_error'] <= 1e-5 and check['changed_predictions'] == 0
            checks[t] = check
    record = dict(tasks=tasks, stage=stage, left_count=left_count, right_count=right_count,
        updates=len(schedule), training_examples=len(schedule) * bs,
        schedule_hash=array_hash(np.asarray(schedule, dtype=np.int64)),
        round_losses=losses, coefficient_count=mean.numel(), export_checks=checks,
        training_seconds=time.monotonic() - started)
    return state, dict(initial=initial_saved, snapshots=torch.stack(snapshots), mean=mean), record
