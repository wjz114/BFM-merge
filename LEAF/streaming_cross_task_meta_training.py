"""Pair actual seen tasks and calibrate a static bilateral merge checkpoint."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger
from streaming_cross_task_meta import objective_gradient


def weighted_loss(merger, wrapper, batch, weights, coefficients):
    x, y = batch
    logits = merger.logits(wrapper, x, merger.compose(coefficients))
    return (F.cross_entropy(logits, y, reduction='none') * weights[y]).mean()


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan,
                condition, device):
    merger = StreamingMerger(leaf, base, previous, incoming, None, count,
                             plan['active'], 'old_rows', None)
    c = merger.coefficients
    rows = sum(base[n].shape[0] if base[n].ndim else 1 for n in plan['active'])
    boundaries = (0, 2 * len(plan['active']), 2 * len(plan['active']) + rows, c.numel())
    assert boundaries == (0, 766, 174216, 347666)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1,
        plan['tasks'].index(t), 217]).permutation(len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, schedule, pair_schedule, rounds = [], [], [], []
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        order = order_rng.permutation(len(tasks)).tolist()
        batches = {t: permutations[t][rnd * bs:(rnd + 1) * bs] for t in tasks}
        records = []
        for position, k in enumerate(order):
            query_task = tasks[k]
            support_task = tasks[order[(position + 1) % len(tasks)]]
            assert query_task != support_task
            qids, sids = batches[query_task], batches[support_task]
            assert len(qids) == len(sids) == bs
            xb, yb = data.train[support_task]
            support_batch = xb[sids].to(device), yb[sids].to(device)
            xb, yb = data.train[query_task]
            query_batch = xb[qids].to(device), yb[qids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(merger, wrappers[support_task], support_batch,
                                 data.weights[support_task], c)
            def query_loss(coef):
                return weighted_loss(merger, wrappers[query_task], query_batch,
                                     data.weights[query_task], coef)
            gradient, diagnostic = objective_gradient(c, loss, query_loss, condition,
                boundaries, plan['virtual_step_rate'], plan['virtual_step_epsilon'])
            c.grad = gradient
            optimizer.step()
            records.append(diagnostic)
            schedule.append([plan['tasks'].index(query_task), *qids.tolist()])
            pair_schedule.append([plan['tasks'].index(support_task), *sids.tolist(),
                                  plan['tasks'].index(query_task), *qids.tolist()])
        rounds.append({key: np.asarray([x[key] for x in records]).mean(0).tolist()
                       for key in records[0]})
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    probes = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        for task in tasks:
            probes[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    del parameters, loss, query_loss, optimizer, c, merger, wrappers, gradient
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            native = PrototypeClassifier(leaf, *data.readouts[task]).eval()(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(probes[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5 and checks[task]['changed_predictions'] == 0
    updates = len(schedule)
    record = {'updates': updates, 'training_examples': updates * bs * 2,
        'training_forward_batches': 2 * updates,
        'support_gradient_calls': updates, 'outer_gradient_calls': updates,
        'outer_calls_with_second_order': updates if condition == 'cross_task_meta' else 0,
        'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
        'pair_schedule_hash': array_hash(np.asarray(pair_schedule, dtype=np.int64)),
        'round_diagnostics': rounds, 'export_checks': checks,
        'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - started,
        'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    return state, {'snapshots': torch.stack(snapshots), 'mean': mean}, record
