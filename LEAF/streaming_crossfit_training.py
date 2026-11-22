"""Learn one static bilateral operator through two source-training fold worlds."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger


CONDITIONS = ('out_of_fold', 'in_fold', 'source_soup', 'single_full')


def source_mean(states, active):
    """Uniform mean of these exact sources; frozen entries must agree."""
    fixed = set(states[0]) - set(active)
    assert all(s.keys() == states[0].keys() for s in states)
    assert all(torch.equal(s[n], states[0][n]) for s in states for n in fixed)
    return {n: (torch.stack([s[n].double() for s in states]).mean(0).float()
                if n in active else v.clone()) for n, v in states[0].items()}


def batch_groups(ids, assignment, condition):
    if condition in ('single_full', 'source_soup'):
        return [(0, ids)]
    assert condition in ('out_of_fold', 'in_fold')
    selected = assignment[ids]
    assert np.isin(selected, [0, 1]).all()
    worlds = 1 - selected if condition == 'out_of_fold' else selected
    return [(world, ids[worlds == world]) for world in (0, 1)
            if np.any(worlds == world)]


def train_stage(leaf, base, previous, incoming, previous_proxies, incoming_proxies,
                count, data, tasks, assignments, plan, condition, device):
    assert condition in CONDITIONS
    proxy_mode = condition in ('out_of_fold', 'in_fold')
    pairs = (list(zip(previous_proxies, incoming_proxies)) if proxy_mode
             else [(previous, incoming)])
    assert len(pairs) == (2 if proxy_mode else 1)
    mergers = [StreamingMerger(leaf, base, old, new, None, count,
                              plan['active'], 'old_rows', None) for old, new in pairs]
    c = mergers[0].coefficients
    for m in mergers[1:]:
        assert torch.equal(m.coefficients, c)
        m.coefficients.requires_grad_(False)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1,
        plan['tasks'].index(t), 217]).permutation(len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    schedule, routing, snapshots, rounds = [], [], [], []
    forward_calls = 0
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        losses = []
        for k in order_rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            groups = batch_groups(ids, assignments.get(task), condition)
            assert sorted(i for _, group in groups for i in group) == sorted(ids)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            x, y = data.train[task]
            for world, group in groups:
                logits = mergers[world].logits(wrappers[task], x[group].to(device),
                                               mergers[world].compose(c))
                labels = y[group].to(device)
                loss = (F.cross_entropy(logits, labels, reduction='none')
                        * data.weights[task][labels]).sum() / bs
                loss.backward()
                loss_value += float(loss.detach())
                forward_calls += 1
                routing.extend([[plan['tasks'].index(task), int(i), world] for i in group])
            optimizer.step()
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            losses.append(loss_value)
        rounds.append(float(np.mean(losses)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    proxy_states = [m.materialize(mean.to(device)) for m in mergers] if proxy_mode else []
    if proxy_mode:
        deployment = StreamingMerger(leaf, base, previous, incoming, None, count,
                                     plan['active'], 'old_rows', None)
    else:
        deployment = mergers[0]
    state = deployment.materialize(mean.to(device))
    all_mergers, all_states = [deployment], [state]
    if proxy_mode:
        all_mergers += mergers
        all_states += proxy_states
    checks = {}
    with torch.no_grad():
        for index, (m, native_state) in enumerate(zip(all_mergers, all_states)):
            parameters = m.compose(mean.to(device))
            functional = {t: m.logits(wrappers[t], data.train[t][0][:bs].to(device),
                                     parameters).cpu() for t in tasks}
            leaf.load_state_dict(native_state, strict=True)
            checks[str(index)] = {}
            for t in tasks:
                native = wrappers[t](data.train[t][0][:bs].to(device)).cpu()
                check = compare(functional[t], native, data.train[t][1][:bs])
                assert check['max_abs_logit_error'] <= 1e-5 and check['changed_predictions'] == 0
                checks[str(index)][t] = check
    leaf.load_state_dict(state, strict=True)
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
        'training_forward_batches': forward_calls, 'backward_calls': forward_calls,
        'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
        'routing_hash': array_hash(np.asarray(routing, dtype=np.int64)),
        'round_losses': rounds, 'export_checks': checks,
        'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - started,
        'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    return state, proxy_states, {'snapshots': torch.stack(snapshots), 'mean': mean}, record
