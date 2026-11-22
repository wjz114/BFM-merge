"""Check fold routing gradients and native exports, using existing source replicas.

Replicas here are engineering stand-ins, not held-out-fold experts. No validation
performance is measured and no checkpoint from this check becomes a pilot source.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData, strict_settings
from streaming_bilateral_merge import StreamingMerger
from streaming_crossfit_training import CONDITIONS, batch_groups, source_mean, train_stage


def check_gradient(leaf, base, old, new, data, task, assignments, plan, device):
    mergers = [StreamingMerger(leaf, base, old[k], new[k], None, 1,
                              plan['active'], 'old_rows', None) for k in (0, 1)]
    c = mergers[0].coefficients
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    ids = np.asarray([np.flatnonzero(assignments[task] == 0)[0],
                      *np.flatnonzero(assignments[task] == 1)[:3]], dtype=np.int64)
    x, y = data.train[task]
    labels = y[ids].to(device)
    checks = {}
    for condition in ('out_of_fold', 'in_fold'):
        world = assignments[task][ids]
        if condition == 'out_of_fold':
            world = 1 - world
        # Independent dense routing: both whole-batch functions, then row masking.
        outputs = [m.logits(wrapper, x[ids].to(device), m.compose(c)) for m in mergers]
        logits = torch.stack(outputs)[torch.tensor(world, device=device, dtype=torch.long), torch.arange(4, device=device)]
        dense_loss = (F.cross_entropy(logits, labels, reduction='none') * data.weights[task][labels]).mean()
        dense_gradient = torch.autograd.grad(dense_loss, c)[0]
        sparse_gradient = torch.zeros_like(c)
        sparse_value, grouped = 0., []
        for k, group in batch_groups(ids, assignments[task], condition):
            target = y[group].to(device)
            output = mergers[k].logits(wrapper, x[group].to(device), mergers[k].compose(c))
            loss = (F.cross_entropy(output, target, reduction='none') * data.weights[task][target]).sum() / 4
            sparse_gradient += torch.autograd.grad(loss, c)[0]
            sparse_value += float(loss.detach())
            grouped.extend((int(i), k) for i in group)
        assert sorted(grouped) == sorted(zip(ids.tolist(), world.tolist()))
        error = float((dense_gradient - sparse_gradient).abs().max())
        loss_error = abs(float(dense_loss.detach()) - sparse_value)
        assert error < 1e-4 and loss_error < 1e-5, (condition, error, loss_error)
        checks[condition] = {'gradient_max_abs_error': error, 'loss_abs_error': loss_error,
                             'unequal_group_sizes': [1, 3], 'CE_examples': 12}
    # Empty-fold routing must retain every sample with the original denominator.
    for condition in ('out_of_fold', 'in_fold'):
        one_ids = np.flatnonzero(assignments[task] == 0)[:4]
        groups = batch_groups(one_ids, assignments[task], condition)
        assert len(groups) == 1 and np.array_equal(groups[0][1], one_ids)
        assert groups[0][0] == (1 if condition == 'out_of_fold' else 0)
    checks['single_nonempty_group'] = True
    return checks


def main():
    root = Path('runs/crossfit_merge_candidate_v1')
    path = root / 'merge_engineering_plan_v2.json'
    plan = json.loads(path.read_text())
    assert not (root / 'merge_engineering.json').exists()
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    strict_settings()
    device = configure(plan['gpu'], plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    data = TaskData(plan, source, device)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    started = time.monotonic()
    assignments, states = {}, {}
    # Four already available task experts; only training calibration EEG is read.
    for task in plan['pilot_tasks']:
        data.arrive(task)
        with np.load(plan['folds']) as folds:
            assignments[task] = folds[f'{task}__calibration_fold'].copy()
        paths = plan['engineering_replica_sources'][task]
        states[task] = []
        for p in paths:
            assert sha256(Path(p)) == plan['expert_sha256'][p]
            states[task].append(torch.load(p, map_location='cpu', weights_only=True))
    a, b = plan['pilot_tasks'][:2]
    gradients = check_gradient(leaf, base, states[a], states[b], data, b, assignments, plan, device)
    all_records, schedules, max_error, updates = {}, {}, 0., 0
    for condition in CONDITIONS:
        previous, proxies, records = None, [], []
        for count, task in enumerate(plan['pilot_tasks']):
            incoming = (source_mean([states[task][0], *states[task]], plan['active'])
                        if condition == 'source_soup' else states[task][0])
            incoming_proxies = states[task] if condition in ('out_of_fold', 'in_fold') else []
            if count == 0:
                previous, proxies = incoming, incoming_proxies
                continue
            native, new_proxies, payload, record = train_stage(leaf, base, previous, incoming,
                proxies, incoming_proxies, count, data, plan['pilot_tasks'][:count + 1],
                assignments, plan, condition, device)
            assert torch.equal(payload['mean'], payload['snapshots'].double().mean(0).float())
            errors = [audit_export(base, previous, incoming, native, payload['mean'], plan['active'])]
            for k, new in enumerate(new_proxies):
                errors.append(audit_export(base, proxies[k], incoming_proxies[k], new,
                                           payload['mean'], plan['active']))
            max_error = max(max_error, *errors)
            record['independent_export_max_abs_errors'] = errors
            records.append(record)
            updates += record['updates']
            previous, proxies = native, new_proxies
            del payload
            gc.collect()
            torch.cuda.empty_cache()
        all_records[condition] = records
        schedules[condition] = [r['schedule_hash'] for r in records]
    assert all(v == schedules['single_full'] for v in schedules.values())
    assert updates == 72
    result = {'plan_sha256': sha256(path), 'status': 'passed', 'gradient_checks': gradients,
        'conditions': all_records, 'updates': updates, 'training_CE_examples': updates * plan['batch_size'],
        'gradient_probe_CE_examples': 24, 'max_independent_export_error': max_error,
        'matched_schedules': True, 'source_models_are_engineering_standins': True,
        'validation_EEG_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    with (root / 'merge_engineering.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'conditions'}), flush=True)


if __name__ == '__main__':
    main()
