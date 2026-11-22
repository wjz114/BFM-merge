"""Describe same-task seed variation; no EEG data, fitting, or performance evaluation."""
import hashlib
import json
import time
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    root = Path('runs/replica_source_audit_v1')
    plan = json.loads((root / 'plan.json').read_text())
    assert not (root / 'comparison.json').exists()
    for path, expected in plan['sha256'].items():
        assert digest(path) == expected, path
    started = time.monotonic()
    base = torch.load(plan['base'], map_location='cpu', weights_only=True)
    active, tasks, seeds = plan['active'], plan['tasks'], plan['seeds']
    assert len(seeds) == 3
    across_mean, across_m2 = {}, {}
    layers = {name: {'within_variance_trace_sum': 0., 'mean_squared_norm_sum': 0.,
                     'nonradial_within_variance_trace_sum': 0.} for name in active}
    records = {}
    for count, task in enumerate(tasks, 1):
        models = [torch.load(plan['experts'][str(seed)][task], map_location='cpu', weights_only=True)
                  for seed in seeds]
        gram = torch.zeros((3, 3), dtype=torch.float64)
        within, nonradial, signal, identity_error = 0., 0., 0., 0.
        for name in active:
            x = torch.stack([(model[name].double() - base[name].double()).reshape(-1) for model in models])
            assert torch.isfinite(x).all()
            mean = x.mean(0)
            centered = x - mean
            variance = centered.square().sum() / 2
            energy = mean.square().sum()
            gram += x @ x.T
            rows = base[name].shape[0] if base[name].ndim else 1
            row_mean = mean.reshape(rows, -1)
            residual = centered.reshape(3, rows, -1)
            norms = row_mean.square().sum(1)
            radial = torch.where(norms[None] > 0,
                (residual * row_mean[None]).sum(2).square() / norms[None].clamp_min(1e-300), 0.).sum() / 2
            orthogonal = (variance - radial).clamp_min(0)
            err = float((x.square().sum() / 3 - energy - variance * (2 / 3)).abs())
            assert err <= 1e-11 * max(float(x.square().sum()), 1e-30)
            identity_error = max(identity_error, err)
            within += float(variance)
            nonradial += float(orthogonal)
            signal += float(energy)
            layers[name]['within_variance_trace_sum'] += float(variance)
            layers[name]['mean_squared_norm_sum'] += float(energy)
            layers[name]['nonradial_within_variance_trace_sum'] += float(orthogonal)
            if count == 1:
                across_mean[name] = mean.clone()
                across_m2[name] = torch.zeros_like(mean)
            else:
                change = mean - across_mean[name]
                across_mean[name].add_(change / count)
                across_m2[name].add_(change * (mean - across_mean[name]))
        norms = gram.diag().sqrt()
        cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-300)
        records[task] = {'replica_gram': gram.tolist(), 'replica_cosine': cosine.tolist(),
            'mean_squared_norm': signal, 'within_variance_trace': within,
            'within_variance_over_mean_energy': within / max(signal, 1e-300),
            'nonradial_within_variance_fraction': nonradial / max(within, 1e-300),
            'energy_identity_max_error': identity_error}
        print(task, records[task]['within_variance_over_mean_energy'], flush=True)
        del models
    for name in active:
        item = layers[name]
        item['within_variance_trace_mean'] = item.pop('within_variance_trace_sum') / len(tasks)
        item['mean_squared_norm_mean'] = item.pop('mean_squared_norm_sum') / len(tasks)
        item['nonradial_within_variance_trace_mean'] = item.pop('nonradial_within_variance_trace_sum') / len(tasks)
        item['between_task_mean_variance_trace'] = float(across_m2[name].sum()) / (len(tasks) - 1)
    within = sum(v['within_variance_trace_mean'] for v in layers.values())
    between = sum(v['between_task_mean_variance_trace'] for v in layers.values())
    nonradial = sum(v['nonradial_within_variance_trace_mean'] for v in layers.values())
    result = {'plan_sha256': digest(root / 'plan.json'), 'tasks': records, 'layers': layers,
        'summary': {'within_variance_trace_mean': within,
            'between_task_mean_variance_trace': between,
            'within_mean_estimator_variance_over_between_task_variance': within / (3 * between),
            'nonradial_within_variance_fraction': nonradial / within},
        'elapsed_seconds': time.monotonic() - started,
        'new_optimizer_updates': 0, 'EEG_examples_read': 0, 'official_test_read': False,
        'scope': 'Descriptive geometry of three seeds trained on the same full task data. '
                 'Within variance measures training and checkpoint-selection variation, not established harmful noise. '
                 'The within/3 estimator interpretation assumes independent centered seed errors and is not a measured posterior. '
                 'Nonradial means outside each task mean row direction; it does not prove functional harm. '
                 'All tasks are inspected offline for design; a future streaming operator must only use arrived experts. '
                 'No new merging operator, generalization improvement, or novelty is established.'}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
