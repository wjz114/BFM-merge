"""Exact cached initial output-bias gradients for every paired training batch."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256
from distillation_control_support import readout


def summarize(batch, task_component):
    mean = batch.mean(0)
    centered = batch-mean
    task_centered = task_component-task_component.mean(0)
    residual = centered-task_centered
    total = centered.square().sum(-1).mean()
    between = task_centered.square().sum(-1).mean()
    within = residual.square().sum(-1).mean()
    cross = 2*(task_centered*residual).sum(-1).mean()
    assert torch.allclose(total, between+within+cross, rtol=1e-12, atol=1e-22)
    m, first = torch.zeros_like(mean), []
    for i, g in enumerate(batch):
        m = .9*m+.1*g
        first.append(m/(1-.9**(i+1)))
    first = torch.stack(first)
    return {'total_batch_variance': float(total), 'between_task_means_component': float(between),
            'within_task_sampling_component': float(within), 'twice_cross_covariance': float(cross),
            'global_mean_gradient_norm': float(mean.norm()),
            'variance_over_mean_norm_sq': float(total/mean.square().sum()),
            'first_moment_deviation_mean_sq': float((first-mean).square().sum(-1).mean()),
            'first_moment_deviation_last800_mean_sq': float((first[800:]-mean).square().sum(-1).mean())}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    cp = root/'seed0/full_model_distillation_targets/run.json'
    dp = root/'inputs/task_mixing_design.json'
    gp = root/'seed0/local_gradient_cache/run.json'
    cache, design, old_gradients = [json.loads(p.read_text()) for p in [cp, dp, gp]]
    plan = design['plan_template']
    assert cache['plan']['initial_checkpoint'] == plan['initial_checkpoint']
    assert cache['plan']['schedule'] == plan['schedule']
    assert sha256(Path(plan['schedule'])) == plan['schedule_sha256']
    source_path = Path(plan['source_fit'])/'run.json'
    source = json.loads(source_path.read_text())
    tasks = cache['datasets']
    assert tasks == old_gradients['datasets'] == source['datasets']
    schedules = {m: np.load(design['schedules'][m]['micro_schedule']) for m in design['arms']}
    for mode, s in schedules.items():
        assert sha256(Path(design['schedules'][mode]['micro_schedule'])) == design['schedules'][mode]['micro_schedule_sha256']
    original = np.load(plan['schedule'])
    calibration = np.load(plan['calibration'])
    hashes = {str(p): sha256(p) for p in [cp, dp, gp, source_path, Path(plan['calibration']), Path('distillation_control_support.py')]}
    micro = {m: torch.empty((1600, 4, 768), dtype=torch.float64) for m in design['arms']}
    task_means, replay_checks, task_statistics = [], {}, {}
    for ti, task in enumerate(tasks):
        record = cache['records'][task]
        path = Path(record['target_path'])
        assert sha256(path) == record['target_sha256']
        hashes[str(path)] = record['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        z = data['initial_raw'].double().requires_grad_(True)
        logits = readout(z, data['prototypes'].double())
        gradient, = torch.autograd.grad(F.cross_entropy(logits, data['truth'], reduction='sum'), z)
        gradient = gradient.detach()
        old_ids = torch.from_numpy(calibration[task].astype(np.int64))
        old_pos = torch.searchsorted(data['pool_indices'], old_ids)
        assert torch.equal(data['pool_indices'][old_pos], old_ids)
        old_weights = torch.tensor(source['ce_class_weights'][task], dtype=torch.float64)
        replay = (gradient[old_pos]*old_weights[data['truth'][old_pos], None]).mean(0)
        old_path = Path(old_gradients['records'][task]['path'])
        assert sha256(old_path) == old_gradients['records'][task]['sha256']
        stored = torch.load(old_path, map_location='cpu', weights_only=True, mmap=True)['qformer.output_embed.2.bias'].double()
        error = replay-stored
        relative = float(error.norm()/stored.norm())
        assert relative < 1e-4, (task, relative)
        replay_checks[task] = {'relative_l2_error': relative, 'max_abs_error': float(error.abs().max())}
        weights = torch.tensor(record['class_weights'], dtype=torch.float64)
        gradient = gradient*weights[data['truth'], None]
        ids = torch.from_numpy(original['local_indices'][original['task'] == ti].reshape(-1))
        pos = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][pos], ids)
        g = gradient[pos]
        task_means.append(g.mean(0))
        task_statistics[task] = {'fitting_exposures': len(ids), 'bias_gradient_norm': float(g.mean(0).norm()),
                                  'per_example_variance': float((g-g.mean(0)).square().sum(-1).mean())}
        for mode, schedule in schedules.items():
            mask = schedule['task'] == ti
            ids = torch.from_numpy(schedule['local_indices'][mask].reshape(-1))
            pos = torch.searchsorted(data['pool_indices'], ids)
            assert torch.equal(data['pool_indices'][pos], ids)
            micro[mode][torch.from_numpy(mask)] = gradient[pos].reshape(-1, 32, 768).mean(1)
    task_means = torch.stack(task_means)
    results, overall_means = {}, []
    for mode, schedule in schedules.items():
        batches = micro[mode].mean(1)
        means = task_means[torch.from_numpy(schedule['task'].astype(np.int64))].mean(1)
        results[mode] = summarize(batches, means)
        overall_means.append(batches.mean(0))
    assert torch.allclose(*overall_means, rtol=1e-12, atol=1e-18)
    ratio_keys = ['total_batch_variance', 'between_task_means_component', 'within_task_sampling_component',
                  'first_moment_deviation_mean_sq', 'first_moment_deviation_last800_mean_sq']
    ratios = {k: results['mixed_task_micro4'][k]/results['same_task_micro4'][k] for k in ratio_keys}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'parameter': 'qformer.output_embed.2.bias', 'parameter_count': 768,
              'old512_native_gradient_replay': replay_checks,
              'per_task': task_statistics, 'batch_organization': results, 'mixed_over_same_ratios': ratios,
              'seconds': time.time()-started,
              'model_inference_performed': False, 'full_network_backward_performed': False,
              'cached_readout_autograd': True, 'optimizer_updates': 0,
              'validation_used_in_statistics': False, 'test_data_read': False,
              'scope': 'FP64 cosine-readout derivatives on cached FP32 initial forward outputs. Their sum is the shared final linear output-bias gradient. Every actual fitting-stream row and full-pool class weight is included. Old512 bias gradients replay against independent native full-network caches within1e-4 relative error. This is one768-parameter block at a fixed initial model; it does not describe all383 parameter tensors, moving-model gradients, actual Adam steps or generalization. The variance decomposition includes the observed task/sampling cross term; no independence assumption. First moments are a fixed-state surrogate with beta1=.9.'}
    path = root/'analysis/task_mixing_initial_bias_gradient_variance.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'ratios': ratios, 'native_replay_max_relative_error': max(v['relative_l2_error'] for v in replay_checks.values()),
                      'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
