"""Independent derivative checks and four actual arrivals; no validation selection."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256, PrototypeClassifier
from report_streaming_replica_filter import audit_export
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger
from streaming_cross_task_meta import CONDITIONS, configure_attention, normalized_step, objective_gradient
from streaming_cross_task_meta_training import train_stage, weighted_loss


def algebra_check():
    rng = torch.Generator().manual_seed(261117)
    x = torch.randn(14, generator=rng, dtype=torch.float64)
    a = torch.randn(19, 14, generator=rng, dtype=torch.float64) / 4
    b = torch.randn(17, 14, generator=rng, dtype=torch.float64) / 4
    ya = torch.randn(19, generator=rng, dtype=torch.float64)
    yb = torch.randn(17, generator=rng, dtype=torch.float64)
    support = lambda z: ((a @ z).tanh() - ya).square().mean()
    query = lambda z: ((b @ z).tanh() - yb).square().mean()
    boundaries, rate, eps = (0, 4, 9, 14), .01, 1e-8
    c = x.clone().requires_grad_(True)
    g = torch.autograd.grad(support(c), c)[0].detach()
    h = torch.autograd.functional.hessian(support, c).detach()
    step = normalized_step(g, boundaries, rate, eps)
    v = (x - step).requires_grad_(True)
    q = torch.autograd.grad(query(v), v)[0].detach()
    probe = torch.empty_like(g)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        u, w = g[lo:hi], q[lo:hi]
        n = hi - lo
        scale = (u.square().mean() + eps ** 2).sqrt()
        probe[lo:hi] = rate * (w / scale - u * (u @ w) / (n * scale ** 3))
    expected = .5 * (g + q - h @ probe)
    c = x.clone().requires_grad_(True)
    actual, _ = objective_gradient(c, support(c), query, 'cross_task_meta', boundaries, rate, eps)
    error = float((actual - expected).abs().max())
    assert error < 1e-12
    def objective(z):
        z = z.detach().clone().requires_grad_(True)
        l = support(z)
        grad = torch.autograd.grad(l, z)[0]
        return float((.5 * (l + query(z - normalized_step(grad, boundaries, rate, eps)))).detach())
    finite = torch.empty_like(x)
    for j in range(x.numel()):
        direction = torch.zeros_like(x)
        direction[j] = 1e-5
        finite[j] = (objective(x + direction) - objective(x - direction)) / 2e-5
    fd_error = float((actual - finite).abs().max())
    assert fd_error < 1e-8
    c = x.clone().requires_grad_(True)
    detached, _ = objective_gradient(c, support(c), query, 'detached_lookahead', boundaries, rate, eps)
    assert torch.allclose(detached, .5 * (g + q), atol=1e-12, rtol=0)
    c = x.clone().requires_grad_(True)
    paired, _ = objective_gradient(c, support(c), query, 'paired_ce', boundaries, rate, eps)
    z = x.clone().requires_grad_(True)
    ce = torch.autograd.grad(.5 * (support(z) + query(z)), z)[0]
    assert torch.allclose(paired, ce, atol=1e-12, rtol=0)
    assert (actual - detached).norm() > 1e-6
    return {'independent_chain_rule_max_error': error, 'finite_difference_max_error': fd_error,
        'detached_gradient_matches_explicit_formula': True, 'paired_ce_matches_direct_gradient': True,
        'full_minus_detached_norm': float((actual - detached).norm())}


def real_derivative_check(leaf, base, data, source, plan, device):
    tasks = ['MI_OpenBMI', 'EMO_FACED']
    for task in tasks:
        data.arrive(task)
    previous, incoming = [torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks]
    merger = StreamingMerger(leaf, base, previous, incoming, None, 1, plan['active'], 'old_rows', None)
    wrappers = [PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks]
    batches = [(data.train[t][0][:4].to(device), data.train[t][1][:4].to(device)) for t in tasks]
    support = lambda c: weighted_loss(merger, wrappers[0], batches[0], data.weights[tasks[0]], c)
    query = lambda c: weighted_loss(merger, wrappers[1], batches[1], data.weights[tasks[1]], c)
    x = merger.coefficients.detach().clone()
    boundaries = (0, 766, 174216, 347666)
    rate, eps = plan['virtual_step_rate'], plan['virtual_step_epsilon']
    c = x.clone().requires_grad_(True)
    full, _ = objective_gradient(c, support(c), query, 'cross_task_meta', boundaries, rate, eps)
    direction = full / full.norm()
    delta = .005
    values = []
    for sign in [-1, 1]:
        c = (x + sign * delta * direction).detach().requires_grad_(True)
        loss = support(c)
        grad = torch.autograd.grad(loss, c)[0]
        value = .5 * (loss.detach() + query(c - normalized_step(grad, boundaries, rate, eps)))
        values.append(float(value.detach()))
    finite = (values[1] - values[0]) / (2 * delta)
    dot = float(full @ direction)
    assert abs(finite - dot) <= 1e-4 + .02 * abs(dot), (finite, dot)
    c = x.clone().requires_grad_(True)
    zero, _ = objective_gradient(c, support(c), query, 'cross_task_meta', boundaries, 0., eps)
    c = x.clone().requires_grad_(True)
    paired, _ = objective_gradient(c, support(c), query, 'paired_ce', boundaries, rate, eps)
    zero_error = float((zero - paired).abs().max())
    assert zero_error < 1e-6
    c = x.clone().requires_grad_(True)
    detached, _ = objective_gradient(c, support(c), query, 'detached_lookahead', boundaries, rate, eps)
    assert torch.isfinite(full).all() and (full - detached).norm() > 0
    return {'native_full_meta_gradient_finite': True, 'directional_derivative': dot,
        'native_central_difference': finite, 'native_finite_difference_absolute_error': abs(finite - dot),
        'finite_difference_delta': delta, 'finite_difference_tolerance': '1e-4 + 2% of analytic derivative',
        'zero_virtual_step_matches_paired_ce_max_error': zero_error,
        'full_minus_detached_norm': float((full - detached).norm()),
        'read_training_rows_only': True}


def main():
    root = Path('runs/streaming_cross_task_meta_v1')
    assert not (root / 'engineering.json').exists()
    spec = json.loads((root / 'engineering_launch_spec.json').read_text())
    for path, digest in spec['sha256'].items():
        assert sha256(Path(path)) == digest, path
    plan = json.loads((root / 'engineering_plan.json').read_text())
    storage = Path(plan['checkpoint_storage_root']) / 'engineering'
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    configure_attention()
    torch.cuda.set_per_process_memory_fraction(.20, device)
    algebra = algebra_check()
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    data = TaskData(plan, source, device)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    derivative = real_derivative_check(leaf, base, data, source, plan, device)
    del leaf, data
    gc.collect()
    torch.cuda.empty_cache()
    records, errors = {}, []
    for condition in CONDITIONS:
        data = TaskData(plan, source, device)
        leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        previous, records[condition] = None, []
        for count, task in enumerate(tasks):
            data.arrive(task)
            assert set(data.train) == set(tasks[:count + 1])
            path = Path(source['experts'][task])
            assert sha256(path) == plan['expert_sha256'][task]
            incoming = torch.load(path, map_location='cpu', weights_only=True)
            if count == 0:
                previous = incoming
                leaf.load_state_dict(previous, strict=True)
                continue
            state, payload, record = train_stage(leaf, base, previous, incoming, count,
                data, tasks[:count + 1], plan, condition, device)
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            error = audit_export(base, previous, incoming, state, payload['mean'], plan['active'])
            errors.append(error)
            record['independent_export_max_abs_error'] = error
            for suffix, value in [('merged.ckpt', state), ('coefficients.pt', payload)]:
                save_tensor_file(value, root / f'engineering_{condition}_stage{count+1}_{suffix}', root, storage)
            records[condition].append(record)
            previous = state
            del incoming, state, payload
            gc.collect()
            torch.cuda.empty_cache()
        assert not data.valid
        del data, leaf, previous
        gc.collect()
        torch.cuda.empty_cache()
    reference = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert all([r['schedule_hash'] for r in rows] == [r['schedule_hash'] for r in reference] for rows in records.values())
    for condition in CONDITIONS[1:]:
        assert [r['pair_schedule_hash'] for r in records[condition]] == [r['pair_schedule_hash'] for r in records[CONDITIONS[0]]]
    updates = sum(r['updates'] for rows in records.values() for r in rows)
    assert updates == 54
    write_json(root / 'engineering.json', {'passed': True, 'algebra': algebra,
        'native_derivative': derivative, 'independent_export_max_abs_error': max(errors),
        'coefficient_optimizer_updates': updates, 'training_CE_examples': updates * 8,
        'derivative_probe_examples_additional_to_training': True,
        'native_exports_pass': True, 'fixed_coefficient_means_exact': True,
        'paired_schedules_exact': True, 'primary_schedule_matches_frozen_engineering': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
    print(json.dumps({'passed': True, 'derivative': derivative, 'updates': updates,
        'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
