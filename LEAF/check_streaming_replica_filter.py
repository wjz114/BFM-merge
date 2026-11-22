"""Independent dense/spectral checks and a four-arrival native training check."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json, train_stage
from run_streaming_replica_filter import load_replicas
from streaming_replica_filter import CONDITIONS, prepare_incoming, filter_rows, shift_rows


def independent_incoming(base, previous, replicas, active, condition, task_index):
    state, errors = base.copy(), []
    for tensor_index, name in enumerate(active):
        shape = base[name].shape
        rows = shape[0] if len(shape) else 1
        values = torch.stack([m[name].double().reshape(rows, -1) for m in replicas], 1)
        center = values.sum(1) / 3
        mean = center.float().double()
        if previous is None or condition == 'source_soup':
            state[name] = mean.reshape(shape).float()
            continue
        residuals = values - center[:, None]
        width = residuals.shape[-1]
        if condition == 'shifted_filter' and width > 1:
            # Use explicit per-row rolls, independently of the production gather.
            residuals = torch.stack([torch.roll(x, 1 + (r + 104729 * tensor_index + 8191 * task_index) % (width - 1), -1)
                                     for r, x in enumerate(residuals)])
        energy = residuals.square().sum((1, 2))
        norm = torch.where(energy > 0, (energy / 2).sqrt(), torch.ones_like(energy))
        z = residuals / norm[:, None, None]
        gram = torch.einsum('rid,rjd->rij', z, z)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        innovation = mean - previous[name].double().reshape(rows, -1)
        rhs = torch.einsum('rid,rd->ri', z, innovation)
        coefficients = torch.einsum('rji,rj->ri', eigenvectors, rhs)
        coefficients = coefficients / (1 + eigenvalues.clamp_min(0))
        solution = torch.einsum('rij,rj->ri', eigenvectors, coefficients)
        correction = torch.einsum('rid,ri->rd', z, solution)
        state[name] = (mean - correction).reshape(shape).float()
        errors.append(float((gram - gram.transpose(1, 2)).abs().max()))
    return state


def algebra_check():
    rng = torch.Generator().manual_seed(172611)
    residuals = torch.randn(8, 3, 7, generator=rng, dtype=torch.float64)
    residuals -= residuals.mean(1, keepdim=True)
    residuals[0] = 0
    residuals[1, 2] = 0
    residuals[1, 1] = -residuals[1, 0]
    v = torch.randn(8, 7, generator=rng, dtype=torch.float64)
    tau = residuals.square().sum((1, 2)) / 2
    c = residuals.transpose(1, 2) @ residuals
    dense = torch.linalg.solve(torch.eye(7, dtype=torch.float64) + c / torch.where(tau > 0, tau, 1.)[:, None, None], v.unsqueeze(-1)).squeeze(-1)
    actual = filter_rows(v, residuals)
    error = float((actual - dense).abs().max())
    assert error < 1e-12 and torch.equal(actual[0], v[0])
    assert (actual.norm(dim=1) <= v.norm(dim=1) + 1e-12).all()
    assert (actual.norm(dim=1) >= v.norm(dim=1) / 3 - 1e-12).all()
    shifted = shift_rows(residuals, 4, 7)
    gram_error = float((residuals @ residuals.transpose(1, 2) - shifted @ shifted.transpose(1, 2)).abs().max())
    assert gram_error < 1e-12
    scale_error = float((filter_rows(v, 17 * residuals) - actual).abs().max())
    assert scale_error < 1e-12
    one = torch.tensor([[[1.], [-1.], [0.]]], dtype=torch.float64)
    assert torch.allclose(filter_rows(torch.ones(1, 1, dtype=torch.float64), one), torch.full((1, 1), 1 / 3, dtype=torch.float64))
    # The directional filter can leave the old/mean row plane.
    old = torch.tensor([1., 0., 0.], dtype=torch.float64)
    mean = torch.tensor([0., 1., 0.], dtype=torch.float64)
    e = torch.tensor([[[0., 1., 1.], [0., -1., -1.], [0., 0., 0.]]], dtype=torch.float64)
    transformed = old + filter_rows((mean - old)[None], e)[0]
    assert abs(float(transformed[2])) > .1
    return {'dense_inverse_max_error': error, 'shifted_Gram_max_error': gram_error,
            'scatter_scale_invariance_max_error': scale_error,
            'zero_and_rank1_pass': True, 'operator_spectral_bounds_pass': True,
            'outside_original_row_plane_example': transformed.tolist()}


def main():
    root = Path('runs/streaming_replica_filter_v1')
    assert not (root / 'engineering.json').exists()
    spec = json.loads((root / 'engineering_launch_spec.json').read_text())
    for path, expected in spec['sha256'].items():
        assert sha256(Path(path)) == expected, path
    plan = json.loads((root / 'engineering_plan.json').read_text())
    storage = Path(plan['checkpoint_storage_root']) / 'engineering'
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    algebra = algebra_check()
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    records, incoming_errors, first_states = {}, [], []
    for condition in CONDITIONS:
        data = TaskData(plan, source, device)
        leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        previous, records[condition] = None, []
        for count, task in enumerate(tasks):
            data.arrive(task)
            assert set(data.train) == set(tasks[:count + 1])
            replicas = load_replicas(plan, task)
            mean, incoming, geometry = prepare_incoming(base, previous, replicas, plan['active'], condition, plan['tasks'].index(task))
            oracle = independent_incoming(base, previous, replicas, plan['active'], condition, plan['tasks'].index(task))
            error = max(float((incoming[n] - oracle[n]).abs().max()) for n in plan['active'])
            assert error < 2e-7
            incoming_errors.append(error)
            if count == 0:
                assert all(torch.equal(incoming[n], mean[n]) for n in incoming)
                first_states.append({n: v.clone() for n, v in incoming.items()})
                previous = incoming
                leaf.load_state_dict(previous, strict=True)
                continue
            state, payload, record = train_stage(leaf, base, previous, incoming, None,
                count, data, tasks[:count + 1], plan, 'old_rows', device, None)
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            assert record['coefficient_count'] == 347666 and payload['mean'].isfinite().all()
            record['independent_incoming_max_abs_error'] = error
            record['filter_geometry'] = geometry
            for suffix, value in [('merged.ckpt', state), ('incoming.ckpt', incoming), ('coefficients.pt', payload)]:
                save_tensor_file(value, root / f'engineering_{condition}_stage{count+1}_{suffix}', root, storage)
            records[condition].append(record)
            previous = state
            del replicas, mean, incoming, oracle, state, payload
            gc.collect()
            torch.cuda.empty_cache()
        assert not data.valid
        del data, leaf, previous
        gc.collect()
        torch.cuda.empty_cache()
    assert all(all(torch.equal(s[n], first_states[0][n]) for n in s) for s in first_states)
    frozen = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert all([r['schedule_hash'] for r in rows] == [r['schedule_hash'] for r in frozen] for rows in records.values())
    updates = sum(r['updates'] for rows in records.values() for r in rows)
    assert updates == 54
    write_json(root / 'engineering.json', {'passed': True, 'algebra': algebra,
        'independent_incoming_max_abs_error': max(incoming_errors),
        'coefficient_optimizer_updates': updates, 'training_CE_examples': updates * 4,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'first_source_soup_identical': True, 'fixed_coefficient_means_exact': True,
        'native_exports_pass': True, 'paired_schedules_with_frozen_engineering': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
