"""Verify residual geometry, fixed-budget integration, and native exports."""
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from fisher_moment_merge import add_moments
from run_streaming_moment_transport import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_fisher_residual import StreamingMerger, residual_row_direction
from streaming_fisher_residual_training import train_stage


def geometry_check():
    g = torch.Generator().manual_seed(92017)
    a, b, f = [torch.randn(7, 11, generator=g, dtype=torch.float64) for _ in range(3)]
    b[1] = 2 * a[1]
    a[2] = 0
    b[3] = 0
    f[4] = a[4] - b[4]
    a[5] = 0
    f[6] = 0
    actual = residual_row_direction(a, b, f)
    expected = []
    for i in range(len(a)):
        basis = torch.stack([a[i], b[i]], dim=1)
        residual = f[i] - basis @ torch.linalg.pinv(basis) @ f[i]
        if residual.norm() <= 1e-6 * f[i].norm() or b[i].norm() == 0:
            expected.append(torch.zeros_like(f[i]))
        else:
            expected.append(residual * b[i].norm() / residual.norm())
    expected = torch.stack(expected)
    error = float((actual.double() - expected).abs().max())
    assert error < 3e-7
    assert (actual[3] == 0).all() and (actual[4] == 0).all() and (actual[6] == 0).all()
    # The fixed shift preserves per-row Fisher multisets, not their coordinates.
    weights = torch.arange(1., 13.).reshape(3, 4)
    old = {'x': torch.zeros(3, 4)}
    incoming = {'x': torch.arange(12.).reshape(3, 4)}
    m = {'x': torch.zeros(3, 3, 4, dtype=torch.float64)}
    add_moments(m, {'x': weights}, incoming, old, ['x'], 0, 0, shuffle=True)
    assert torch.equal(m['x'][0].sort(1).values, weights.double().sort(1).values)
    assert not torch.equal(m['x'][0], weights.double())
    assert torch.equal(m['x'][1], m['x'][0] * incoming['x'])
    assert torch.equal(m['x'][2], incoming['x'].double())
    return {'independent_pseudoinverse_max_abs_error': error,
            'zero_collinear_and_in_plane_cases_pass': True,
            'shift_preserves_Fisher_row_multisets_and_changes_assignment': True}


def main():
    root = Path('runs/streaming_fisher_residual_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_fisher_residual_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    geometry = geometry_check()
    plan = json.loads(Path('runs/streaming_moment_transport_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4,
                per_arm_coefficients={'fisher_residual': 521116, 'shifted_fisher_residual': 521116})
    source = json.loads(Path(plan['source_fit']).read_text())
    cache = json.loads(Path(plan['fisher_cache']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    data = TaskData(plan, source, device)
    for task in tasks:
        data.arrive(task)
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
        assert cache['experts'][task]['expert_sha256'] == plan['expert_sha256'][task]
        for key, value in cache['experts'][task]['calibration_audit'].items():
            assert data.records[task][key] == value
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {t: torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records, first_moments = {}, None
    for condition in ('fisher_residual', 'shifted_fisher_residual'):
        moments = {n: torch.zeros((3, *base[n].shape), dtype=torch.float64) for n in plan['active']}
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        records[condition] = []
        for count, task in enumerate(tasks):
            incoming = experts[task]
            payload = cache['experts'][task]
            assert sha256(Path(payload['path'])) == payload['sha256']
            fisher = torch.load(payload['path'], map_location='cpu', weights_only=True)
            add_moments(moments, fisher, incoming, base, plan['active'],
                        plan['tasks'].index(task), plan['seed'],
                        shuffle=condition == 'shifted_fisher_residual')
            del fisher
            if count == 0:
                continue
            reference = BilateralMerger(leaf, base, previous, incoming, {}, count,
                                        plan['active'], 'old_rows', {})
            candidate = StreamingMerger(leaf, base, previous, incoming, moments, count,
                                        plan['active'], condition, {})
            a, b = reference.materialize(), candidate.materialize()
            assert all(torch.equal(a[n], b[n]) for n in a)
            initial = candidate.coefficients.detach().cpu().clone()
            assert initial.numel() == 521116 and (initial[347666:] == 0).all()
            assert candidate.residual_audit['nonzero_rows'] > 0
            assert candidate.residual_audit['max_abs_cosine_to_old_or_new'] < 2e-6
            assert candidate.residual_audit['max_relative_target_norm_error'] < 2e-7
            del reference, candidate, a, b
            state, coefficients, record = train_stage(leaf, base, previous, incoming, moments, count,
                data, tasks[:count + 1], plan, condition, device, {})
            assert torch.equal(coefficients['initial'], initial)
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert float(coefficients['mean'][347666:].abs().max()) > 0
            record['initial_native_weights_exactly_match_bilateral'] = True
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(coefficients, root / f'engineering_{condition}_stage{count + 1}_coefficients.pt', root, storage)
            records[condition].append(record)
            previous = state
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
        if first_moments is None:
            first_moments = moments
        else:
            assert all(torch.equal(moments[n][2], first_moments[n][2]) for n in moments)
            assert any(not torch.equal(moments[n][1], first_moments[n][1]) for n in moments)
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    for items in records.values():
        assert [r['schedule_hash'] for r in items] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for items in records.values() for r in items)
    assert updates == 36 and not data.valid
    moment_bytes = sum(v.numel() * v.element_size() for v in moments.values())
    assert moment_bytes == 618037248
    sources = ['fisher_moment_merge.py', 'streaming_fisher_residual.py',
               'streaming_fisher_residual_training.py', 'run_streaming_fisher_residual.py',
               'check_streaming_fisher_residual.py']
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'new_Fisher_estimation_examples': 0, 'new_Fisher_estimation_VJPs': 0,
        'strict_deterministic_algorithms': True, 'geometry': geometry,
        'initial_native_weights_exactly_match_bilateral': True,
        'control_preserves_original_uniform_sum_and_changes_weighted_moments': True,
        'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
        'paired_schedules_with_frozen_engineering': True, 'coefficient_count': 521116,
        'nonzero_learned_residual_coefficients': True, 'moment_bytes': moment_bytes,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}})


if __name__ == '__main__':
    main()
