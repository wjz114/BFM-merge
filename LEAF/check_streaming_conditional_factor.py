"""Independent conditional rank-one oracle and actual native-model integration."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256, PrototypeClassifier
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_conditional_factor import (
    StreamingMerger, append_expert, history_bytes, fit_factors, projection_columns,
    FILM_W, FILM_B, CROSS, WIDTH)
from streaming_conditional_factor_training import train_stage


def geometry_check():
    generator = torch.Generator().manual_seed(92017)
    x = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    x[-1] = x[0]
    x[:, -1] = 1.
    target = torch.randn(7, 13, 6, generator=generator, dtype=torch.float64)
    a0 = torch.randn(7, 13, generator=generator, dtype=torch.float64)
    f0 = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    target[1] = 0
    a0[2] = 0
    target[3] = a0[3, :, None] * (f0[3] @ x.T)[None, :]
    target[4] = 0
    a0[4] = 0
    a, f, audit = fit_factors(target, x, a0, f0)
    projection = x @ torch.linalg.pinv(x)
    svd_errors, optimal_errors = [], []
    for c in range(len(target)):
        u, s, vh = torch.linalg.svd(target[c] @ projection, full_matrices=False)
        oracle = s[0] * u[:, :1] @ vh[:1]
        actual = a[c, :, None] * (f[c] @ x.T)[None, :]
        svd_errors.append(float((actual - oracle).abs().max()))
        optimal_errors.append(float(((actual - target[c]).square().sum() -
                                     (oracle - target[c]).square().sum()).abs()))
    assert max(svd_errors) < 1e-10 and max(optimal_errors) < 1e-10
    assert torch.equal(a[1], a0[1]) and torch.equal(a[4], a0[4])
    assert torch.allclose(f[4], f0[4], atol=1e-12, rtol=0)
    assert torch.allclose((f @ x.T)[:, 0], (f @ x.T)[:, -1], atol=1e-12, rtol=0)
    # Underdetermined instruction maps must retain the reference nullspace.
    wide_x = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    wide_f0 = torch.randn(7, 8, generator=generator, dtype=torch.float64)
    _, wide_f, _ = fit_factors(target[:, :, :3], wide_x, a0, wide_f0)
    null = torch.eye(8, dtype=torch.float64) - torch.linalg.pinv(wide_x) @ wide_x
    null_error = float(((wide_f - wide_f0) @ null).abs().max())
    assert null_error < 1e-11
    return {'independent_full_SVD_max_product_error': max(svd_errors),
            'independent_full_SVD_max_objective_error': max(optimal_errors),
            'instruction_nullspace_change': null_error,
            'duplicate_instructions_zero_targets_and_zero_references_pass': True,
            'fit_audit': audit}


def main():
    root = Path('runs/streaming_conditional_factor_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_conditional_factor_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    geometry = geometry_check()
    plan = json.loads(Path('runs/streaming_column_residual_v1/plan.json').read_text())
    conditions = ('coupled_factors', 'factorized_control')
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4,
                per_arm_coefficients={c: 347922 for c in conditions})
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    data = TaskData(plan, source, device)
    for task in tasks:
        data.arrive(task)
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {t: torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records, first_history = {}, None
    for condition in conditions:
        history = []
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        records[condition] = []
        for count, task in enumerate(tasks):
            incoming = experts[task]
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            append_expert(history, incoming, wrapper.instruction, task)
            del wrapper
            assert history_bytes(history) == (count + 1) * 2105352
            if count == 0:
                continue
            reference = BilateralMerger(leaf, base, previous, incoming, {}, count,
                                        plan['active'], 'old_rows', {})
            candidate = StreamingMerger(leaf, base, previous, incoming, history, count,
                                        plan['active'], condition, {})
            a, b = reference.materialize(), candidate.materialize()
            assert all(torch.equal(a[n], b[n]) for n in a)
            initial = candidate.coefficients.detach().cpu().clone()
            assert initial.numel() == 347922 and (initial[347666:] == 0).all()
            assert candidate.factor_audit['relative_optimality_gap'] < 1e-10
            assert candidate.factor_audit['max_affine_scale_reconstruction_error'] < 1e-10
            assert candidate.factor_audit['max_relative_column_norm_error'] < 1e-12
            # A channel gate jointly changes gamma and all K/V projections;
            # beta rows and Q rows stay exactly at their bilateral values.
            eta = initial.to(device)
            eta[-WIDTH:] = 1.
            endpoint = candidate.materialize(eta)
            assert torch.equal(endpoint[FILM_W][WIDTH:], a[FILM_W][WIDTH:])
            assert torch.equal(endpoint[FILM_B][WIDTH:], a[FILM_B][WIDTH:])
            assert all(torch.equal(endpoint[n][:WIDTH], a[n][:WIDTH]) for n in CROSS)
            x = torch.stack([r['instruction'] for r in history])
            columns = torch.stack([r['columns'].double() for r in history], dim=-1)
            scales = torch.stack([r['scale'] for r in history], dim=-1)
            if condition == 'factorized_control':
                columns = columns.mean(-1, keepdim=True).expand_as(columns)
            target = columns * scales[:, None, :]
            af = projection_columns(endpoint).double()
            ff = torch.cat([endpoint[FILM_W][:WIDTH].double(),
                            endpoint[FILM_B][:WIDTH, None].double() + 1.], dim=1)
            error = float((target - af[:, :, None] * (ff @ x.T)[:, None, :]).square().sum())
            relative_export_gap = abs(error - candidate.factor_audit['rank_one_optimum_squared_error']) / float(target.square().sum())
            assert relative_export_gap < 1e-7
            del reference, candidate, a, b, endpoint, eta, columns, scales, target, af, ff
            state, coefficients, record = train_stage(leaf, base, previous, incoming, history, count,
                data, tasks[:count + 1], plan, condition, device, {})
            assert torch.equal(coefficients['initial'], initial)
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert float(coefficients['mean'][347666:].abs().max()) > 0
            record['initial_native_weights_exactly_match_bilateral'] = True
            record['native_FP32_factor_endpoint_relative_optimality_gap'] = relative_export_gap
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(coefficients, root / f'engineering_{condition}_stage{count + 1}_coefficients.pt', root, storage)
            records[condition].append(record)
            previous = state
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
        if first_history is None:
            first_history = history
        else:
            assert all(a['task'] == b['task'] and all(torch.equal(a[k], b[k])
                       for k in ('columns', 'scale', 'instruction')) for a, b in zip(first_history, history))
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    for items in records.values():
        assert [r['schedule_hash'] for r in items] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for items in records.values() for r in items)
    assert updates == 36 and not data.valid
    sources = ['streaming_conditional_factor.py', 'streaming_conditional_factor_training.py',
               'run_streaming_conditional_factor.py', 'check_streaming_conditional_factor.py']
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'strict_deterministic_algorithms': True, 'geometry': geometry,
        'initial_native_weights_exactly_match_bilateral': True,
        'original_history_identical_across_calibration_paths': True,
        'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
        'paired_schedules_with_frozen_engineering': True, 'coefficient_count': 347922,
        'nonzero_learned_extra_coefficients': True, 'history_bytes_per_expert': 2105352,
        'history_bytes_after_four_experts': history_bytes(history), 'history_bytes_after_16_experts': 33685632,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: sha256(Path(name)) for name in sources}})


if __name__ == '__main__':
    main()
