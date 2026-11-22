"""Independent complement checks and three-arm native-model calibration."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_native_complement import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_native_complement import StreamingMerger, project_complement
from streaming_projected_sketch_v2 import orthonormal_rows
from streaming_native_complement_training import train_stage


def geometry_check():
    generator = torch.Generator().manual_seed(92017)
    directions = torch.randn(7, 2, 11, generator=generator, dtype=torch.float64)
    directions[1, 1] = 2 * directions[1, 0]
    directions[2, 0] = 0
    directions[3] = 0
    q = orthonormal_rows(directions)
    value = torch.randn(7, 11, generator=generator)
    actual = project_complement(value, q)
    expected, projectors = [], []
    for d, v in zip(directions, value):
        projection = torch.eye(11, dtype=torch.float64) - torch.linalg.pinv(d) @ d
        projectors.append(projection)
        expected.append(projection @ v.double())
    expected = torch.stack(expected)
    error = float((actual.double() - expected).abs().max())
    assert error < 1e-6
    parameter = torch.nn.Parameter(torch.zeros_like(value))
    hook = parameter.register_hook(lambda gradient: project_complement(gradient, q))
    (parameter * value).sum().backward()
    assert torch.allclose(parameter.grad.double(), expected, atol=1e-6, rtol=0)
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    optimizer.step()
    with torch.no_grad():
        parameter.copy_(project_complement(parameter, q))
    expected_step = torch.stack([p @ (-1e-5 * e / (e.abs() + 1e-8)) for p, e in zip(projectors, expected)])
    step_error = float((parameter.detach().double() - expected_step).abs().max())
    assert step_error < 1e-10
    hook.remove()
    zero_q = orthonormal_rows(torch.tensor([[[0.], [0.]], [[1.], [2.]]]))
    scalar_result = project_complement(torch.ones(2), zero_q)
    assert torch.equal(scalar_result, torch.tensor([1., 0.]))
    return {'independent_pseudoinverse_projection_error': error,
            'independent_projected_Adam_first_step_error': step_error,
            'gradient_hook_and_zero_collinear_scalar_cases_pass': True}


def main():
    root = Path('runs/streaming_native_complement_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_native_complement_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    geometry = geometry_check()
    plan = json.loads(Path('runs/streaming_conditional_factor_v1/plan.json').read_text())
    conditions = ('complement_hybrid', 'unprojected_hybrid', 'native_only')
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4, native_learning_rate=1e-5)
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    data = TaskData(plan, source, device)
    for task in tasks:
        data.arrive(task)
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {t: torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records = {}
    for condition in conditions:
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        records[condition] = []
        for count, task in enumerate(tasks[1:], 1):
            incoming = experts[task]
            reference = BilateralMerger(leaf, base, previous, incoming, {}, count,
                                        plan['active'], 'old_rows', {})
            candidate = StreamingMerger(leaf, base, previous, incoming, count,
                                        plan['active'], condition, {})
            a, b = reference.materialize(), candidate.materialize()
            assert all(torch.equal(a[n], b[n]) for n in a)
            initial = candidate.coefficients.detach().cpu().clone()
            assert initial.numel() == 347666 and all((v == 0).all() for v in candidate.residual.values())
            del reference, candidate, a, b
            state, parameters, record = train_stage(leaf, base, previous, incoming, count,
                data, tasks[:count + 1], plan, condition, device, {})
            assert torch.equal(parameters['initial'], initial)
            assert torch.equal(parameters['snapshots'].double().mean(0).float(), parameters['mean'])
            assert len(parameters['native_snapshots']) == 2
            for name in plan['active']:
                expected = torch.stack([s[name] for s in parameters['native_snapshots']]).double().mean(0).float()
                assert torch.equal(expected, parameters['native_mean'][name])
            assert record['native_residual_audit']['squared_norm'] > 0
            if condition == 'complement_hybrid':
                assert record['native_residual_audit']['max_relative_row_projection'] < 2e-6
                assert record['basis_audit']['padded_basis_bytes'] == 206012416
            else:
                assert record['basis_audit']['padded_basis_bytes'] == 0
            if condition == 'native_only':
                assert all(torch.equal(s, initial) for s in parameters['snapshots'])
                assert record['trainable_parameters'] == 25751552
            else:
                assert not torch.equal(parameters['mean'], initial)
                assert record['trainable_parameters'] == 26099218
            assert all(torch.equal(state[n], base[n]) for n in base if n not in plan['active'])
            record['initial_native_weights_exactly_match_bilateral'] = True
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(parameters, root / f'engineering_{condition}_stage{count + 1}_parameters.pt', root, storage)
            records[condition].append(record)
            previous = state
            del parameters, expected
            gc.collect()
            torch.cuda.empty_cache()
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    for items in records.values():
        assert [r['schedule_hash'] for r in items] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for items in records.values() for r in items)
    assert updates == 54 and not data.valid
    sources = ['streaming_native_complement.py', 'streaming_native_complement_training.py',
               'run_streaming_native_complement.py', 'check_streaming_native_complement.py',
               'streaming_projected_sketch_v2.py']
    write_json(root / 'engineering.json', {
        'passed': True, 'optimizer_update_calls': updates, 'joint_coefficient_and_native_updates': 36,
        'native_only_updates': 18, 'CE_example_passes': updates * 4,
        'extra_direction_construction_EEG_passes': 0, 'strict_deterministic_algorithms': True,
        'geometry': geometry, 'initial_native_weights_exactly_match_bilateral': True,
        'fixed_coefficient_and_native_means_exact': True, 'native_exports_pass': True,
        'paired_schedules_with_frozen_engineering': True,
        'nonzero_learned_native_residuals': True, 'native_only_coefficients_fixed': True,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: sha256(Path(name)) for name in sources}})


if __name__ == '__main__':
    main()
