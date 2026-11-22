"""Independent matrix optimizer and actual native-export engineering checks."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_row_matrix import matrix_inverse_root_action, RowMatrixAdam, transport_state
from streaming_row_matrix_training import train_stage


def algebra_check():
    generator = torch.Generator().manual_seed(271117)
    g = torch.randn(12, 2, 5, generator=generator, dtype=torch.float64)
    covariance = g @ g.transpose(1, 2)
    covariance[0] = 0
    covariance[1] = torch.eye(2, dtype=torch.float64)
    vector = torch.randn(12, 2, generator=generator, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    oracle = (eigenvectors @ ((eigenvectors.transpose(1, 2) @ vector.unsqueeze(-1)) /
                              (eigenvalues.clamp_min(0).sqrt() + 1e-8).unsqueeze(-1))).squeeze(-1)
    actual = matrix_inverse_root_action(covariance, vector, 1e-8)
    root_error = float((actual - oracle).abs().max())
    assert torch.allclose(actual, oracle, atol=1e-8, rtol=1e-12)
    rotation, _ = torch.linalg.qr(torch.randn(12, 2, 2, generator=generator, dtype=torch.float64))
    rotated_v = rotation @ covariance @ rotation.transpose(1, 2)
    rotated_m = (rotation @ vector.unsqueeze(-1)).squeeze(-1)
    rotated_u = matrix_inverse_root_action(rotated_v, rotated_m, 1e-8)
    rotation_error = float((rotated_u - (rotation @ actual.unsqueeze(-1)).squeeze(-1)).abs().max())
    assert torch.allclose(rotated_u, (rotation @ actual.unsqueeze(-1)).squeeze(-1), atol=1e-7, rtol=1e-11)
    # Recover moments from an explicit native-space sequence, then compare
    # direct re-projection with the compressed change-of-basis rule.
    q_old = torch.linalg.qr(torch.randn(4, 7, 2, generator=generator, dtype=torch.float64))[0].transpose(1, 2).float()
    q_new = torch.linalg.qr(torch.randn(4, 7, 2, generator=generator, dtype=torch.float64))[0].transpose(1, 2).float()
    scale_old = torch.tensor([1., 2., 0., .3])
    scale_new = torch.tensor([3., .5, 1., 0.])
    sequence = torch.randn(19, 4, 2, generator=generator, dtype=torch.float64)
    sequence[:, 2] = 0
    old_m, old_v = torch.zeros(4, 2, dtype=torch.float64), torch.zeros(4, 2, 2, dtype=torch.float64)
    direct_m, direct_v = old_m.clone(), old_v.clone()
    for g in sequence:
        old_m = .9 * old_m + .1 * g
        old_v = .999 * old_v + .001 * g.unsqueeze(-1) * g.unsqueeze(-2)
        native = (q_old.double().transpose(1, 2) @ (g / scale_old.double().clamp_min(1e-300)[:, None]).unsqueeze(-1)).squeeze(-1)
        projected = scale_new.double()[:, None] * (q_new.double() @ native.unsqueeze(-1)).squeeze(-1)
        direct_m = .9 * direct_m + .1 * projected
        direct_v = .999 * direct_v + .001 * projected.unsqueeze(-1) * projected.unsqueeze(-2)
    old = {'basis': {'w': q_old}, 'scale': scale_old, 'm': old_m, 'v': old_v, 'step': len(sequence)}
    moved, _ = transport_state(old, {'w': q_new}, scale_new, ['w'])
    mean_error = float((moved['m'] - direct_m).abs().max())
    covariance_error = float((moved['v'] - direct_v).abs().max())
    assert mean_error < 1e-12 and covariance_error < 1e-12 and moved['step'] == 19
    # Compare the diagonal arm to independently implemented native FP64 Adam.
    initial = torch.randn(8, 2, generator=generator)
    parameter = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.double())
    optimizer = RowMatrixAdam(parameter, 'diagonal_reset')
    native = torch.optim.Adam([reference], lr=.01)
    diagonal_errors = []
    for _ in range(25):
        gradient = torch.randn(8, 2, generator=generator)
        parameter.grad = gradient.clone()
        reference.grad = gradient.double()
        optimizer.step()
        native.step()
        with torch.no_grad():
            reference.copy_(reference.float().double())
        diagonal_errors.append(float((parameter.detach().double() - reference.detach()).abs().max()))
        assert torch.allclose(optimizer.m, native.state[reference]['exp_avg'], atol=1e-15, rtol=1e-14)
        assert torch.allclose(optimizer.v.diagonal(dim1=1, dim2=2), native.state[reference]['exp_avg_sq'], atol=1e-15, rtol=1e-14)
    assert max(diagonal_errors) < 1e-6
    assert torch.count_nonzero(optimizer.v[:, 0, 1]) == 0
    return {'spectral_inverse_root_oracle_max_error': root_error,
            'orthogonal_coordinate_rotation_max_error': rotation_error,
            'explicit_native_sequence_transport_mean_max_error': mean_error,
            'explicit_native_sequence_transport_covariance_max_error': covariance_error,
            'diagonal_vs_native_FP64_Adam_max_parameter_error': max(diagonal_errors),
            'diagonal_native_Adam_moments_match': True, 'zero_and_repeated_eigenvalues_pass': True}


def main():
    root = Path('runs/streaming_row_matrix_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_row_matrix_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    algebra = algebra_check()
    plan = json.loads((root / 'engineering_plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    records = {}
    for condition in ('diagonal_reset', 'matrix_reset', 'matrix_transport'):
        data = TaskData(plan, source, device)
        leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        previous, history, records[condition], accumulated_steps = None, None, [], 0
        for count, task in enumerate(tasks):
            data.arrive(task)
            assert set(data.train) == set(tasks[:count + 1])
            assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
            incoming = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
            if count == 0:
                previous = incoming
                leaf.load_state_dict(previous, strict=True)
                continue
            state, payload, new_history, record = train_stage(leaf, base, previous, incoming,
                history, count, data, tasks[:count + 1], plan, condition, device)
            assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
            assert record['coefficient_count'] == 346900
            assert float(payload['mean'].abs().max()) > 0
            if condition == 'matrix_transport':
                assert record['optimizer_initial_step'] == accumulated_steps
                assert new_history is not None and record['persistent_auxiliary_history_bytes'] > 0
                if history is not None:
                    # Independently reconstruct every transported row in native coordinates.
                    offset, errors = 0, []
                    for name in plan['active']:
                        rows = len(payload['basis'][name]);sl = slice(offset, offset + rows);offset += rows
                        old_q, new_q = history['basis'][name].double(), payload['basis'][name].double()
                        old_scale, new_scale = history['scale'][sl].double(), payload['scale'][sl].double()
                        lifted = (old_q.transpose(1, 2) @ history['m'][sl].unsqueeze(-1)).squeeze(-1)
                        lifted = torch.where(old_scale[:, None] > 0, lifted / old_scale.clamp_min(1e-300)[:, None], 0.)
                        expected = new_scale[:, None] * (new_q @ lifted.unsqueeze(-1)).squeeze(-1)
                        errors.append(float((expected - payload['optimizer_initial']['m'][sl]).abs().max()))
                    assert max(errors) < 1e-10
                    record['independent_native_first_moment_transport_error'] = max(errors)
            else:
                assert record['optimizer_initial_step'] == 0 and new_history is None
                assert torch.count_nonzero(payload['optimizer_initial']['m']) == 0
                assert torch.count_nonzero(payload['optimizer_initial']['v']) == 0
            accumulated_steps += record['updates']
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(payload, root / f'engineering_{condition}_stage{count + 1}_parameters.pt', root, storage)
            records[condition].append(record)
            previous, history = state, new_history
            del incoming, state, payload
            gc.collect()
            torch.cuda.empty_cache()
        assert not data.valid
        del data, leaf, previous, history, new_history
        gc.collect()
        torch.cuda.empty_cache()
    frozen = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert all([r['schedule_hash'] for r in rows] == [r['schedule_hash'] for r in frozen] for rows in records.values())
    updates = sum(r['updates'] for rows in records.values() for r in rows)
    assert updates == 54
    sources = ['streaming_row_matrix.py', 'streaming_row_matrix_training.py', 'check_streaming_row_matrix.py']
    write_json(root / 'engineering.json', {'passed': True, 'algebra': algebra,
        'coefficient_optimizer_updates': updates, 'training_CE_examples': updates * 4,
        'extra_direction_construction_forward_examples': 0, 'extra_direction_construction_backward_probes': 0,
        'native_initialization_matches_bilateral': True, 'fixed_coefficient_means_exact': True,
        'native_exports_pass': True, 'paired_schedules_with_frozen_engineering': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: sha256(Path(name)) for name in sources}})


if __name__ == '__main__':
    main()
