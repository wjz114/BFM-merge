"""Independent SVD/algebra checks and strict native calibration smoke test."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf
from run_streaming_strict_history import TaskData, strict_settings, train_stage, write_json
from streaming_complement_memory import CONDITIONS, RANK, compress_rows, update_memory


def covariance(k):
    return k.double().transpose(-1, -2) @ k.double()


def algebra():
    g = torch.Generator().manual_seed(52381)
    b = torch.randn(9, 4, 13, generator=g, dtype=torch.float64)
    anchor = torch.randn(9, 13, generator=g, dtype=torch.float64)
    q = anchor / anchor.norm(dim=1, keepdim=True)
    r = b - (b * q[:, None]).sum(2, keepdim=True) * q[:, None]
    _, s, vh = torch.linalg.svd(r, full_matrices=False)
    expected = s[:, :2, None] * vh[:, :2]
    k, stats = compress_rows(b, anchor, 'complement_memory')
    error = float((covariance(k) - covariance(expected)).abs().max())
    assert error < 2e-5 and stats['max_stored_anchor_cosine_abs'] < 1e-6
    rotation = torch.linalg.qr(torch.randn(9, 4, 4, generator=g, dtype=torch.float64)).Q
    rotated, _ = compress_rows(rotation @ b, anchor, 'complement_memory')
    rotation_error = float((covariance(k) - covariance(rotated)).abs().max())
    assert rotation_error < 2e-5
    toy = torch.tensor([[[10., 0., 0.], [0., 2., 0.], [0., 0., 1.], [0., 0., 0.]]], dtype=torch.float64)
    a = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    plain, _ = compress_rows(toy, a, 'joint_memory')
    complement, _ = compress_rows(toy, a, 'complement_memory')
    rank_plain = int(torch.linalg.matrix_rank(torch.cat([a[:, None], plain.double()], 1)))
    rank_complement = int(torch.linalg.matrix_rank(torch.cat([a[:, None], complement.double()], 1)))
    assert (rank_plain, rank_complement) == (2, 3)
    basis = torch.linalg.qr(torch.randn(13, 3, generator=g, dtype=torch.float64)).Q.T
    coefficients = torch.randn(4, 3, generator=g, dtype=torch.float64)
    low_rank = (coefficients @ basis)[None]
    low_anchor = low_rank[:, 0] + .3 * low_rank[:, 2]
    retained, _ = compress_rows(low_rank, low_anchor, 'complement_memory')
    represented = torch.cat([low_anchor[:, None], retained.double()], 1)
    projector = torch.linalg.pinv(represented) @ represented
    span_error = float((low_rank - low_rank @ projector).norm() / low_rank.norm())
    assert span_error < 1e-6
    zero, _ = compress_rows(b, torch.zeros_like(anchor), 'complement_memory')
    unprojected, _ = compress_rows(b, anchor, 'joint_memory')
    assert torch.equal(zero, unprojected)
    parallel = torch.arange(1, 5, dtype=torch.float64)[None, :, None] * anchor[:, None]
    empty, _ = compress_rows(parallel, anchor, 'complement_memory')
    assert torch.count_nonzero(empty) == 0
    return {'independent_svd_covariance_max_error': error, 'slot_rotation_covariance_max_error': rotation_error,
            'stored_anchor_cosine_max': stats['max_stored_anchor_cosine_abs'],
            'capacity_toy_joint_rank': rank_plain, 'capacity_toy_complement_rank': rank_complement,
            'rank3_span_relative_error': span_error, 'zero_anchor_matches_unprojected': True,
            'parallel_dictionary_zero_after_projection': True}


def main():
    root = Path('runs/streaming_complement_memory_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    algebra_checks = algebra()
    plan = json.loads(Path('runs/streaming_strict_history_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records, first_coefficients = {}, None
    for condition in CONDITIONS:
        history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        update_memory(history, base, None, previous, previous, 0, plan['active'], condition)
        records[condition] = []
        for count in range(1, len(tasks)):
            state, coefficients, record = train_stage(
                leaf, base, previous, experts[tasks[count]], history, count, data,
                tasks[:count + 1], plan, 'history_sketch', device, {})
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            if count == 1:
                if first_coefficients is None:
                    first_coefficients = coefficients['snapshots'].clone()
                else:
                    assert torch.equal(first_coefficients, coefficients['snapshots'])
            record['memory_update'] = update_memory(history, base, previous, experts[tasks[count]], state,
                                                     count, plan['active'], condition)
            if condition == 'complement_memory':
                assert record['memory_update']['max_stored_anchor_cosine_abs'] < 1e-6
            torch.save(state, root / f'engineering_{condition}_stage{count + 1}.ckpt')
            records[condition].append(record)
            previous = state
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
        torch.save(history, root / f'engineering_{condition}_history.pt')
        print(condition, 'complete', flush=True)
        del history, previous, state
    schedules = [[r['schedule_hash'] for r in records[c]] for c in CONDITIONS]
    assert schedules[0] == schedules[1]
    updates = sum(r['updates'] for arm in records.values() for r in arm)
    assert updates == 36 and not data.valid
    write_json(root / 'engineering.json', {
        'passed': True, 'algebra': algebra_checks, 'coefficient_optimizer_updates': updates,
        'CE_example_passes': updates * 4, 'extra_score_VJPs': 0,
        'strict_deterministic_algorithms': True, 'fixed_coefficient_means_exact': True,
        'stage2_before_policy_intervention_exact_coefficients': True, 'paired_schedules': True,
        'coefficient_count_both_arms': 694566, 'records': records,
        'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
