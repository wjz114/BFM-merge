"""Native export, initialization, and calibration checks for both column arms."""
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf
from run_streaming_full_history import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger as BilateralMerger
from streaming_column_residual import StreamingMerger
from streaming_column_residual_training import train_stage


def main():
    root = Path('runs/streaming_column_residual_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_column_residual_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    plan = json.loads(Path('runs/streaming_complement_memory_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4,
                per_arm_coefficients={'raw_columns': 460306, 'projected_columns': 460306})
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records = {}
    for condition in ('raw_columns', 'projected_columns'):
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        records[condition] = []
        for count in range(1, len(tasks)):
            incoming = experts[tasks[count]]
            reference = BilateralMerger(leaf, base, previous, incoming, {}, count,
                                        plan['active'], 'old_rows', {})
            candidate = StreamingMerger(leaf, base, previous, incoming, {}, count,
                                        plan['active'], condition, {})
            a, b = reference.materialize(), candidate.materialize()
            assert all(torch.equal(a[name], b[name]) for name in a)
            assert candidate.coefficients.numel() == 460306 and len(candidate.column_slices) == 124
            del reference, candidate, a, b
            state, coefficients, record = train_stage(
                leaf, base, previous, incoming, {}, count, data, tasks[:count + 1],
                plan, condition, device, {})
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert float(coefficients['mean'][347666:].abs().max()) > 0
            record['initial_native_weights_exactly_match_bilateral'] = True
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            records[condition].append(record)
            previous = state
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    for items in records.values():
        assert [r['schedule_hash'] for r in items] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for items in records.values() for r in items)
    assert updates == 36 and not data.valid
    sources = ['streaming_column_residual.py', 'streaming_column_residual_training.py',
               'check_streaming_column_residual.py']
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'extra_score_VJPs': 0, 'strict_deterministic_algorithms': True,
        'initial_native_weights_exactly_match_bilateral': True,
        'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
        'paired_schedules_with_frozen_engineering': True, 'nonzero_learned_column_coefficients': True,
        'coefficient_count': 460306, 'eligible_matrix_count': 124,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}})


if __name__ == '__main__':
    main()
