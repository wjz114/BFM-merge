"""Check full causal history and its exact pre-intervention prefix."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from run_streaming_full_history import save_tensor_file
from d_downstream_qformer_ce import load_leaf
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_full_history_training import train_stage
from streaming_full_history import initialize_history, update_memory


def main():
    root = Path('runs/streaming_full_history_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_full_history_v1/engineering')
    storage.mkdir(exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    plan = json.loads(Path('runs/streaming_complement_memory_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4, bilateral_coefficients=347666, row_count=173450)
    plan['per_arm_coefficients']['history_sketch'] = 694566
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = initialize_history(base, plan['active'])
    previous = experts[tasks[0]]
    leaf.load_state_dict(previous, strict=True)
    update_memory(history, base, None, previous, previous, 0, plan['active'], 'full_history')
    records = []
    for count in range(1, len(tasks)):
        state, coefficients, record = train_stage(
            leaf, base, previous, experts[tasks[count]], history, count, data,
            tasks[:count + 1], plan, 'full_history', device, {})
        assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
        if count == 1:
            reference = torch.load(f'runs/streaming_complement_memory_v1/engineering_joint_memory_stage{count + 1}.ckpt',
                                   map_location='cpu', weights_only=True)
            assert all(torch.equal(state[name], reference[name]) for name in state)
        record['memory_update'] = update_memory(history, base, previous, experts[tasks[count]], state,
                                                 count, plan['active'], 'full_history')
        save_tensor_file(state, root / f'engineering_full_history_stage{count + 1}.ckpt', root, storage)
        records.append(record)
        previous = state
        del coefficients
        gc.collect()
        torch.cuda.empty_cache()
    save_tensor_file(history, root / 'engineering_full_history_history.pt', root, storage)
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert [r['schedule_hash'] for r in records] == [r['schedule_hash'] for r in reference_records]
    prefix_updates = sum(r['updates'] for r in records)
    assert prefix_updates == 18 and not data.valid
    # Stress the final scientific slot count with synthetic repeated known
    # expert vectors; this is engineering, not a causal performance run.
    stress_history = {name: torch.stack([experts[tasks[i % 4]][name] - base[name] for i in range(15)])
                      for name in plan['active']}
    plan.update(rounds=1, average_rounds=[1])
    stress_state, stress_coefficients, stress_record = train_stage(
        leaf, base, previous, experts[tasks[-1]], stress_history, 15, data,
        tasks, plan, 'full_history', device, {})
    assert stress_record['updates'] == 4 and stress_record['coefficient_count'] == 2949416
    assert torch.equal(stress_coefficients['snapshots'].double().mean(0).float(), stress_coefficients['mean'])
    save_tensor_file(stress_state, root / 'engineering15slot_stress.ckpt', root, storage)
    updates = prefix_updates + stress_record['updates']
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'extra_score_VJPs': 0, 'strict_deterministic_algorithms': True,
        'fixed_coefficient_means_exact': True, 'stage2_with_zero_projected_history_exact_native_weights': True,
        'paired_prefix_schedules_with_real_joint_engineering': True,
        'coefficient_counts_by_prefix_stage': [r['coefficient_count'] for r in records],
        'max_slot_stress_record': stress_record, 'max_slot_stress_is_synthetic_engineering': True,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
