"""Native export and coefficient-mean checks for the four strict controls."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf
from run_streaming_strict_history import TaskData, strict_settings, train_stage, write_json
from streaming_history_sketch_merge import RANK, update_sketch


def main():
    root = Path('runs/streaming_strict_history_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    plan = json.loads(Path('runs/streaming_history_sketch_v1/plan.json').read_text())
    counts = {'incoming_rows': 174216, 'old_rows': 347666,
              'history_sketch': 694566, 'random_sketch': 694566}
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4, per_arm_coefficients=counts)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records, reference_history = {}, None
    for condition in counts:
        history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
        previous = experts[tasks[0]]
        leaf.load_state_dict(previous, strict=True)
        update_sketch(history, previous, base, plan['active'])
        records[condition] = []
        for count in range(1, len(tasks)):
            state, coefficients, record = train_stage(
                leaf, base, previous, experts[tasks[count]], history, count, data,
                tasks[:count + 1], plan, condition, device, {})
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            record['history_sketch_update'] = update_sketch(history, experts[tasks[count]], base, plan['active'])
            torch.save(state, root / f'engineering_{condition}_stage{count + 1}.ckpt')
            records[condition].append(record)
            previous = state
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
        if reference_history is None:
            reference_history = {name: value.clone() for name, value in history.items()}
        else:
            assert all(torch.equal(value, reference_history[name]) for name, value in history.items())
        torch.save(history, root / f'engineering_{condition}_history.pt')
        print(condition, 'complete', flush=True)
        del history, previous, state
    reference_schedules = [r['schedule_hash'] for r in records['incoming_rows']]
    assert all([r['schedule_hash'] for r in arm] == reference_schedules for arm in records.values())
    updates = sum(r['updates'] for arm in records.values() for r in arm)
    assert updates == 72 and not data.valid
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'extra_score_VJPs': 0, 'strict_deterministic_algorithms': True,
        'fixed_coefficient_means_exact': True, 'all_four_raw_histories_identical': True,
        'paired_schedules': True, 'coefficient_counts': counts, 'records': records,
        'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
