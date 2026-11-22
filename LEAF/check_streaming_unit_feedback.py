"""Check unit-weight feedback and its exact pre-intervention prefix."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf
from run_streaming_strict_history import TaskData, strict_settings, train_stage, write_json
from streaming_unit_feedback import RANK, update_memory


def main():
    root = Path('runs/streaming_unit_feedback_v1')
    assert not (root / 'engineering.json').exists()
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    plan = json.loads(Path('runs/streaming_complement_memory_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4)
    plan['per_arm_coefficients']['history_sketch'] = 694566
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    previous = experts[tasks[0]]
    leaf.load_state_dict(previous, strict=True)
    update_memory(history, base, None, previous, previous, 0, plan['active'], 'unit_feedback')
    records = []
    for count in range(1, len(tasks)):
        state, coefficients, record = train_stage(
            leaf, base, previous, experts[tasks[count]], history, count, data,
            tasks[:count + 1], plan, 'history_sketch', device, {})
        assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
        if count in (1, 2):
            reference = torch.load(f'runs/streaming_complement_memory_v1/engineering_joint_memory_stage{count + 1}.ckpt',
                                   map_location='cpu', weights_only=True)
            assert all(torch.equal(state[name], reference[name]) for name in state)
        record['memory_update'] = update_memory(history, base, previous, experts[tasks[count]], state,
                                                 count, plan['active'], 'unit_feedback')
        torch.save(state, root / f'engineering_unit_feedback_stage{count + 1}.ckpt')
        records.append(record)
        previous = state
        del coefficients
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(history, root / 'engineering_unit_feedback_history.pt')
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert [r['schedule_hash'] for r in records] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for r in records)
    assert updates == 18 and not data.valid
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'extra_score_VJPs': 0, 'strict_deterministic_algorithms': True,
        'fixed_coefficient_means_exact': True, 'stages2_and3_before_feedback_intervention_exact_native_weights': True,
        'paired_schedules_with_real_joint_engineering': True, 'coefficient_count': 694566,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
