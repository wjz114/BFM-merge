"""Real four-task export and exact unchanged-CE trajectory checks."""
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_strict_history import TaskData, strict_settings, write_json
from source_behavior_repair_training import CONDITIONS, train_stage
from streaming_native_complement_training import train_stage as reference_train_stage


def main():
    root = Path('runs/source_behavior_preserving_repair_v1')
    path = root / 'engineering_plan.json'
    plan = json.loads(path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    strict_settings()
    device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    data.experts, data.teacher_logits = {}, {}
    for task in plan['pilot_tasks']:
        data.arrive(task)
        source_path = Path(source['experts'][task])
        assert sha256(source_path) == plan['expert_sha256'][str(source_path)]
        data.experts[task] = torch.load(source_path, map_location='cpu', weights_only=True)
        target_path = Path(plan['teacher_targets'][task])
        assert sha256(target_path) == plan['teacher_sha256'][task]
        target = torch.load(target_path, map_location='cpu', weights_only=True)
        assert torch.equal(target['labels'], data.train[task][1])
        data.teacher_logits[task] = target['logits']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    previous = torch.load(plan['previous'], map_location='cpu', weights_only=True)
    incoming = data.experts[plan['pilot_tasks'][-1]]
    records, plain, start = {}, None, time.monotonic()
    for condition in CONDITIONS:
        leaf.load_state_dict(previous, strict=True)
        state, payload, record = train_stage(leaf, base, previous, incoming, 3, data,
            plan['pilot_tasks'], plan, condition, device)
        assert record['updates'] == 8
        records[condition] = record
        if condition == 'plain_ce':
            plain = state, payload
        else:
            assert all(torch.isfinite(v).all() for v in payload['native_mean'].values())
        del state, payload
    leaf.load_state_dict(previous, strict=True)
    state, payload, reference = reference_train_stage(leaf, base, previous, incoming, 3, data,
        plan['pilot_tasks'], plan, 'unprojected_hybrid', device, None)
    assert all(torch.equal(v, state[n]) for n, v in plain[0].items())
    assert torch.equal(payload['snapshots'], plain[1]['snapshots'])
    assert all(torch.equal(v, payload['native_mean'][n]) for n, v in plain[1]['native_mean'].items())
    assert all(torch.equal(a[n], b[n]) for a, b in zip(payload['native_snapshots'],
        plain[1]['native_snapshots']) for n in a)
    result = dict(status='passed', plan_sha256=sha256(path), records=records,
        unchanged_plain_CE_trajectory_bitwise=True, optimizer_updates=32,
        CE_examples=128, auxiliary_student_forward_examples=32,
        official_test_read=False, elapsed_seconds=time.monotonic() - start,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
