"""Engineering checks: history recurrence, maximum-shape backward, native export."""
import json
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf
from run_streaming_residual_merge_v3 import TaskData, train_stage, write_json
from streaming_residual_merge import add_history


def main():
    root = Path('runs/streaming_merge_v3')
    root.mkdir(exist_ok=True)
    destination = root / 'engineering.json'
    assert not destination.exists()
    source_path = Path('runs/paper_merge_v2/seed0/original_fp32_tensor_balanced_ce/run.json')
    source = json.loads(source_path.read_text())
    plan = {'calibration': 'runs/aligned_merge_v1/seed0/tensor_balanced_ce/calibration_indices.npz',
            'coverage_audit': 'runs/paper_merge_v2/inputs/full_control_coverage_data/run.json',
            'calibration_reference': 'runs/paper_merge_v2/seed0/contrast_head_training_full/run.json',
            'seed': 0, 'rounds': 1, 'average_rounds': [1], 'batch_size': 16,
            'learning_rate': .01, 'tasks': source['datasets'], 'active': source['groups']}
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.06, device)
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie']
    for t in tasks:
        data.arrive(t)
        print(t, tuple(data.train[t][0].shape), flush=True)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    first, second, third = [torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks]
    history = {n: torch.zeros_like(base[n], dtype=torch.float64) for n in source['groups']}
    add_history(history, first, base, source['groups'])
    assert all(torch.equal((history[n]).float(), first[n] - base[n]) for n in history)
    add_history(history, second, base, source['groups'])
    assert all(torch.equal(history[n], (first[n] - base[n]).double() + (second[n] - base[n]).double()) for n in history)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    # Use a nonuniform previous merge so the historical residual is nonzero.
    previous = {n: (.8 * first[n] + .2 * second[n]) if n in history else base[n] for n in base}
    results = {}
    for condition in ['history_incoming_rows']:
        state, coefficients, record = train_stage(leaf, base, previous, third, history, 2,
                                                   data, tasks, plan, condition, device)
        assert record['row_gate_mean_abs'] > 0
        torch.save(state, root / f'engineering_{condition}.ckpt')
        leaf.load_state_dict(torch.load(root / f'engineering_{condition}.ckpt', map_location=device, weights_only=True), strict=True)
        results[condition] = record
        torch.cuda.empty_cache()
    write_json(destination, {'passed': True, 'native_updates': 3,
                             'training': results, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                             'scope': 'Each condition: three real native FP32 backward updates, including both 2000-sample tasks; no validation selection.'})
    print(json.dumps({'passed': True, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}), flush=True)


if __name__ == '__main__':
    main()
