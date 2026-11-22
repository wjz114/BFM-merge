"""Real-input collection, nested-baseline identity, gradients and export checks."""
import json
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf
from run_streaming_innovation_merge import TaskData, train_stage, write_json
from streaming_residual_merge_v2 import StreamingMerger as Baseline
from streaming_subspace_merge import observable_linears
from streaming_innovation_merge import StreamingMerger, collect_split


def main():
    root = Path('runs/streaming_innovation_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    source = json.loads(Path('runs/paper_merge_v2/seed0/original_fp32_tensor_balanced_ce/run.json').read_text())
    plan = json.loads(Path('runs/streaming_merge_v2/plan.json').read_text())
    plan.update(rounds=1, average_rounds=[1], subspace_rank=16, svd_oversampling=8,
                svd_power_iterations=2, subspace_examples_per_task=16, tokens_per_task_layer=128)
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie']
    for task in tasks:
        data.arrive(task)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    previous = torch.load(source['experts'][tasks[0]], map_location='cpu', weights_only=True)
    incoming = torch.load(source['experts'][tasks[-1]], map_location='cpu', weights_only=True)
    plan['split_layers'] = list(observable_linears(leaf, plan['active']))
    records = {}
    # Two observed old tasks exercise text-dependent and EEG-dependent inputs.
    for condition in ['innovation_rows', 'innovation_split', 'innovation_random']:
        projections, split, audit = collect_split(leaf, base, previous, incoming, data, tasks[:2],
                                                 plan, 3, condition, device)
        if condition == 'innovation_split':
            baseline = Baseline(leaf, base, previous, incoming, {}, 2, plan['active'], 'incoming_rows')
            candidate = StreamingMerger(leaf, base, previous, incoming, {}, 2, plan['active'], condition, projections)
            with torch.no_grad():
                wrapper = PrototypeClassifier(leaf, *data.readouts[tasks[-1]]).eval()
                x = data.train[tasks[-1]][0][:16].to(device)
                error = (baseline.logits(wrapper, x) - candidate.logits(wrapper, x)).abs().max().item()
                assert error < 1e-5, error
                nested = StreamingMerger(leaf, base, previous, incoming, {}, 2, plan['active'], 'innovation_rows', projections)
                assert torch.equal(nested.logits(wrapper, x), candidate.logits(wrapper, x))
                del nested
            del baseline, candidate, wrapper, x
        state, coefficients, record = train_stage(leaf, base, previous, incoming, {}, 2,
                                                 data, tasks, plan, condition, device, projections)
        if condition != 'innovation_rows':
            assert record['split_gate_mean_abs'] > 0
        record['input_split'] = audit
        records[condition] = record
        torch.save(state, root / f'engineering_{condition}.ckpt')
        del projections, split, state
        torch.cuda.empty_cache()
        print(condition, record['coefficient_count'], audit['seconds'], flush=True)
    assert records['innovation_split']['coefficient_count'] == records['innovation_random']['coefficient_count']
    write_json(root / 'engineering.json', {'passed': True, 'native_updates': 9,
               'zero_split_gates_recover_innovation_rows_exactly': True, 'initial_logit_difference_from_incoming_rows': error, 'records': records,
               'split_layers': plan['split_layers'], 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
