"""Check sketch algebra and real-model coefficient training/export."""
import json
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf
from run_streaming_history_sketch_merge import TaskData, train_stage, write_json
from streaming_bilateral_merge import StreamingMerger as Baseline
from streaming_history_sketch_merge import StreamingMerger, collect_split, update_sketch, residual_directions, RANK
from streaming_subspace_merge import observable_linears


def algebra():
    generator = torch.Generator().manual_seed(9)
    vectors = torch.randn(3, 5, 11, generator=generator)
    base = {'w': torch.zeros(5, 11)}
    history = {'w': torch.zeros(RANK, 5, 11)}
    errors = []
    for i in range(3):
        old = history['w'].clone().transpose(0, 1)
        update_sketch(history, {'w': vectors[i]}, base, ['w'])
        appended = torch.cat([old, vectors[i, :, None, :]], 1)
        _, s, vt = torch.linalg.svd(appended, full_matrices=False)
        expected = s[:, :RANK, None] * vt[:, :RANK]
        actual = history['w'].transpose(0, 1)
        error = (actual.transpose(1, 2) @ actual - expected.transpose(1, 2) @ expected).abs().max()
        assert error < 2e-5, error
        errors.append(float(error))
    h, _ = residual_directions(history['w'], vectors[0], vectors[1], 3)
    for x in vectors[:2]:
        assert ((h * x[:, None, :]).sum(2).abs().max()) < 2e-5
    one = {'w': torch.zeros(RANK, 5, 11)}
    update_sketch(one, {'w': vectors[0]}, base, ['w'])
    residual, _ = residual_directions(one['w'], vectors[0], vectors[1], 1)
    assert torch.count_nonzero(residual) == 0
    return {'covariance_max_errors': errors, 'single_old_vector_residual_exactly_zero': True,
            'orthogonal_to_current_old_new_row_span': True}


def main():
    root = Path('runs/streaming_history_sketch_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    algebra_checks = algebra()
    source = json.loads(Path('runs/paper_merge_v2/seed0/original_fp32_tensor_balanced_ce/run.json').read_text())
    plan = json.loads(Path('runs/streaming_bilateral_v1/plan.json').read_text())
    plan.update(rounds=1, average_rounds=[1])
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie']
    for task in tasks:
        data.arrive(task)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    history = {n: torch.zeros((RANK, *base[n].shape)) for n in plan['active']}
    old = [torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks[:2]]
    for expert in old:
        update_sketch(history, expert, base, plan['active'])
    previous = dict(base)
    for n in plan['active']:
        previous[n] = .5 * old[0][n] + .5 * old[1][n]
    del old
    incoming = torch.load(source['experts'][tasks[-1]], map_location='cpu', weights_only=True)
    plan['split_layers'] = list(observable_linears(leaf, plan['active']))
    records = {}
    direction_norms = None
    for condition in ['history_sketch', 'random_sketch']:
        projections, split, audit = collect_split(leaf, base, previous, incoming, data, tasks[:2],
                                                 plan, 3, condition, device)
        candidate = StreamingMerger(leaf, base, previous, incoming, history, 2, plan['active'], condition, projections)
        norms = {n: d.reshape(RANK, d.shape[1] if d.ndim > 1 else 1, -1).norm(dim=2).cpu()
                 for n, d in candidate.sketch_directions.items()}
        if condition == 'history_sketch':
            direction_norms = norms
            baseline = Baseline(leaf, base, previous, incoming, {}, 2, plan['active'], 'old_rows', {})
            with torch.no_grad():
                wrapper = PrototypeClassifier(leaf, *data.readouts[tasks[-1]]).eval()
                x = data.train[tasks[-1]][0][:16].to(device)
                assert torch.equal(baseline.logits(wrapper, x), candidate.logits(wrapper, x))
            del baseline, wrapper, x
        else:
            assert all(torch.allclose(v, direction_norms[n], rtol=2e-5, atol=1e-8) for n, v in norms.items())
        del candidate
        state, coefficients, record = train_stage(leaf, base, previous, incoming, history, 2,
                                                 data, tasks, plan, condition, device, projections)
        assert record['sketch_gate_mean_abs'] > 0
        record['input_split'] = audit
        records[condition] = record
        torch.save(state, root / f'engineering_{condition}.ckpt')
        del projections, split, state
        torch.cuda.empty_cache()
        print(condition, record['coefficient_count'], record['sketch_gate_mean_abs'], flush=True)
    assert records['history_sketch']['coefficient_count'] == records['random_sketch']['coefficient_count']
    write_json(root / 'engineering.json', {'passed': True, 'native_updates': 6,
               'zero_added_gates_recover_bilateral_exactly': True, 'algebra': algebra_checks,
               'random_row_norm_match': True, 'records': records,
               'history_bytes': sum(x.numel() * x.element_size() for x in history.values()),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
