"""Check the predictive-Fisher estimator, moment algebra, and native exports."""
import json
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf
from fisher_moment_merge import fisher_probe, estimate_fisher, add_moments, moment_centers, build_merger
from run_fisher_moment_merge import TaskData, train_stage, write_json


def algebra():
    logits = torch.tensor([[.2, -.7, .5], [-.2, .8, -.1]], dtype=torch.float64, requires_grad=True)
    signs = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(3):
        signs = torch.cat([torch.cat([signs, signs], 1), torch.cat([signs, -signs], 1)], 0)
    diagonal = torch.zeros_like(logits)
    for row in signs:
        probe = fisher_probe(logits, row[:6].reshape_as(logits))
        gradient, = torch.autograd.grad(probe, logits)
        diagonal += gradient.square() / len(signs)
    probability = logits.detach().softmax(-1)
    expected = probability * (1 - probability) / logits.shape[0]
    error = float((diagonal - expected).abs().max())
    assert error < 1e-12
    base = {'w': torch.zeros(2, 3)}
    experts = [{'w': torch.tensor([[1., 2., 3.], [4., 5., 6.]])},
               {'w': torch.tensor([[3., 4., 7.], [2., 1., 0.]])}]
    weights = [{'w': torch.tensor([[1., 2., 0.], [1., 3., 0.]])},
               {'w': torch.tensor([[3., 2., 0.], [2., 1., 0.]])}]
    results = []
    for order in ([0, 1], [1, 0]):
        state = {'w': torch.zeros(3, 2, 3, dtype=torch.float64)}
        for i in order:
            add_moments(state, weights[i], experts[i], base, ['w'], i, 0)
        results.append(moment_centers(state, 2))
    expected_center = torch.tensor([[2.5, 3., 5.], [8 / 3, 4., 3.]])
    assert torch.allclose(results[0][1]['w'], expected_center)
    assert torch.equal(results[0][1]['w'], results[1][1]['w'])
    return {'predictive_fisher_diagonal_max_error': error, 'weighted_mean_and_zero_precision_fallback': True,
            'two_expert_moment_order_invariance': True}


def main():
    root = Path('runs/fisher_moment_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    checks = algebra()
    plan = json.loads(Path('runs/streaming_bilateral_v1/plan.json').read_text())
    plan.update(rounds=1, average_rounds=[1])
    source = json.loads(Path(plan['source_fit']).read_text())
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie']
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    moments = {c: {n: torch.zeros((3, *base[n].shape), dtype=torch.float64) for n in plan['active']}
               for c in ['moment_restart', 'shuffled_restart']}
    fisher_audits = {}
    previous = None
    for i, task in enumerate(tasks):
        data.arrive(task)
        incoming = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        leaf.load_state_dict(incoming, strict=True)
        fisher, audit = estimate_fisher(leaf, data, task, plan, device, examples=32)
        fisher_audits[task] = audit
        for condition in moments:
            add_moments(moments[condition], fisher, incoming, base, plan['active'],
                        plan['tasks'].index(task), plan['seed'], condition == 'shuffled_restart')
        if i == 0:
            previous = incoming
        del fisher
    records = {}
    for condition in ['incoming_rows', 'moment_restart', 'shuffled_restart']:
        history = moments['moment_restart' if condition == 'incoming_rows' else condition]
        if condition != 'incoming_rows':
            merger = build_merger(leaf, base, previous, incoming, history, 2, plan['active'], condition, {})
            uniform, _ = moment_centers(history, 3)
            state = dict(base)
            state.update({n: base[n] + uniform[n] for n in plan['active']})
            with torch.no_grad():
                wrapper = PrototypeClassifier(leaf, *data.readouts[tasks[-1]]).eval()
                x = data.train[tasks[-1]][0][:16].to(device)
                functional = merger.logits(wrapper, x)
                leaf.load_state_dict(state, strict=True)
                assert torch.equal(functional, wrapper(x))
            del merger, uniform, state, wrapper, x
        state, coefficients, record = train_stage(leaf, base, previous, incoming, history, 2,
                                                 data, tasks, plan, condition, device, {})
        assert record['row_gate_mean_abs'] > 0
        records[condition] = record
        torch.save(state, root / f'engineering_{condition}.ckpt')
        print(condition, record['coefficient_count'], record['training_seconds'], flush=True)
        del state, coefficients
        torch.cuda.empty_cache()
    assert records['moment_restart']['coefficient_count'] == records['shuffled_restart']['coefficient_count']
    write_json(root / 'engineering.json', {'passed': True, 'algebra': checks, 'records': records,
               'native_coefficient_updates': 9, 'fisher_backward_probes': 6, 'fisher_forward_examples': 96,
               'fisher_audits': fisher_audits, 'restart_initialization_matches_uniform_native_exactly': True,
               'moment_state_bytes': sum(v.numel() * v.element_size() for v in moments['moment_restart'].values()),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
