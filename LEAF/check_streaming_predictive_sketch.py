"""Exact predictive-score estimator check and a training-only LEAF stream check."""
import gc
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf
from run_streaming_history_sketch_merge import TaskData, write_json
from run_streaming_predictive_sketch import train_stage
from streaming_history_sketch_merge import RANK, StreamingMerger as OriginalMerger
from streaming_predictive_sketch import StreamingMerger, observe_predictive, predictive_probe, update_sketch


def algebra():
    generator = torch.Generator().manual_seed(71509)
    x = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    dictionary = torch.randn(3, 12, generator=generator, dtype=torch.float64)
    sample_weights = torch.tensor([.7, 1.3], dtype=torch.float64)
    probability = (x @ weight.detach().T).softmax(-1)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(2):
        for c in range(3):
            logit_score = torch.eye(3, dtype=torch.float64)[c] - probability[i]
            gradient = (logit_score[:, None] * x[i][None, :]).flatten()
            response = dictionary @ gradient
            expected += sample_weights[i] * probability[i, c] / 2 * torch.outer(response, response)
    actual = torch.zeros_like(expected)
    for pattern in itertools.product((-1., 1.), repeat=6):
        signs = torch.tensor(pattern, dtype=torch.float64).reshape(2, 3)
        score = predictive_probe(x @ weight.T, signs, sample_weights)
        gradient, = torch.autograd.grad(score, weight)
        response = dictionary @ gradient.flatten()
        actual += torch.outer(response, response) / 64
    error = float((actual - expected).abs().max())
    assert error < 1e-12, error
    return {'all_64_rademacher_patterns_enumerated': True,
            'independent_weighted_predictive_fisher_max_abs_error': error,
            'nonuniform_sample_weights': sample_weights.tolist(),
            'torch_dtype': 'float64'}


def main():
    root = Path('runs/streaming_predictive_sketch_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    started = time.monotonic()
    algebra_record = algebra()
    plan = json.loads(Path('runs/streaming_history_sketch_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4, predictive_probe_seed_base=59333)
    source = json.loads(Path(plan['source_fit']).read_text())
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {task: torch.load(source['experts'][task], map_location='cpu', weights_only=True) for task in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    update_sketch(history, experts[tasks[0]], base, plan['active'], 'energy_history')
    plain = OriginalMerger(leaf, base, experts[tasks[0]], experts[tasks[1]], history, 1,
                           plan['active'], 'history_sketch', {})
    observed = StreamingMerger(leaf, base, experts[tasks[0]], experts[tasks[1]], history, 1,
                               plan['active'], 'energy_history', {})
    wrapper = PrototypeClassifier(leaf, *data.readouts[tasks[0]]).eval()
    x, y = data.train[tasks[0]]
    gradients, logits = [], []
    for merger in (plain, observed):
        parameters = merger.compose()
        output = merger.logits(wrapper, x[:4].to(device), parameters)
        if merger is observed:
            signs = torch.randint(0, 2, output.shape, device=device, dtype=torch.int8,
                                  generator=torch.Generator(device=device).manual_seed(59333)) * 2 - 1
            observe_predictive(merger, parameters, output, signs,
                               data.weights[tasks[0]][y[:4].to(device)])
        truth = y[:4].to(device)
        loss = (F.cross_entropy(output, truth, reduction='none') * data.weights[tasks[0]][truth]).mean()
        loss.backward()
        gradients.append(merger.coefficients.grad.detach().clone())
        logits.append(output.detach().clone())
    logit_error = float((logits[0] - logits[1]).abs().max())
    gradient_error = float((gradients[0] - gradients[1]).abs().max())
    assert logit_error <= 1e-6 and gradient_error <= 1e-6
    del plain, observed, wrapper, gradients, logits, loss, output, merger, history, parameters, signs
    gc.collect()
    records = {}
    for condition in ('energy_history', 'predictive_history'):
        history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
        previous = experts[tasks[0]]
        update_sketch(history, previous, base, plan['active'], condition)
        records[condition] = []
        for count in range(1, len(tasks)):
            state, coefficients, record, statistics = train_stage(
                leaf, base, previous, experts[tasks[count]], history, count,
                data, tasks[:count + 1], plan, condition, device, {})
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            record['history_sketch_update'] = update_sketch(history, experts[tasks[count]], base,
                                                           plan['active'], condition, statistics)
            assert all(torch.isfinite(value).all() for value in history.values())
            torch.save(state, root / f'engineering_{condition}_stage{count + 1}.ckpt')
            records[condition].append(record)
            write_json(root / 'engineering_progress.json', {'records': records})
            previous = state
            del coefficients, statistics
            gc.collect()
            torch.cuda.empty_cache()
        torch.save(history, root / f'engineering_{condition}_history.pt')
        print(condition, 'complete', flush=True)
        del previous, state, history
    assert all(records['energy_history'][i]['schedule_hash'] == records['predictive_history'][i]['schedule_hash']
               for i in range(3))
    assert all(records['energy_history'][i]['probe_signs_hash'] == records['predictive_history'][i]['probe_signs_hash']
               for i in range(3))
    updates = sum(record['updates'] for run in records.values() for record in run)
    assert updates == 36 and not data.valid
    write_json(root / 'engineering.json', {
        'passed': True, 'algebra': algebra_record, 'coefficient_optimizer_updates': updates,
        'training_examples': updates * 4, 'purity_probe_ce_backward_passes': 2, 'purity_probe_score_vjps': 1,
        'training_score_vjps': updates, 'training_score_vjp_example_passes': updates * 4,
        'purity_probe_score_vjp_example_passes': 4,
        'purity_probe_probe_example_passes': 8,
        'purity_probe_max_abs_logits': logit_error, 'purity_probe_max_abs_coefficient_gradient': gradient_error,
        'fixed_coefficient_means_exact': True, 'paired_schedules': True, 'paired_probe_signs': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
