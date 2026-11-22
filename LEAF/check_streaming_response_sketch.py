"""Independent response-space algebra and a training-only LEAF stream check."""
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf
from run_streaming_history_sketch_merge import TaskData, write_json
from run_streaming_response_sketch import train_stage
from streaming_history_sketch_merge import RANK, StreamingMerger as OriginalMerger
from streaming_response_sketch import StreamingMerger, select_rows, update_sketch


def algebra():
    generator = torch.Generator().manual_seed(71509)
    b = torch.randn(8, 3, 13, generator=generator)
    g = torch.randn(8, 13, 17, generator=generator)
    response = b.double() @ g.double()
    covariance = response @ response.transpose(-1, -2)
    kept, record = select_rows(b, covariance)
    u, _, _ = torch.linalg.svd(response)
    reference = u[:, :, :2].transpose(-1, -2) @ b.double()
    gram = lambda value: value.double().transpose(-1, -2) @ value.double()
    error = float((gram(kept) - gram(reference)).abs().max())
    assert error < 1e-5
    orthogonal, _ = torch.linalg.qr(torch.randn(8, 3, 3, generator=generator, dtype=torch.float64))
    rotated, _ = select_rows((orthogonal @ b.double()).float(), orthogonal @ covariance @ orthogonal.transpose(-1, -2))
    rotation_error = float((gram(kept) - gram(rotated)).abs().max())
    assert rotation_error < 1e-5
    zero, _ = select_rows(b, torch.zeros_like(covariance))
    u0, _, _ = torch.linalg.svd(b.double())
    reference0 = u0[:, :, :2].transpose(-1, -2) @ b.double()
    zero_error = float((gram(zero) - gram(reference0)).abs().max())
    assert zero_error < 1e-5
    anisotropic = torch.diag(torch.tensor([10., 2., 1.])).unsqueeze(0)
    probes = torch.diag(torch.tensor([.01, 1., 10.])).unsqueeze(0)
    r = anisotropic @ probes
    result, _ = select_rows(anisotropic, r @ r.transpose(-1, -2))
    assert torch.equal(gram(result), torch.diag(torch.tensor([0., 4., 1.], dtype=torch.float64)).unsqueeze(0))
    return {'independent_response_SVD_covariance_error': error,
            'latent_rotation_covariance_error': rotation_error,
            'zero_response_energy_SVD_error': zero_error,
            'anisotropic_low_parameter_energy_direction_retained': True,
            'response_energy_example': record}


def main():
    root = Path('runs/streaming_response_sketch_v1')
    root.mkdir(exist_ok=True)
    assert not (root / 'engineering.json').exists()
    started = time.monotonic()
    algebra_record = algebra()
    plan = json.loads(Path('runs/streaming_history_sketch_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4)
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
    observed.responses.enabled = True
    wrapper = PrototypeClassifier(leaf, *data.readouts[tasks[0]]).eval()
    x, y = data.train[tasks[0]]
    gradients, logits = [], []
    for merger in (plain, observed):
        output = merger.logits(wrapper, x[:4].to(device))
        truth = y[:4].to(device)
        loss = (F.cross_entropy(output, truth, reduction='none') * data.weights[tasks[0]][truth]).mean()
        loss.backward()
        gradients.append(merger.coefficients.grad.detach().clone())
        logits.append(output.detach().clone())
    logit_error = float((logits[0] - logits[1]).abs().max())
    gradient_error = float((gradients[0] - gradients[1]).abs().max())
    assert logit_error <= 1e-6 and gradient_error <= 1e-6
    del plain, observed, wrapper, gradients, logits, loss, output, merger, history
    gc.collect()
    records = {}
    for condition in ('energy_history', 'response_history'):
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
    assert all(records['energy_history'][i]['schedule_hash'] == records['response_history'][i]['schedule_hash']
               for i in range(3))
    updates = sum(record['updates'] for run in records.values() for record in run)
    assert updates == 36 and not data.valid
    write_json(root / 'engineering.json', {
        'passed': True, 'algebra': algebra_record, 'coefficient_optimizer_updates': updates,
        'training_examples': updates * 4, 'hook_purity_probe_backward_passes': 2,
        'hook_purity_probe_example_passes': 8,
        'hook_purity_max_abs_logits': logit_error, 'hook_purity_max_abs_coefficient_gradient': gradient_error,
        'fixed_coefficient_means_exact': True, 'paired_schedules': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})


if __name__ == '__main__':
    main()
