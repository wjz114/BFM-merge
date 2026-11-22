"""Product-invariant derivatives, real LEAF equivalence and dense export oracle."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from operator_residual_audit import audit
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_crossfit_training import train_stage as reference_train
from streaming_operator_residual import CONDITIONS, FactorSumProjection, OperatorMerger
from streaming_operator_residual_training import train_stage


def algebra():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 6, 9, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 6, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    with torch.no_grad(): y[:, 3:] *= .05
    target = torch.randn(2, 9, 8, generator=generator, dtype=torch.float64)
    f, g = FactorSumProjection.apply(x, y, 3)
    actual = f.transpose(-1, -2) @ g
    u, s, vh = torch.linalg.svd(x.transpose(-1, -2) @ y, full_matrices=False)
    expected = (u[..., :3] * s[..., None, :3]) @ vh[..., :3, :]
    projection_error = float((actual - expected).abs().max())
    a = torch.autograd.grad((actual - target).square().sum(), (x, y))
    b = torch.autograd.grad((expected - target).square().sum(), (x, y))
    gradient_error = max(float((p - q).abs().max()) for p, q in zip(a, b))
    assert projection_error < 1e-12 and gradient_error < 1e-10
    y0 = y.detach().clone(); y0[:, 3:] = 0; y0.requires_grad_()
    f, g = FactorSumProjection.apply(x.detach(), y0, 3)
    assert torch.equal(f, x.detach()[:, :3]) and torch.equal(g, y0.detach()[:, :3])
    gy, = torch.autograd.grad((f.transpose(-1, -2) @ g - target).square().sum(), y0)
    direction = torch.randn(y0.shape, generator=generator, dtype=torch.float64)
    values = []
    for sign in [-1, 1]:
        f, g = FactorSumProjection.apply(x.detach(), y0.detach() + sign * 1e-6 * direction, 3)
        values.append((f.transpose(-1, -2) @ g - target).square().sum())
    fd_error = float(((values[1] - values[0]) / 2e-6 - (gy * direction).sum()).abs())
    assert fd_error < 1e-6
    return dict(projection_error=projection_error, gradient_error=gradient_error, zero_rank_directional_fd_error=fd_error)


def main():
    root = Path('runs/streaming_operator_residual_v1')
    path = root / 'engineering_plan.json'
    plan = json.loads(path.read_text())
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    strict_settings(); device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2, device)
    toy = algebra()
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    for t in plan['pilot_tasks']: data.arrive(t)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    previous = torch.load(plan['previous'], map_location='cpu', weights_only=True)
    incoming = torch.load(source['experts'][plan['pilot_tasks'][-1]], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    started = time.monotonic()
    gradients, logits, gate_gradient_norm = {}, {}, None
    task = plan['pilot_tasks'][-1]
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    batch, labels = data.train[task][0][:4].to(device), data.train[task][1][:4].to(device)
    for condition in ['operator_zero', 'operator_covariance']:
        m = OperatorMerger(leaf, base, previous, incoming, 3, plan['active'], condition, plan['seed'])
        prediction = m.logits(wrapper, batch)
        logits[condition] = prediction.detach()
        loss = (F.cross_entropy(prediction, labels, reduction='none') * data.weights[task][labels]).mean()
        g, = torch.autograd.grad(loss, m.coefficients)
        gradients[condition] = g[:347666].detach()
        if condition == 'operator_covariance': gate_gradient_norm = float(g[347666:].norm())
        del m, prediction, loss
    assert torch.equal(logits['operator_zero'], logits['operator_covariance'])
    gradient_relative = float((gradients['operator_zero'] - gradients['operator_covariance']).norm()
        / gradients['operator_zero'].norm().clamp_min(1e-30))
    assert gradient_relative < 1e-4 and gate_gradient_norm > 0
    records, states, audits = {}, {}, {}
    for condition in CONDITIONS:
        state, payload, record = train_stage(leaf, base, previous, incoming, 3, data,
            plan['pilot_tasks'], plan, condition, device)
        audits[condition] = audit(base, previous, incoming, state, payload, plan['active'], 3, condition)
        records[condition] = record; states[condition] = state, payload
    reference, _, rp, rr = reference_train(leaf, base, previous, incoming, [], [], 3, data,
        plan['pilot_tasks'], {}, plan, 'single_full', device)
    assert all(torch.equal(reference[n], states['operator_zero'][0][n]) for n in reference)
    assert torch.equal(rp['snapshots'], states['operator_zero'][1]['snapshots'][:, :347666])
    assert all(r['schedule_hash'] == rr['schedule_hash'] for r in records.values())
    result = dict(status='passed', plan_sha256=sha256(path), algebra=toy, audits=audits,
        zero_gate_logits_bitwise=True, zero_control_trajectory_bitwise=True,
        zero_gate_native_gradient_relative_error=gradient_relative, initial_gate_gradient_norm=gate_gradient_norm,
        records=records, coefficient_updates=16, training_examples=64,
        extra_gradient_probe_examples=8, official_test_read=False,
        elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
