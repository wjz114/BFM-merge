"""Check tangent centering algebra and independently contract native gradients."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256, PrototypeClassifier
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger
from streaming_crossfit_training import train_stage as reference_train
from streaming_expert_centering import CONDITIONS, source_gradient, train_stage


def toy_algebra():
    x = torch.tensor([[-.3, 1.1], [1.2, .2], [.7, -.6]], dtype=torch.float64)
    y = torch.tensor([0, 1, 0])
    source = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    student = torch.tensor([-.1, .7], dtype=torch.float64, requires_grad=True)
    result = {}
    for nonlinear in [False, True]:
        def logits(p):
            a = x @ p
            if nonlinear:
                a = a + torch.sin(p[0]) * x[:, 0].square()
            return torch.stack([a, torch.zeros_like(a)], dim=-1)
        target = logits(source).detach()
        ge, = torch.autograd.grad(F.cross_entropy(logits(source), y), source)
        gs, = torch.autograd.grad(F.cross_entropy(logits(student), y), student)
        gkl, = torch.autograd.grad(F.kl_div(logits(student).log_softmax(-1),
            target.log_softmax(-1), reduction='batchmean', log_target=True), student)
        jt = torch.autograd.functional.jacobian(logits, source)
        js = torch.autograd.functional.jacobian(logits, student)
        residual = target.softmax(-1) - F.one_hot(y, 2)
        correction = ((js - jt) * residual[:, :, None]).sum((0, 1)) / len(y)
        assert torch.allclose(gs - ge - gkl, correction, atol=1e-12, rtol=1e-12)
        error = float((gs - ge - gkl).norm())
        assert error > 1e-3 if nonlinear else error < 1e-12
        def tilted(p):
            return F.cross_entropy(logits(p), y) - torch.dot(ge, p)
        finite = torch.stack([(tilted(student.detach() + 1e-6 * torch.eye(2, dtype=torch.float64)[j]) -
            tilted(student.detach() - 1e-6 * torch.eye(2, dtype=torch.float64)[j])) / 2e-6 for j in range(2)])
        assert torch.allclose(finite, gs - ge, atol=1e-9, rtol=1e-8)
        result['nonlinear' if nonlinear else 'affine'] = dict(centered_minus_KL_gradient_norm=error,
            identity_error=float((gs - ge - gkl - correction).abs().max()),
            finite_difference_error=float((finite - gs + ge).abs().max()))
    return result


def main():
    root = Path('runs/streaming_expert_centering_v1')
    path = root / 'engineering_plan.json'
    plan = json.loads(path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    toy = toy_algebra()
    strict_settings()
    device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    data.experts = {}
    for task in plan['pilot_tasks']:
        data.arrive(task)
        source_path = Path(source['experts'][task])
        assert sha256(source_path) == plan['expert_sha256'][str(source_path)]
        incoming = torch.load(source_path, map_location='cpu', weights_only=True)
        data.experts[task] = {n: incoming[n].to(device) for n in plan['active']}
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    previous = torch.load(plan['previous'], map_location='cpu', weights_only=True)
    started = time.monotonic()
    merger = StreamingMerger(leaf, base, previous, incoming, None, 3, plan['active'], 'old_rows', None)
    task = plan['pilot_tasks'][0]
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    batch, labels = data.train[task][0][:4].to(device), data.train[task][1][:4].to(device)
    target, _, projected = source_gradient(merger, wrapper, batch, labels, data.weights[task], data.experts[task])
    params = merger.anchor.copy()
    for n in plan['active']:
        params[n] = data.experts[task][n].detach().clone().requires_grad_(True)
    native_logits = merger.logits(wrapper, batch, params)
    assert torch.equal(target, native_logits.detach())
    loss = (F.cross_entropy(native_logits, labels, reduction='none') * data.weights[task][labels]).mean()
    native_gradients = torch.autograd.grad(loss, [params[n] for n in plan['active']], allow_unused=True)
    old_scalars, new_scalars, new_rows, old_rows = [], [], [], []
    for n, g in zip(plan['active'], native_gradients):
        g = torch.zeros_like(params[n]) if g is None else g
        old, new = merger.directions[n][:2]
        rows = g.shape[0] if g.ndim else 1
        a, b = (g * old).reshape(rows, -1).sum(1), (g * new).reshape(rows, -1).sum(1)
        old_scalars.append(a.sum());new_scalars.append(b.sum())
        new_rows.append(b);old_rows.append(a)
    oracle = torch.cat([torch.stack(old_scalars), torch.stack(new_scalars), *new_rows, *old_rows])
    absolute = float((oracle - projected).abs().max())
    relative = float((oracle - projected).norm() / oracle.norm().clamp_min(1e-30))
    assert relative < 1e-5 and absolute < 1e-6
    del merger, params, native_gradients, native_logits, loss
    records, states, maximum = {}, {}, 0.
    for condition in CONDITIONS:
        state, payload, record = train_stage(leaf, base, previous, incoming, 3, data,
            plan['pilot_tasks'], plan, condition, device)
        maximum = max(maximum, audit_export(base, previous, incoming, state, payload['mean'], plan['active']))
        records[condition] = record
        states[condition] = state, payload
    reference, _, rp, rr = reference_train(leaf, base, previous, incoming, [], [], 3, data,
        plan['pilot_tasks'], {}, plan, 'single_full', device)
    assert all(torch.equal(reference[n], states['balanced_ce'][0][n]) for n in reference)
    assert torch.equal(rp['snapshots'], states['balanced_ce'][1]['snapshots'])
    assert all(r['schedule_hash'] == rr['schedule_hash'] for r in records.values())
    assert any(not torch.equal(states['expert_centered'][0][n], states['output_kl'][0][n]) for n in plan['active'])
    result = dict(status='passed', plan_sha256=sha256(path), toy=toy,
        source_logits_exact=True, native_gradient_pullback_max_abs_error=absolute,
        native_gradient_pullback_relative_error=relative, unchanged_CE_trajectory_bitwise=True,
        independent_export_max_abs_error=maximum, records=records, optimizer_updates=32,
        student_training_examples=128, source_gradient_examples=104, official_test_read=False,
        elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
