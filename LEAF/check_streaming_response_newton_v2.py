"""Independent dense GGN solve, real Jacobian contraction and baseline replay."""
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
from streaming_expert_centering import train_stage as reference_train
from streaming_response_newton_v2 import (CONDITIONS, contrast_jacobian, dual_ggn,
                                      helmert, objective, source_logits, train_stage)


def toy_checks():
    torch.manual_seed(917)
    z = torch.randn(3, 4, dtype=torch.float64)
    teacher = torch.randn_like(z)
    labels = torch.tensor([0, 2, 3])
    weights = torch.tensor([.7, 1., 1.2, 1.3], dtype=torch.float64)
    j = torch.randn(3, 3, 7, dtype=torch.float64)
    h = helmert(4, z.device)
    assert torch.allclose(h.T @ h, torch.eye(3, dtype=torch.float64), atol=1e-14)
    assert h.sum(0).abs().max() < 1e-14
    result = {}
    for kl in (False, True):
        c = torch.zeros(7, dtype=torch.float64, requires_grad=True)
        def loss(v):
            logits = z + torch.einsum('bkd,d,ck->bc', j, v, h)
            return objective(logits, teacher, labels, weights, kl, True)
        gradient, = torch.autograd.grad(loss(c), c)
        hessian = torch.autograd.functional.hessian(loss, c)
        direction, info = dual_ggn(j, z, teacher, labels, weights, kl, 1.)
        oracle = torch.linalg.solve(hessian + info['damping'] * torch.eye(7, dtype=torch.float64), -gradient)
        error = float((oracle - direction).abs().max())
        assert error < 1e-12 and float(gradient @ direction) < 0
        result['kl' if kl else 'ce'] = dict(max_dense_solve_error=error, **info)
    return result


def main():
    root = Path('runs/streaming_response_newton_v1')
    path = root / 'engineering_plan_v2.json'
    plan = json.loads(path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    toy = toy_checks()
    strict_settings();device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device);data.experts = {}
    for task in plan['pilot_tasks']:
        data.arrive(task)
        p = Path(source['experts'][task])
        assert sha256(p) == plan['expert_sha256'][str(p)]
        incoming = torch.load(p, map_location='cpu', weights_only=True)
        data.experts[task] = {n: incoming[n].to(device) for n in plan['active']}
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    previous = torch.load(plan['previous'], map_location='cpu', weights_only=True)
    started = time.monotonic()
    merger = StreamingMerger(leaf, base, previous, incoming, None, 3, plan['active'], 'old_rows', None)
    task = 'EMO_FACED'
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    x, y = data.train[task]
    batch, labels = x[:4].to(device), y[:4].to(device)
    target = source_logits(merger, wrapper, batch, data.experts[task], labels, data.weights[task])
    z = merger.logits(wrapper, batch)
    gradient, = torch.autograd.grad(objective(z, target, labels, data.weights[task], True, True),
                                   merger.coefficients, retain_graph=True)
    jacobian, h = contrast_jacobian(z, merger.coefficients)
    p, q = z.detach().double().softmax(-1), target.double().softmax(-1)
    output_grad = ((p - q) @ h.double()) * data.weights[task][labels, None].double() / len(labels)
    contracted = (jacobian.double() * output_grad[:, :, None]).sum((0, 1))
    relative = float((contracted - gradient).norm() / gradient.norm())
    assert relative < 2e-5
    _, solve_info = dual_ggn(jacobian, z.detach(), target, labels, data.weights[task], True, 1.)
    del merger, jacobian, z, gradient, contracted
    records, states, maximum = {}, {}, 0.
    for condition in CONDITIONS:
        state, payload, record = train_stage(leaf, base, previous, incoming, 3, data,
            plan['pilot_tasks'], plan, condition, device)
        maximum = max(maximum, audit_export(base, previous, incoming, state, payload['mean'], plan['active']))
        records[condition], states[condition] = record, (state, payload)
    baseline_checks = {}
    for condition, reference_condition in [('adam_ce', 'balanced_ce'), ('adam_kl', 'output_kl')]:
        state, payload, record = reference_train(leaf, base, previous, incoming, 3, data,
            plan['pilot_tasks'], plan, reference_condition, device)
        assert all(torch.equal(state[n], states[condition][0][n]) for n in state)
        assert torch.equal(payload['snapshots'], states[condition][1]['snapshots'])
        assert record['schedule_hash'] == records[condition]['schedule_hash']
        baseline_checks[condition] = True
    assert len({r['schedule_hash'] for r in records.values()}) == 1
    assert all(records[c]['accepted_updates'] > 0 for c in ('ggn_ce', 'ggn_kl'))
    result = dict(status='passed', plan_sha256=sha256(path), toy=toy,
        real_kl_jacobian_contraction_relative_error=relative, real_solve=solve_info,
        adam_baselines_bitwise=baseline_checks, independent_export_max_abs_error=maximum,
        records=records, engineering_update_opportunities=48, engineering_student_examples=192,
        engineering_teacher_examples=192, extra_jacobian_probe_examples=4,
        official_test_read=False, elapsed_seconds=time.monotonic() - started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering_v2.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
