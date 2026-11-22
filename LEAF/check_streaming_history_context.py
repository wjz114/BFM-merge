"""Check source refinement, history perturbations and ordinary native exports."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_crossfit_training import train_stage
from streaming_history_context import CONDITIONS, adapt_stage, flatten, from_flat, historical_offset, isotropic_offset


def adam_equivalence():
    alpha = .25
    old = torch.tensor([.2, -.3], dtype=torch.float64)
    expert = torch.nn.Parameter(torch.tensor([.1, .6], dtype=torch.float64))
    merged = torch.nn.Parameter((alpha * expert.detach() + (1 - alpha) * old).clone())
    first = torch.optim.Adam([expert], lr=.01, eps=1e-8)
    second = torch.optim.Adam([merged], lr=alpha * .01, eps=1e-8 / alpha)
    maximum = 0.
    for step in range(8):
        first.zero_grad(); second.zero_grad()
        blend = alpha * expert + (1 - alpha) * old
        (blend.sin().square().sum() + .1 * blend.sum()).backward()
        (merged.sin().square().sum() + .1 * merged.sum()).backward()
        first.step(); second.step()
        maximum = max(maximum, float((alpha * expert.detach() + (1 - alpha) * old - merged.detach()).abs().max()))
    assert maximum < 1e-14
    return maximum


def main():
    root = Path('runs/streaming_history_context_v1')
    path = root / 'engineering_plan.json'
    plan = json.loads(path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    strict_settings(); device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    experts = []
    for task in plan['pilot_tasks']:
        data.arrive(task)
        p = Path(source['experts'][task])
        assert sha256(p) == plan['expert_sha256'][str(p)]
        experts.append(torch.load(p, map_location='cpu', weights_only=True))
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    previous = torch.load(plan['previous'], map_location='cpu', weights_only=True)
    history = [{n: s[n].to(device) for n in plan['active']} for s in experts[:-1]]
    started = time.monotonic()
    equivalence = adam_equivalence()
    offset = historical_offset(history, plan['active'], 1)
    maximum_offset_error = 0.
    for n in plan['active']:
        full = torch.stack([s[n].double() for s in history]).mean(0)
        remaining = torch.stack([s[n].double() for j, s in enumerate(history) if j != 1]).mean(0)
        oracle = (len(history) / (len(history) + 1) * (remaining - full)).float()
        maximum_offset_error = max(maximum_offset_error, float((offset[n] - oracle).abs().max()))
    assert maximum_offset_error < 2e-7
    iso, norm_error = isotropic_offset(offset, 1729)
    history_flat, iso_flat = flatten(offset, plan['active']), flatten(iso, plan['active'])
    assert history_flat.norm() > 0 and not torch.equal(history_flat, iso_flat)
    del offset, iso, history_flat, iso_flat
    records, refined, export_maximum = {}, {}, 0.
    for condition in CONDITIONS:
        if condition == 'native_after':
            initial, _, coefficient_payload, merge_record = train_stage(leaf, base, previous, experts[-1],
                [], [], 3, data, plan['pilot_tasks'], {}, plan, 'single_full', device)
            export_maximum = max(export_maximum, audit_export(base, previous, experts[-1], initial,
                coefficient_payload['mean'], plan['active']))
            state, payload, record = adapt_stage(leaf, initial, previous, history, data,
                plan['pilot_tasks'], plan, condition, device)
        else:
            state, payload, record = adapt_stage(leaf, experts[-1], previous, history, data,
                plan['pilot_tasks'], plan, condition, device)
            native, _, coefficient_payload, merge_record = train_stage(leaf, base, previous, state,
                [], [], 3, data, plan['pilot_tasks'], {}, plan, 'single_full', device)
            export_maximum = max(export_maximum, audit_export(base, previous, state, native,
                coefficient_payload['mean'], plan['active']))
        reconstructed = from_flat((payload['average_sum'] / payload['average_count']).float(), base, plan['active'])
        assert all(torch.equal(state[n], reconstructed[n]) for n in state)
        records[condition] = dict(adaptation=record, coefficient_updates=merge_record['updates'])
        refined[condition] = state
    source_schedules = [records[c]['adaptation']['schedule_hash'] for c in CONDITIONS if c != 'native_after']
    assert len(set(source_schedules)) == 1
    assert any(not torch.equal(refined['history_context'][n], refined['fixed_context'][n]) for n in plan['active'])
    assert any(not torch.equal(refined['history_context'][n], refined['isotropic_context'][n]) for n in plan['active'])
    # Independent plain native CE replay of the new-source control.
    leaf.load_state_dict(experts[-1], strict=True)
    params = dict(leaf.named_parameters())
    for n, p in params.items():
        p.requires_grad_(n in plan['active']); p.grad = None
    optimizer = torch.optim.Adam([params[n] for n in plan['active']], lr=plan['adapt_learning_rate'])
    task = plan['pilot_tasks'][-1]
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    x, y = data.train[task]
    for row in records['source_ce']['adaptation']['schedule']:
        ids = row[1:]; labels = y[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(wrapper(x[ids].to(device)), labels, reduction='none') * data.weights[task][labels]).mean()
        loss.backward(); optimizer.step()
    assert all(torch.equal(params[n].detach().cpu(), refined['source_ce'][n]) for n in plan['active'])
    leaf.requires_grad_(False)
    stage2 = {}
    for condition in ('fixed_context', 'history_context', 'isotropic_context'):
        state, _, record = adapt_stage(leaf, experts[1], experts[0], history[:1], data,
            plan['pilot_tasks'][:2], plan, condition, device)
        stage2[condition] = state
        assert all(c['historical_norm'] == 0 for c in record['contexts'])
    assert all(torch.equal(stage2['fixed_context'][n], stage2[c][n]) for c in stage2 for n in base)
    result = dict(status='passed', plan_sha256=sha256(path), adam_equivalence_error=equivalence,
        historical_offset_oracle_max_error=maximum_offset_error, isotropic_tensor_norm_relative_error=norm_error,
        source_CE_independent_replay_bitwise=True, one_history_context_arms_bitwise=True,
        all_native_averages_reconstructed=True, independent_coefficient_export_max_error=export_maximum,
        records=records, optimizer_updates=112, native_updates=72, coefficient_updates=40,
        training_examples=448, official_test_read=False, elapsed_seconds=time.monotonic() - started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
