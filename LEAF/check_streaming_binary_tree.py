"""Real fitting equivalence, event matching and causal cache checks before freezing."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_binary_tree import events, live_intervals, train_event
from streaming_crossfit_training import train_stage as reference_train


def main():
    root = Path('runs/streaming_binary_tree_v1')
    path = root / 'engineering_plan.json'
    plan = json.loads(path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    # Independent scalar timeline: every interval represents its exact leaf average.
    timeline, cache = [], {}
    for stage in range(1, 17):
        cache[(stage, stage)] = float(stage)
        for a, b in events(stage):
            assert a[1] + 1 == b[0] and b[1] == stage
            na, nb = a[1] - a[0] + 1, b[1] - b[0] + 1
            cache[(a[0], b[1])] = (na * cache[a] + nb * cache[b]) / (na + nb)
        keep = live_intervals(stage)
        cache = {k: cache[k] for k in keep}
        assert cache[(1, stage)] == (stage + 1) / 2
        assert all(cache[k] == (k[0] + k[1]) / 2 for k in cache)
        assert len(cache) == 2 * stage.bit_count() - 1
        timeline.append(dict(stage=stage, intervals=keep, events=events(stage)))
    strict_settings()
    device = configure(0, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = plan['engineering_tasks']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    cache, previous, records, started, maximum = {}, None, [], time.monotonic(), 0.
    updates = 0
    for stage, task in enumerate(tasks, 1):
        data.arrive(task)
        expert_path = Path(source['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        incoming = torch.load(expert_path, map_location='cpu', weights_only=True)
        cache[(stage, stage)] = incoming
        if stage == 1:
            previous = incoming
            continue
        initial, stage_records = None, []
        for a, b in events(stage):
            group = tasks[a[0] - 1:b[1]]
            tree, tp, tr = train_event(leaf, base, cache[a], cache[b], a[1] - a[0] + 1,
                b[1] - b[0] + 1, stage, data, group, plan, device)
            maximum = max(maximum, audit_export(base, cache[a], cache[b], tree, tp['mean'], plan['active']))
            cache[(a[0], b[1])] = tree
            chain, cp, cr = train_event(leaf, base, previous, incoming, stage - 1, 1,
                stage, data, group, plan, device, initial)
            maximum = max(maximum, audit_export(base, previous, incoming, chain, cp['mean'], plan['active']))
            assert tr['schedule_hash'] == cr['schedule_hash']
            if initial is not None:
                assert torch.equal(cp['initial'], initial)
            initial = cp['mean']
            updates += tr['updates'] + cr['updates']
            if stage < 4:
                assert all(torch.equal(tree[n], chain[n]) for n in tree)
                assert torch.equal(tp['snapshots'], cp['snapshots'])
            stage_records.append(dict(tree=tr, matched_chain=cr))
        if stage == 4:
            # Exact conventional one-phase path, not the extra-event control.
            ordinary, op, rec = train_event(leaf, base, previous, incoming, 3, 1,
                4, data, tasks, plan, device)
            reference, _, rp, rr = reference_train(leaf, base, previous, incoming, [], [],
                3, data, tasks, {}, plan, 'single_full', device)
            assert all(torch.equal(ordinary[n], reference[n]) for n in reference)
            assert torch.equal(op['snapshots'], rp['snapshots'])
            assert rec['schedule_hash'] == rr['schedule_hash']
            updates += rec['updates'] + rr['updates']
            assert any(not torch.equal(tree[n], chain[n]) for n in plan['active'])
        previous = chain
        cache = {k: cache[k] for k in live_intervals(stage)}
        records.append(stage_records)
        gc.collect()
        torch.cuda.empty_cache()
    assert updates == 60
    result = dict(status='passed', plan_sha256=sha256(path), causal_timeline=timeline,
        real_event_records=records, ordinary_one_phase_trajectory_bitwise=True,
        tree_chain_before_first_regrouping_bitwise=True, event_data_schedules_matched=True,
        independent_native_export_max_abs_error=maximum, optimizer_updates=updates,
        CE_examples=updates * plan['batch_size'], official_test_read=False,
        elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['causal_timeline', 'real_event_records']}, indent=2))


if __name__ == '__main__':
    main()
