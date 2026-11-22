"""Paired sample/label schedules, independent native formula, and EEG loader checks."""
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from calibration_domain_training_v2 import DomainData, load_incoming, train_stage
from d_downstream_qformer_ce import load_leaf, sha256
from report_streaming_replica_filter import audit_export
from run_streaming_strict_history import strict_settings


def main():
    root = Path('runs/calibration_domain_pilot_v1')
    path = root / 'engineering_plan_v2.json'
    plan = json.loads(path.read_text())
    assert not (root / 'engineering.json').exists()
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    strict_settings()
    device = configure(plan['gpu'], plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    started = time.monotonic()
    records, schedules, updates, max_error = {}, {}, 0, 0.
    for condition in plan['conditions']:
        for domain in plan['domains']:
            name = f'{condition}_{domain}'
            data = DomainData(plan, source, device, domain)
            leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
            previous, records[name], schedules[name] = None, [], []
            for count, task in enumerate(plan['pilot_tasks']):
                data.arrive(task)
                incoming, _ = load_incoming(plan, source, task, condition)
                if count == 0:
                    previous = incoming
                    continue
                state, payload, record = train_stage(leaf, base, previous, incoming, count,
                    data, plan['pilot_tasks'][:count + 1], plan, device)
                assert torch.equal(payload['mean'], payload['snapshots'].double().mean(0).float())
                error = audit_export(base, previous, incoming, state, payload['mean'], plan['active'])
                max_error = max(max_error, error)
                record['independent_export_max_abs_error'] = error
                records[name].append(record)
                schedules[name].append((record['schedule_hash'], record['label_schedule_hash']))
                updates += record['updates']
                previous = state
                del payload, incoming
                gc.collect()
                torch.cuda.empty_cache()
            del previous, data, leaf
    assert all(v == next(iter(schedules.values())) for v in schedules.values())
    assert updates == 72
    result = dict(status='passed', plan_sha256=sha256(path), records=records, updates=updates,
        training_CE_examples=updates * plan['batch_size'], paired_index_and_label_schedules=True,
        independent_export_max_abs_error=max_error, independent_EEG_hashes_matched=True,
        development_calibration_EEG_read=True, development_evaluation_EEG_read=False,
        official_test_read=False, elapsed_seconds=time.monotonic() - started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    with (root / 'engineering.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
