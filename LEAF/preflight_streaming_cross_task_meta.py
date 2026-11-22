"""Check the two longest calibration inputs at the scientific batch size."""
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_cross_task_meta import configure_attention
from streaming_cross_task_meta_training import train_stage


def main():
    root = Path('runs/streaming_cross_task_meta_v1')
    assert not (root / 'preflight.json').exists()
    plan = json.loads((root / 'draft_scientific_plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    assert json.loads((root / 'engineering.json').read_text())['passed']
    plan = dict(plan, rounds=1, average_rounds=[1], validation_stages=[])
    strict_settings()
    device = configure(0, 0)
    configure_attention()
    torch.cuda.set_per_process_memory_fraction(.2, device)
    started = time.monotonic()
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['EMO_FACED', 'ADHD_AliMotie']
    for task in tasks:
        data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    previous, incoming = [torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks]
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    state, payload, record = train_stage(leaf, base, previous, incoming, 1, data, tasks,
        plan, 'cross_task_meta', device)
    assert not data.valid
    write_json(root / 'preflight.json', {'passed': True, 'source_sha256': sha256(Path(__file__)),
        'tasks': tasks, 'batch_shapes': {t: [16, *data.train[t][0].shape[1:]] for t in tasks},
        'record': record, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'gpu_memory_fraction': .2, 'elapsed_seconds': time.monotonic() - started,
        'development_validation_read': False, 'official_test_read': False})
    print(json.dumps({'passed': True, 'peak_reserved_GiB': torch.cuda.max_memory_reserved(device) / 2**30,
        'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
