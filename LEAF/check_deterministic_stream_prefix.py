"""Run an unchanged three-expert calibration prefix with strict determinism."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_history_sketch_merge import TaskData, write_json
from run_streaming_predictive_sketch import train_stage
from streaming_history_sketch_merge import RANK, collect_split
from streaming_predictive_sketch import update_sketch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    root = Path('runs/deterministic_stream_prefix_v1')
    spec = json.loads((root / 'plan.json').read_text())
    for name, digest in spec['sha256'].items():
        assert sha256(Path(name)) == digest
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    args.outdir.mkdir(exist_ok=False)
    started = time.monotonic()
    plan = json.loads(Path(spec['scientific_plan']).read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    tasks = plan['tasks'][:3]
    previous, records = None, []
    run = {'status': 'running', 'stages': records, 'tasks': tasks,
           'strict_deterministic_algorithms': True, 'development_validation_read': False,
           'official_test_read': False, 'scope': 'Three-expert training-only prefix; same seed in both processes.'}
    for count, task in enumerate(tasks):
        data.arrive(task)
        incoming = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        statistics = None
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0, 'first_expert_sha256': sha256(Path(source['experts'][task]))}
        else:
            folder = args.outdir / f'stage_{count + 1:02d}'
            folder.mkdir()
            projections, split, audit = collect_split(leaf, base, previous, incoming, data,
                                                       tasks[:count], plan, count + 1,
                                                       'energy_history', device)
            state, coefficients, record, statistics = train_stage(
                leaf, base, previous, incoming, history, count, data, tasks[:count + 1],
                plan, 'energy_history', device, projections)
            record['input_split'] = audit
            torch.save(state, folder / 'merged.ckpt')
            torch.save(coefficients, folder / 'coefficients.pt')
            torch.save(statistics, folder / 'responses.pt')
            del projections, split, coefficients
        record['history_sketch_update'] = update_sketch(history, incoming, base, plan['active'],
                                                       'energy_history', statistics)
        record.update(stage=count + 1, arriving_task=task)
        records.append(record)
        previous = state
        del incoming, state, statistics
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.outdir / 'run.json', run)
    torch.save(history, args.outdir / 'history.pt')
    assert not data.valid
    run.update(status='completed', optimizer_updates=sum(r['updates'] for r in records),
               CE_example_passes=sum(r['training_examples'] for r in records),
               score_VJPs=sum(r.get('score_vjp_count', 0) for r in records),
               score_VJP_example_passes=sum(r.get('score_vjp_example_passes', 0) for r in records),
               auxiliary_forward_examples=sum(r['input_split']['extra_forward_examples'] for r in records[1:]),
               elapsed_seconds=time.monotonic() - started,
               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    assert run['optimizer_updates'] == run['score_VJPs'] == 60
    write_json(args.outdir / 'run.json', run)
    print(json.dumps({k: v for k, v in run.items() if k != 'stages'}), flush=True)


if __name__ == '__main__':
    main()
