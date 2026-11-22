"""Prepare per-expert arrival payloads using original training calibration only."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from fisher_moment_merge import estimate_fisher
from run_streaming_subspace_merge import TaskData, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval()
    records = {}
    run = {'status': 'running', 'plan': str(args.plan), 'plan_sha256': sha256(args.plan),
           'experts': records, 'official_test_read': False, 'validation_read': False,
           'scope': 'Independent per-expert arrival payloads. A merger loads only the currently arriving task payload.'}
    started = time.monotonic()
    for task in plan['tasks']:
        data.arrive(task)
        checkpoint = Path(source['experts'][task])
        assert sha256(checkpoint) == plan['expert_sha256'][task]
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        leaf.load_state_dict(state, strict=True)
        fisher, record = estimate_fisher(leaf, data, task, plan, device, plan['fisher_examples'])
        path = args.outdir / f'{task}.pt'
        torch.save(fisher, path)
        record.update(path=str(path), sha256=sha256(path), expert_sha256=sha256(checkpoint),
                      calibration_audit=data.records[task])
        records[task] = record
        run.update(elapsed_seconds=time.monotonic() - started,
                   forward_examples=sum(x['forward_examples'] for x in records.values()),
                   backward_probes=sum(x['backward_probes'] for x in records.values()))
        write_json(args.outdir / 'run.json', run)
        print(task, record['raw_global_l2'], run['elapsed_seconds'], flush=True)
        del fisher, state, data.train[task]
        gc.collect()
        torch.cuda.empty_cache()
    run.update(status='completed', peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
