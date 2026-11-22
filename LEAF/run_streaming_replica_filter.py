"""Sequential replica-filtered experts with unchanged bilateral CE calibration."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json, train_stage
from streaming_bilateral_merge import collect_split
from streaming_replica_filter import CONDITIONS, prepare_incoming


def load_replicas(plan, task):
    models = []
    for seed in plan['replica_seeds']:
        path = Path(plan['replica_experts'][str(seed)][task])
        assert sha256(path) == plan['replica_expert_sha256'][str(seed)][task]
        models.append(torch.load(path, map_location='cpu', weights_only=True))
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    storage = Path(plan['checkpoint_storage_root']) / f'{args.condition}_{args.order}'
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = plan['tasks'] if args.order == 'forward' else list(reversed(plan['tasks']))
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    previous, records = None, []
    run = {'plan': str(args.plan), 'plan_sha256': sha256(args.plan), 'condition': args.condition,
           'order': tasks, 'stages': records, 'status': 'running', 'gpu': args.gpu,
           'setting': 'three same-task source replicas arrive together; original512 merge calibration',
           'strict_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
           'cudnn_deterministic': torch.backends.cudnn.deterministic,
           'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
           'history_used_by_merger': False, 'persistent_auxiliary_history_bytes': 0,
           'extra_direction_construction_forward_examples': 0,
           'extra_direction_construction_backward_probes': 0,
           'source_expert_count': 48, 'official_test_read': False}
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        construction_started = time.monotonic()
        replicas = load_replicas(plan, task)
        mean, incoming, geometry = prepare_incoming(base, previous, replicas, plan['active'],
            args.condition, plan['tasks'].index(task))
        construction_seconds = time.monotonic() - construction_started
        del replicas, mean
        directory = args.outdir / f'stage_{stage:02d}'
        directory.mkdir()
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            projections, split, audit = collect_split(leaf, base, previous, incoming, data,
                tasks[:count], plan, stage, 'old_rows', device)
            save_tensor_file(split, directory / 'input_split.pt', args.outdir, storage)
            write_json(directory / 'input_split.json', audit)
            state, payload, record = train_stage(leaf, base, previous, incoming, None,
                count, data, tasks[:stage], plan, 'old_rows', device, projections)
            record['input_split'] = audit
            save_tensor_file(payload, directory / 'coefficients.pt', args.outdir, storage)
            record['coefficients_sha256'] = sha256(directory / 'coefficients.pt')
            del projections, split, payload
        save_tensor_file(incoming, directory / 'incoming.ckpt', args.outdir, storage)
        record.update(incoming_sha256=sha256(directory / 'incoming.ckpt'),
                      construction_seconds=construction_seconds, filter_geometry=geometry)
        previous = state
        checkpoint = save_tensor_file(state, directory / 'merged.ckpt', args.outdir, storage)
        record.update(stage=stage, arriving_task=task, checkpoint_storage_path=str(checkpoint),
                      checkpoint_sha256=sha256(directory / 'merged.ckpt'))
        del incoming, state
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation'] = evaluate(leaf, data, tasks[:stage], plan['validation_batch_size'], device)
            record['macro_BAcc'] = float(np.mean([m['BAcc'] for m in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records, elapsed_seconds=time.monotonic() - started,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.outdir / 'run.json', run)
        print(json.dumps({'condition': args.condition, 'stage': stage, 'task': task,
                          'macro_BAcc': record.get('macro_BAcc'), 'seconds': run['elapsed_seconds']}), flush=True)
    run.update(status='completed', updates=sum(r['updates'] for r in records),
               training_examples=sum(r['training_examples'] for r in records),
               elapsed_seconds=time.monotonic() - started)
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
