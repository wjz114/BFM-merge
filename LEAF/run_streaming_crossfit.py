"""Incremental fold-proxy coefficient transfer and matched source controls."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json
from streaming_crossfit_training import CONDITIONS, source_mean, train_stage


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
    tasks = plan['pilot_tasks'] if args.order == 'forward' else list(reversed(plan['pilot_tasks']))
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    fold_audit = json.loads(Path(plan['fold_audit']).read_text())['audits']
    previous, previous_proxies, records, assignments = None, [], [], {}
    proxy_mode = args.condition in ('out_of_fold', 'in_fold')
    run = {'plan': str(args.plan), 'plan_sha256': sha256(args.plan), 'condition': args.condition,
        'order': tasks, 'stages': records, 'status': 'running', 'gpu': args.gpu,
        'setting': 'four-task development pilot; sequential source-training-fold proxies',
        'strict_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'persistent_training_proxy_count': 2 if proxy_mode else 0,
        'inference_checkpoint_count': 1, 'inference_routing': False,
        'source_experts_used_per_task': 1 if args.condition == 'single_full' else 3,
        'official_test_read': False}
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        assert set(data.train) == set(tasks[:stage])
        with np.load(plan['folds']) as folds:
            assignments[task] = folds[f'{task}__calibration_fold'].copy()
        assert array_hash(assignments[task]) == fold_audit[task]['calibration_fold_hash']
        assert array_hash(data.indices[task].astype(np.int64)) == fold_audit[task]['calibration_ids_hash']
        paths = [source['experts'][task]]
        if args.condition != 'single_full':
            paths += plan['half_experts'][task]
        models = []
        for path in paths:
            assert sha256(Path(path)) == plan['expert_sha256'][path]
            models.append(torch.load(path, map_location='cpu', weights_only=True))
        incoming = source_mean(models, plan['active']) if args.condition == 'source_soup' else models[0]
        incoming_proxies = models[1:] if proxy_mode else []
        directory = args.outdir / f'stage_{stage:02d}'
        directory.mkdir()
        if count == 0:
            state, proxy_states = incoming, incoming_proxies
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            state, proxy_states, payload, record = train_stage(leaf, base, previous, incoming,
                previous_proxies, incoming_proxies, count, data, tasks[:stage], assignments,
                plan, args.condition, device)
            save_tensor_file(payload, directory / 'coefficients.pt', args.outdir, storage)
            record['coefficients_sha256'] = sha256(directory / 'coefficients.pt')
            del payload
        previous, previous_proxies = state, proxy_states
        checkpoint = save_tensor_file(state, directory / 'merged.ckpt', args.outdir, storage)
        proxy_records = []
        for world, proxy in enumerate(proxy_states):
            proxy_path = save_tensor_file(proxy, directory / f'proxy{world}.ckpt', args.outdir, storage)
            proxy_records.append({'world': world, 'path': str(proxy_path), 'sha256': sha256(proxy_path)})
        record.update(stage=stage, arriving_task=task, source_paths=paths,
            source_sha256=[sha256(Path(p)) for p in paths], checkpoint_storage_path=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint), proxies=proxy_records)
        del models, incoming, incoming_proxies, state, proxy_states
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
