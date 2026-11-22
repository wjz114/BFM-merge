"""Causal binary-counter merges, exporting one native model at every arrival."""
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
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json
from streaming_binary_tree import CONDITIONS, events, live_intervals, interval_key, train_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    name = f'{args.condition}_{args.order}'
    outdir = args.plan.parent / name
    outdir.mkdir(exist_ok=False)
    storage = Path(plan['checkpoint_storage_root']) / name
    storage.mkdir(parents=True, exist_ok=False)
    strict_settings()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    tasks = plan['tasks'] if args.order == 'forward' else list(reversed(plan['tasks']))
    cache, previous, records, started = {}, None, [], time.monotonic()
    run = dict(plan=str(args.plan), plan_sha256=sha256(args.plan), condition=args.condition,
        order=tasks, gpu=args.gpu, status='running', stages=records,
        inference_checkpoint_count=1, inference_routing=False, official_test_read=False,
        source_experts_used_per_task=1, strict_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
    for stage, task in enumerate(tasks, 1):
        data.arrive(task)
        assert set(data.train) == set(tasks[:stage])
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        incoming = torch.load(path, map_location='cpu', weights_only=True)
        directory = outdir / f'stage_{stage:02d}'
        directory.mkdir()
        event_records = []
        if args.condition == 'binary_tree':
            cache[(stage, stage)] = incoming
        if stage == 1:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            checkpoint = save_tensor_file(state, directory / 'merged.ckpt', outdir, storage)
        else:
            initial = None
            for j, (left_interval, right_interval) in enumerate(events(stage)):
                event_tasks = tasks[left_interval[0] - 1:right_interval[1]]
                if args.condition == 'binary_tree':
                    left, right = cache[left_interval], cache[right_interval]
                    left_count = left_interval[1] - left_interval[0] + 1
                    right_count = right_interval[1] - right_interval[0] + 1
                else:
                    left, right, left_count, right_count = previous, incoming, stage - 1, 1
                state, payload, record = train_event(leaf, base, left, right,
                    left_count, right_count, stage, data, event_tasks, plan, device, initial)
                eventdir = directory / f'event_{j:02d}'
                eventdir.mkdir()
                checkpoint = save_tensor_file(state, eventdir / 'merged.ckpt', outdir, storage)
                save_tensor_file(payload, eventdir / 'coefficients.pt', outdir, storage)
                record.update(left_interval=left_interval, right_interval=right_interval,
                    basis_intervals=([left_interval, right_interval] if args.condition == 'binary_tree'
                                     else [(1, stage - 1), (stage, stage)]),
                    checkpoint_sha256=sha256(checkpoint),
                    coefficients_sha256=sha256(eventdir / 'coefficients.pt'))
                event_records.append(record)
                if args.condition == 'binary_tree':
                    cache[(left_interval[0], right_interval[1])] = state
                else:
                    initial = payload['mean']
                del payload, left, right
            (directory / 'merged.ckpt').symlink_to(checkpoint)
            del initial
        previous = state
        if args.condition == 'binary_tree':
            keep = live_intervals(stage)
            assert all(k in cache for k in keep)
            cache = {k: cache[k] for k in keep}
            assert cache[(1, stage)] is previous
        native_bytes = sum(v.numel() * v.element_size() for v in state.values())
        record = dict(stage=stage, arriving_task=task, source_path=str(path),
            source_sha256=plan['expert_sha256'][task], checkpoint_storage_path=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint), events=event_records,
            updates=sum(e['updates'] for e in event_records),
            training_examples=sum(e['training_examples'] for e in event_records),
            persistent_native_checkpoint_count=(len(cache) if args.condition == 'binary_tree' else 1),
            persistent_native_bytes=native_bytes * (len(cache) if args.condition == 'binary_tree' else 1),
            persistent_intervals=[interval_key(k) for k in cache])
        del state, incoming
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation'] = evaluate(leaf, data, tasks[:stage], plan['validation_batch_size'], device)
            record['macro_BAcc'] = float(np.mean([v['BAcc'] for v in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records, elapsed_seconds=time.monotonic() - started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(outdir / 'run.json', run)
        print(json.dumps(dict(name=name, stage=stage, macro_BAcc=record.get('macro_BAcc'),
            events=len(event_records), live_models=record['persistent_native_checkpoint_count'],
            seconds=run['elapsed_seconds'])), flush=True)
    run.update(status='completed', updates=sum(r['updates'] for r in records),
               training_examples=sum(r['training_examples'] for r in records))
    write_json(outdir / 'run.json', run)


if __name__ == '__main__':
    main()
