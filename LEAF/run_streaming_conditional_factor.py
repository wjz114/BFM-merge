"""Bilateral calibration with coupled conditional projection factors."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256, PrototypeClassifier
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json
from streaming_conditional_factor_training import train_stage
from streaming_conditional_factor import append_expert, history_bytes
from streaming_history_sketch_merge import collect_split


def save_tensor_file(value, path, logical_root, storage_root):
    target = storage_root / path.relative_to(logical_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists() and not path.exists()
    torch.save(value, target)
    path.symlink_to(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--condition', choices=['coupled_factors', 'factorized_control'], required=True)
    p.add_argument('--order', choices=['forward', 'reverse'], default='forward')
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    checkpoint_storage = Path(plan['checkpoint_storage_root']) / f'seed{plan["seed"]}' / f'{args.condition}_{args.order}'
    checkpoint_storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = plan['tasks'] if args.order == 'forward' else list(reversed(plan['tasks']))
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = []
    previous, records = None, []
    run = {'plan': str(args.plan), 'plan_sha256': sha256(args.plan), 'condition': args.condition,
           'order': tasks, 'stages': records, 'status': 'running', 'gpu': args.gpu,
           'setting': 'sequential experts with 512 retained training calibration rows per seen task',
           'strict_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
           'cudnn_deterministic': torch.backends.cudnn.deterministic,
           'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
           'history_used_by_merger': True, 'memory_policy': 'original_expert_conditional_factors',
           'calibration_task_order': 'Actual arrival order, matched to the strict bilateral reference.',
           'extra_direction_construction_forward_examples': 0,
           'extra_direction_construction_backward_probes': 0}
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        incoming = torch.load(path, map_location='cpu', weights_only=True)
        wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
        history_started = time.monotonic()
        append_expert(history, incoming, wrapper.instruction, task)
        history_seconds = time.monotonic() - history_started
        del wrapper
        stage_dir = args.outdir / f'stage_{stage:02d}'
        stage_dir.mkdir()
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            projections, split, audit = collect_split(leaf, base, previous, incoming, data, tasks[:count],
                                                       plan, stage, 'history_sketch', device)
            save_tensor_file(split, stage_dir / 'input_split.pt', args.outdir, checkpoint_storage)
            write_json(stage_dir / 'input_split.json', audit)
            calibration_started = time.monotonic()
            state, coefficients, record = train_stage(leaf, base, previous, incoming, history,
                                                      count, data, tasks[:stage], plan, args.condition, device, projections)
            record['calibration_call_seconds'] = time.monotonic() - calibration_started
            record['input_split'] = audit
            save_tensor_file(coefficients, stage_dir / 'coefficients.pt', args.outdir, checkpoint_storage)
            record['coefficients_sha256'] = sha256(stage_dir / 'coefficients.pt')
            del coefficients
            del projections, split
        record['history_update_seconds'] = history_seconds
        record['history_bytes'] = history_bytes(history)
        record['retained_coefficient_bytes'] = 0
        record['history_arrival'] = {'task': task, 'expert_sha256': plan['expert_sha256'][task],
                                     'seen_experts': len(history)}
        previous = state
        checkpoint_path = save_tensor_file(state, stage_dir / 'merged.ckpt', args.outdir, checkpoint_storage)
        record['checkpoint_storage_path'] = str(checkpoint_path)
        record.update(stage=stage, arriving_task=task, checkpoint_sha256=sha256(stage_dir / 'merged.ckpt'))
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
    save_tensor_file({'history': history, 'count': len(tasks)}, args.outdir / 'history.pt', args.outdir, checkpoint_storage)
    run['history_sha256'] = sha256(args.outdir / 'history.pt')
    run.update(status='completed', updates=sum(r['updates'] for r in records),
               training_examples=sum(r['training_examples'] for r in records))
    run['elapsed_seconds'] = time.monotonic() - started
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
