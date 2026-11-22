"""Fit paired calibration pools or replay frozen original512 baselines."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from calibration_domain_training import DomainData, load_incoming, train_stage
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import strict_settings, evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=['single_full', 'source_soup'], required=True)
    parser.add_argument('--domain', choices=['training_calibration', 'development_calibration',
                                            'original512_reference'], required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    name = f'{args.condition}_{args.domain}_{args.order}'
    outdir = args.plan.parent / name
    outdir.mkdir(exist_ok=False)
    reference = args.domain == 'original512_reference'
    storage = Path(plan['checkpoint_storage_root']) / name
    if not reference:
        storage.mkdir(parents=True, exist_ok=False)
    strict_settings()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = DomainData(plan, source, device, args.domain)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    tasks = plan['pilot_tasks'] if args.order == 'forward' else list(reversed(plan['pilot_tasks']))
    records, previous = [], None
    started = time.monotonic()
    run = dict(plan=str(args.plan), plan_sha256=sha256(args.plan), condition=args.condition,
        domain=args.domain, order=tasks, gpu=args.gpu, status='running', stages=records,
        inference_checkpoint_count=1, inference_routing=False, official_test_read=False,
        calibration_EEG_read=not reference, reference_replay=reference,
        source_experts_used_per_task=1 if args.condition == 'single_full' else 3,
        strict_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if reference:
        prior_path = Path(plan['reference_runs'][f'{args.condition}_{args.order}'])
        prior = json.loads(prior_path.read_text())
        assert prior['status'] == 'completed' and prior['order'] == tasks
        run['reference_run'] = str(prior_path)
        run['reference_run_sha256'] = sha256(prior_path)
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task, calibration=not reference)
        assert set(data.readouts) == set(tasks[:stage])
        if reference:
            saved = prior['stages'][count]
            checkpoint = Path(saved['checkpoint_storage_path'])
            assert sha256(checkpoint) == saved['checkpoint_sha256']
            leaf.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            record = dict(updates=0, training_examples=0, checkpoint_storage_path=str(checkpoint),
                          checkpoint_sha256=saved['checkpoint_sha256'])
        else:
            incoming, paths = load_incoming(plan, source, task, args.condition)
            directory = outdir / f'stage_{stage:02d}'
            directory.mkdir()
            if count == 0:
                state = incoming
                leaf.load_state_dict(state, strict=True)
                record = dict(updates=0, training_examples=0)
            else:
                state, payload, record = train_stage(leaf, base, previous, incoming, count,
                    data, tasks[:stage], plan, device)
                save_tensor_file(payload, directory / 'coefficients.pt', outdir, storage)
                record['coefficients_sha256'] = sha256(directory / 'coefficients.pt')
                del payload
            previous = state
            checkpoint = save_tensor_file(state, directory / 'merged.ckpt', outdir, storage)
            record.update(source_paths=paths, source_sha256=[plan['expert_sha256'][p] for p in paths],
                checkpoint_storage_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            del state, incoming
        record.update(stage=stage, arriving_task=task)
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation'] = evaluate(leaf, data, tasks[:stage], plan['validation_batch_size'], device)
            record['macro_BAcc'] = float(np.mean([m['BAcc'] for m in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records, elapsed_seconds=time.monotonic() - started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(outdir / 'run.json', run)
        print(json.dumps(dict(name=name, stage=stage, macro_BAcc=record.get('macro_BAcc'),
                              seconds=run['elapsed_seconds'])), flush=True)
    run.update(status='completed', updates=sum(r['updates'] for r in records),
        training_examples=sum(r['training_examples'] for r in records))
    write_json(outdir / 'run.json', run)


if __name__ == '__main__':
    main()
