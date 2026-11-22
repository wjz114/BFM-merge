"""Refine arriving sources under historical contexts, then merge one native model."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, evaluate, write_json
from streaming_crossfit_training import train_stage
from streaming_history_context import CONDITIONS, adapt_stage


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
    tasks = plan['pilot_tasks'] if args.order == 'forward' else list(reversed(plan['pilot_tasks']))
    history, records, previous, started = [], [], None, time.monotonic()
    run = dict(plan=str(args.plan), plan_sha256=sha256(args.plan), condition=args.condition,
        order=tasks, gpu=args.gpu, status='running', stages=records, inference_checkpoint_count=1,
        inference_routing=False, official_test_read=False, new_source_pretraining_updates=0)
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        assert set(data.train) == set(tasks[:stage])
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][str(path)]
        incoming = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(incoming[n], base[n]) for n in set(base) - set(plan['active']))
        directory = outdir / f'stage_{stage:02d}'
        directory.mkdir()
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = dict(updates=0, training_examples=0)
        else:
            operand = incoming
            if args.condition != 'native_after':
                operand, average_payload, adaptation = adapt_stage(leaf, incoming, previous, history,
                    data, tasks[:stage], plan, args.condition, device)
                save_tensor_file(operand, directory / 'adapted_incoming.ckpt', outdir, storage)
            merged, _, payload, record = train_stage(leaf, base, previous, operand, [], [], count,
                data, tasks[:stage], {}, plan, 'single_full', device)
            save_tensor_file(payload, directory / 'coefficients.pt', outdir, storage)
            record['coefficients_sha256'] = sha256(directory / 'coefficients.pt')
            del payload
            if args.condition == 'native_after':
                save_tensor_file(merged, directory / 'before_native.ckpt', outdir, storage)
                state, average_payload, adaptation = adapt_stage(leaf, merged, previous, history,
                    data, tasks[:stage], plan, args.condition, device)
            else:
                state = merged
            save_tensor_file(average_payload, directory / 'adaptation_average.pt', outdir, storage)
            record['adaptation_average_sha256'] = sha256(directory / 'adaptation_average.pt')
            record['adaptation'] = adaptation
            del average_payload, operand, merged
        history.append({n: incoming[n].to(device) for n in plan['active']})
        previous = state
        checkpoint = save_tensor_file(state, directory / 'merged.ckpt', outdir, storage)
        record.update(stage=stage, arriving_task=task, source_path=str(path),
            source_sha256=plan['expert_sha256'][str(path)], checkpoint_sha256=sha256(checkpoint),
            original_history_count=len(history))
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
            seconds=run['elapsed_seconds'])), flush=True)
    run.update(status='completed', coefficient_updates=sum(r['updates'] for r in records),
        native_updates=sum(r.get('adaptation', {}).get('updates', 0) for r in records),
        training_examples=sum(r['training_examples'] + r.get('adaptation', {}).get('training_examples', 0)
                              for r in records))
    write_json(outdir / 'run.json', run)


if __name__ == '__main__':
    main()
