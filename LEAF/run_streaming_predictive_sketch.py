"""Compare energy and predictive-score history compression on identical replay."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from run_streaming_history_sketch_merge import TaskData, evaluate, write_json
from streaming_history_sketch_merge import RANK, collect_split
from streaming_predictive_sketch import StreamingMerger, observe_predictive, update_sketch


def train_stage(leaf, base, previous, incoming, history, count, data, tasks,
                plan, condition, device, projections):
    merger = StreamingMerger(leaf, base, previous, incoming, history, count,
                             plan['active'], condition, projections)
    wrappers = {task: PrototypeClassifier(leaf, *data.readouts[task]).eval() for task in tasks}
    optimizer = torch.optim.Adam([merger.coefficients], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {task: np.random.default_rng([plan['seed'], count + 1,
                    plan['tasks'].index(task), 217]).permutation(len(data.train[task][1])) for task in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, losses, schedule = [], [], []
    probe_seed = plan['predictive_probe_seed_base'] + 100000 * plan['seed'] + 1000 * count
    probe_generator = torch.Generator(device=device).manual_seed(probe_seed)
    probe_signs, observed_counts = [], []
    start = time.monotonic()
    for rnd in range(plan['rounds']):
        round_loss = []
        for k in order_rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            truth = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            parameters = merger.compose()
            logits = merger.logits(wrappers[task], x[ids].to(device), parameters)
            signs = torch.randint(0, 2, logits.shape, generator=probe_generator,
                                  device=device, dtype=torch.int8) * 2 - 1
            observed_counts.append(observe_predictive(
                merger, parameters, logits, signs, data.weights[task][truth]))
            probe_signs.append(signs.detach().cpu())
            loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(merger.coefficients.grad).all()
            optimizer.step()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
            del parameters, signs
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(merger.coefficients.detach().cpu().clone())
    statistics = merger.responses.export()
    statistics['probe_calls'] = statistics.pop('hook_calls')
    statistics['probe_signs'] = probe_signs
    statistics['probe_seed'] = probe_seed
    probe_hash = array_hash(torch.cat([value.flatten() for value in probe_signs]).numpy())
    statistics['probe_signs_hash'] = probe_hash
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    probes = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        for task in tasks:
            probes[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    del parameters, logits, loss, optimizer, wrappers, merger
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            native = wrapper(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(probes[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5
            assert checks[task]['changed_predictions'] == 0
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_balanced_ce': losses, 'export_checks': checks,
              'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - start,
              'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max()),
              'score_vjp_count': len(probe_signs),
              'score_vjp_example_passes': len(probe_signs) * bs,
              'score_vjp_coefficient_gradient_untouched': True,
              'score_vjp_observed_tensor_counts': sorted(set(observed_counts)),
              'probe_seed': probe_seed, 'probe_signs_hash': probe_hash,
              'probe_calls': statistics['probe_calls'],
              'appended_dictionary_bytes': statistics['appended_dictionary_bytes'],
              'response_covariance_bytes': statistics['response_covariance_bytes']}
    assert all(0 <= n <= len(schedule) for n in statistics['probe_calls'].values())
    assert all(torch.isfinite(x).all() for x in statistics['covariance'].values())
    return state, {'snapshots': torch.stack(snapshots), 'mean': mean}, record, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--condition', choices=['energy_history', 'predictive_history'], required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], default='forward')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = plan['tasks'] if args.order == 'forward' else list(reversed(plan['tasks']))
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = {name: torch.zeros((RANK, *base[name].shape), dtype=torch.float32) for name in plan['active']}
    previous, records = None, []
    run = {'plan': str(args.plan), 'plan_sha256': sha256(args.plan), 'condition': args.condition,
           'order': tasks, 'stages': records, 'status': 'running', 'gpu': args.gpu,
           'setting': 'sequential experts with 512 retained training calibration rows per seen task'}
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        incoming = torch.load(path, map_location='cpu', weights_only=True)
        stage_dir = args.outdir / f'stage_{stage:02d}'
        stage_dir.mkdir()
        statistics = None
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            projections, split, audit = collect_split(leaf, base, previous, incoming, data, tasks[:count],
                                                       plan, stage, args.condition, device)
            torch.save(split, stage_dir / 'input_split.pt')
            write_json(stage_dir / 'input_split.json', audit)
            state, coefficients, record, statistics = train_stage(
                leaf, base, previous, incoming, history, count, data, tasks[:stage],
                plan, args.condition, device, projections)
            record['input_split'] = audit
            torch.save(coefficients, stage_dir / 'coefficients.pt')
            torch.save(statistics, stage_dir / 'responses.pt')
            record['response_statistics_sha256'] = sha256(stage_dir / 'responses.pt')
            record['coefficients_sha256'] = sha256(stage_dir / 'coefficients.pt')
            del projections, split, coefficients
        sketch_started = time.monotonic()
        record['history_sketch_update'] = update_sketch(history, incoming, base, plan['active'],
                                                       args.condition, statistics)
        record['history_sketch_update']['seconds'] = time.monotonic() - sketch_started
        previous = state
        torch.save(state, stage_dir / 'merged.ckpt')
        record.update(stage=stage, arriving_task=task, checkpoint_sha256=sha256(stage_dir / 'merged.ckpt'))
        del incoming, state, statistics
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
    torch.save({'sketch': history, 'count': len(tasks)}, args.outdir / 'history.pt')
    run.update(status='completed', updates=sum(record['updates'] for record in records),
               training_examples=sum(record['training_examples'] for record in records),
               score_vjp_count=sum(record.get('score_vjp_count', 0) for record in records),
               score_vjp_example_passes=sum(record.get('score_vjp_example_passes', 0) for record in records))
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
