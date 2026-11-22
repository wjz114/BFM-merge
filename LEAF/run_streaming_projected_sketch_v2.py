"""Native-coordinate, norm-restored Adam in bounded streaming expert spaces."""
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
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from row_projected_adam import ProjectedAdam
from run_streaming_correction_retention import reload_checks
from run_streaming_subspace_merge import TaskData, evaluate, write_json
from streaming_correction_retention import average_snapshots, initialize
from streaming_projected_sketch_v2 import RANK, build_bases, update_sketch


def train_stage(leaf, initial, bases, data, tasks, plan, stage, device, outdir):
    active = set(plan['active'])
    leaf.load_state_dict(initial, strict=True)
    leaf.eval()
    parameters = []
    for name, value in leaf.named_parameters():
        value.requires_grad_(name in active)
        if name in active:
            parameters.append((name, value))
    assert [n for n, _ in parameters] == list(bases)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    snapshot_steps = [rnd * len(tasks) for rnd in plan['average_rounds']]
    optimizer = ProjectedAdam(parameters, bases, outdir, snapshot_steps,
                              lr=plan['learning_rate'], weight_decay=0)
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], stage, plan['tasks'].index(t), 217]).permutation(
        len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], stage, 931])
    snapshots, losses, schedule = [], [], []
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
            logits = wrappers[task](x[ids].to(device))
            loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for _, v in parameters], float('inf'), error_if_nonfinite=True)
            optimizer.step()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append({n: v.detach().cpu().clone() for n, v in parameters})
    assert len(snapshots) == len(plan['average_rounds'])
    assert all(torch.equal(v.cpu(), initial[n]) for n, v in leaf.state_dict().items() if n not in active)
    state = average_snapshots(snapshots, initial, active)
    leaf.load_state_dict(state, strict=True)
    leaf.requires_grad_(False)
    coordinate_audit = optimizer.audit_average(snapshot_steps)
    assert coordinate_audit['max_abs_weight_mean_vs_coordinate_mean'] <= 1e-6
    probes = {}
    with torch.no_grad():
        for task in tasks:
            probes[task] = wrappers[task](data.train[task][0][:bs].to(device)).cpu()
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_balanced_ce': losses, 'training_seconds': time.monotonic() - start,
              'trainable_native_parameters': sum(p.numel() for _, p in parameters),
              'stored_coordinate_count': sum(b.numel() for b in optimizer.coordinates),
              'average_rounds': plan['average_rounds'], 'fixed_tensors_exact': True,
              'coordinate_average_audit': coordinate_audit,
              'projection_statistics': {
                  'zero_projection_steps': sum(r['zero_projection'] for r in optimizer.statistics),
                  'mean_restoration_scale': float(np.mean([r['restoration_scale'] for r in optimizer.statistics])),
                  'max_actual_step_norm_relative_error': max(abs(r['actual_to_proposal_norm_ratio'] - 1)
                      for r in optimizer.statistics if r['actual_to_proposal_norm_ratio'] is not None)}}
    del optimizer, wrappers, logits, loss
    return state, snapshots, probes, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--condition', choices=['bilateral', 'history_sketch', 'random_sketch'], required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], required=True)
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
    history = {} if args.condition == 'bilateral' else {
        n: torch.zeros((RANK, *base[n].shape), dtype=torch.float32) for n in plan['active']}
    previous, records = None, []
    run = {'plan': str(args.plan), 'plan_sha256': sha256(args.plan), 'condition': args.condition,
           'order': tasks, 'stages': records, 'status': 'running', 'gpu': args.gpu,
           'official_test_read': False, 'setting': 'Original512replay with native projected Adam; ordinary shared output.'}
    for count, task in enumerate(tasks):
        stage = count + 1
        data.arrive(task)
        expert_path = Path(source['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        incoming = torch.load(expert_path, map_location='cpu', weights_only=True)
        assert all(torch.equal(incoming[n], base[n]) for n in base if n not in plan['active'])
        stage_dir = args.outdir / f'stage_{stage:02d}'
        stage_dir.mkdir()
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            initial = initialize(previous, incoming, plan['active'], count, 'recursive')
            bases, audit = build_bases(base, previous, incoming, history, count, plan['active'], args.condition)
            torch.save(bases, stage_dir / 'bases.pt')
            state, snapshots, probes, record = train_stage(leaf, initial, bases, data, tasks[:stage],
                                                           plan, stage, device, stage_dir)
            record['basis_audit'] = audit
            torch.save({'rounds': plan['average_rounds'], 'snapshots': snapshots}, stage_dir / 'snapshots.pt')
            record['snapshots_sha256'] = sha256(stage_dir / 'snapshots.pt')
            record['bases_sha256'] = sha256(stage_dir / 'bases.pt')
            del initial, snapshots, bases
        if history:
            record['history_update'] = update_sketch(history, incoming, base, plan['active'])
        previous = state
        checkpoint = stage_dir / 'merged.ckpt'
        torch.save(state, checkpoint)
        record.update(stage=stage, arriving_task=task, checkpoint_sha256=sha256(checkpoint))
        if count:
            record['export_checks'] = reload_checks(leaf, checkpoint, probes, data, tasks[:stage],
                                                    plan['batch_size'], device)
            del probes
        del incoming, state
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation'] = evaluate(leaf, data, tasks[:stage], plan['validation_batch_size'], device)
            record['macro_BAcc'] = float(np.mean([v['BAcc'] for v in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records, elapsed_seconds=time.monotonic() - started,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.outdir / 'run.json', run)
        print(json.dumps({'condition': args.condition, 'stage': stage, 'macro_BAcc': record.get('macro_BAcc'),
                          'seconds': run['elapsed_seconds']}), flush=True)
    if history:
        torch.save({'sketch': history, 'count': len(tasks)}, args.outdir / 'history.pt')
    run.update(status='completed', updates=sum(s['updates'] for s in records),
               training_examples=sum(s['training_examples'] for s in records),
               persistent_sketch_bytes=sum(v.numel() * v.element_size() for v in history.values()))
    assert run['updates'] == plan['updates_per_arm']
    assert run['training_examples'] == plan['training_examples_per_arm']
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
