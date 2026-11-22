"""Train and evaluate matched sequential merges; never load official test data."""
import argparse
import gc
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import (ROOT, PrototypeClassifier, classification_metrics,
                                    load_leaf, prepare_text, sha256)
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from streaming_residual_merge import add_history
from streaming_subspace_merge import StreamingMerger, collect_split


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


class TaskData:
    def __init__(self, plan, source, device):
        self.source, self.device = source, device
        self.indices = np.load(plan['calibration'])
        self.audit = json.loads(Path(plan['coverage_audit']).read_text())['audits']
        self.reference = json.loads(Path(plan['calibration_reference']).read_text())['dataset_audits']
        self.config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
        self.train, self.valid, self.readouts, self.weights, self.records = {}, {}, {}, {}, {}

    def arrive(self, task):
        labels, text, instruction, prototypes = prepare_text(self.config, task)
        assert labels == self.source['labels'][task]
        assert text == self.source['instruction'][task]
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
            rows, targets = row_map(h, task, 'calibration')
            assert array_hash(targets.astype(np.int64)) == self.audit[task]['train_labels_hash']
            x = selected_eeg(h, [rows[int(i)] for i in self.indices[task]])
            y = targets[self.indices[task]].astype(np.int64)
        assert array_hash(x) == self.reference[task]['eeg_hash']
        counts = torch.bincount(torch.from_numpy(y), minlength=len(labels))
        weights = len(y) / (len(labels) * counts.float())
        assert weights.tolist() == self.source['ce_class_weights'][task]
        self.train[task] = torch.from_numpy(x), torch.from_numpy(y)
        self.readouts[task] = instruction.to(self.device), prototypes.to(self.device)
        self.weights[task] = weights.to(self.device)
        self.records[task] = {'calibration_eeg_hash': array_hash(x), 'calibration_labels_hash': array_hash(y),
                              'calibration_size': len(y), 'class_weights': weights.tolist()}

    def validation(self, task):
        if task not in self.valid:
            with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
                rows, y = row_map(h, task, 'validation')
                x = selected_eeg(h, rows)
                y = y.astype(np.int64)
            assert array_hash(x) == self.audit[task]['validation_eeg_hash']
            assert array_hash(y) == self.audit[task]['validation_labels_hash']
            self.valid[task] = torch.from_numpy(x), torch.from_numpy(y)
            self.records[task]['validation_size'] = len(y)
        return self.valid[task]


@torch.no_grad()
def evaluate(leaf, data, tasks, batch_size, device):
    metrics = {}
    for task in tasks:
        wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
        x, y = data.validation(task)
        logits = torch.cat([wrapper(x[i:i + batch_size].to(device)).cpu()
                            for i in range(0, len(y), batch_size)])
        metrics[task] = classification_metrics(logits, y)
    return metrics


def train_stage(leaf, base, previous, incoming, history, count, data, tasks, plan, condition, device, projections):
    merger = StreamingMerger(leaf, base, previous, incoming, history, count, plan['active'], condition, projections)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([merger.coefficients], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1, plan['tasks'].index(t), 217]).permutation(
        len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
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
            logits = merger.logits(wrappers[task], x[ids].to(device))
            loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(merger.coefficients.grad).all()
            optimizer.step()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(merger.coefficients.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    # Native export must preserve actual task logits, including historical tasks.
    functional_probes = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        for task in tasks:
            functional_probes[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    del parameters, logits, loss, optimizer, wrappers, merger
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            native = wrapper(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(functional_probes[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5
            assert checks[task]['changed_predictions'] == 0
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_balanced_ce': losses, 'export_checks': checks,
              'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - start,
              'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    record['row_gate_mean_abs'] = float(mean[2 * len(plan['active']):174216].abs().mean())
    if mean.numel() > 174216:
        record['split_gate_mean_abs'] = float(mean[174216:].abs().mean())
    return state, {'snapshots': torch.stack(snapshots), 'mean': mean}, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--condition', choices=['incoming_rows', 'activation_split', 'random_split'], required=True)
    p.add_argument('--order', choices=['forward', 'reverse'], default='forward')
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
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
    history = {n: torch.zeros_like(base[n], dtype=torch.float64) for n in plan['active']}
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
        if count == 0:
            state = incoming
            leaf.load_state_dict(state, strict=True)
            record = {'updates': 0, 'training_examples': 0}
        else:
            projections, split, audit = collect_split(leaf, base, previous, incoming, data, tasks[:count],
                                                       plan, stage, args.condition, device)
            torch.save(split, stage_dir / 'input_split.pt')
            write_json(stage_dir / 'input_split.json', audit)
            state, coefficients, record = train_stage(leaf, base, previous, incoming, history,
                                                      count, data, tasks[:stage], plan, args.condition, device, projections)
            record['input_split'] = audit
            torch.save(coefficients, stage_dir / 'coefficients.pt')
            del projections, split
        if args.condition == 'history_rows':
            add_history(history, incoming, base, plan['active'])
        previous = state
        torch.save(state, stage_dir / 'merged.ckpt')
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
    if args.condition == 'history_rows':
        torch.save({'sum': history, 'count': len(tasks)}, args.outdir / 'history.pt')
    run.update(status='completed', updates=sum(r['updates'] for r in records),
               training_examples=sum(r['training_examples'] for r in records))
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
