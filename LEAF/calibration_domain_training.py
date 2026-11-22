"""Matched calibration provenance diagnostic for the existing bilateral merger."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier, prepare_text, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from run_streaming_strict_history import TaskData
from streaming_crossfit_training import source_mean
from streaming_bilateral_merge import StreamingMerger as BilateralMerger


class DomainData(TaskData):
    def __init__(self, plan, source, device, domain):
        super().__init__(plan, source, device)
        self.domain = domain
        self.pools = np.load(plan['pools'])
        self.golden = json.loads(Path(plan['data_audit']).read_text())['tasks']

    def read_subset(self, task, split):
        partition = 'calibration' if split == 'training_calibration' else 'validation'
        ids = self.pools[f'{task}__{split}_ids']
        expected = self.golden[task][split]
        assert array_hash(ids) == expected['ids_hash']
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
            rows, targets = row_map(h, task, partition)
            key = 'train_labels_hash' if partition == 'calibration' else 'validation_labels_hash'
            assert array_hash(targets.astype(np.int64)) == self.audit[task][key]
            x = selected_eeg(h, [rows[int(i)] for i in ids])
            y = targets[ids].astype(np.int64)
        assert array_hash(x) == expected['eeg_hash'], (task, split, 'EEG')
        assert array_hash(y) == expected['labels_hash'], (task, split, 'labels')
        assert len(y) == expected['rows']
        return torch.from_numpy(x), torch.from_numpy(y)

    def arrive(self, task, calibration=True):
        labels, text, instruction, prototypes = prepare_text(self.config, task)
        assert labels == self.source['labels'][task]
        assert text == self.source['instruction'][task]
        self.readouts[task] = instruction.to(self.device), prototypes.to(self.device)
        self.records[task] = {'calibration_domain': self.domain,
                             'calibration_EEG_read': calibration}
        if calibration:
            x, y = self.read_subset(task, self.domain)
            counts = torch.bincount(y, minlength=len(labels))
            weights = len(y) / (len(labels) * counts.float())
            assert torch.equal(weights, torch.ones_like(weights))
            self.train[task] = x, y
            self.weights[task] = weights.to(self.device)
            self.records[task].update(calibration_eeg_hash=array_hash(x.numpy()),
                calibration_labels_hash=array_hash(y.numpy()), calibration_size=len(y),
                class_weights=weights.tolist())

    def validation(self, task):
        assert task in self.readouts, 'A task must have arrived before evaluation'
        if task not in self.valid:
            self.valid[task] = self.read_subset(task, 'development_evaluation')
            x, y = self.valid[task]
            self.records[task].update(validation_size=len(y),
                validation_eeg_hash=array_hash(x.numpy()), validation_labels_hash=array_hash(y.numpy()))
        return self.valid[task]


def load_incoming(plan, source, task, condition):
    paths = [source['experts'][task]]
    if condition == 'source_soup':
        paths += plan['half_experts'][task]
    assert condition in ('single_full', 'source_soup')
    models = []
    for path in paths:
        assert sha256(Path(path)) == plan['expert_sha256'][path]
        models.append(torch.load(path, map_location='cpu', weights_only=True))
    incoming = source_mean(models, plan['active']) if len(models) > 1 else models[0]
    return incoming, paths


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, device):
    merger = BilateralMerger(leaf, base, previous, incoming, None, count, plan['active'], 'old_rows', None)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([merger.coefficients], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1, plan['tasks'].index(t), 217]).integers(0, len(data.train[t][1]), size=plan['rounds'] * bs) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, losses, schedule, label_schedule = [], [], [], []
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
            label_schedule.append([plan['tasks'].index(task), *y[ids].tolist()])
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
              'label_schedule_hash': array_hash(np.asarray(label_schedule, dtype=np.int64)),
              'sampling': 'with_replacement',
              'round_balanced_ce': losses, 'export_checks': checks,
              'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - start,
              'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    record['row_gate_mean_abs'] = float(mean[2 * len(plan['active']):174216].abs().mean())
    if mean.numel() > 174216:
        record['retention_gate_mean_abs'] = float(mean[174216:347666].abs().mean())
    if mean.numel() > 347666:
        record['sketch_gate_mean_abs'] = float(mean[347666:].abs().mean())
    assert mean.numel() == 347666
    return state, {'snapshots': torch.stack(snapshots), 'mean': mean,
                   'schedule': torch.tensor(schedule, dtype=torch.int64),
                   'label_schedule': torch.tensor(label_schedule, dtype=torch.int64)}, record

