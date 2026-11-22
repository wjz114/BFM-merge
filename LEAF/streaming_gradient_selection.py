"""Arrival-time paired-block selection in the bilateral coefficient space."""
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from d_downstream_qformer_ce import PrototypeClassifier
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from run_streaming_strict_history import TaskData
from streaming_bilateral_merge import StreamingMerger


def align_shared_rows(original, fresh, labels):
    """Keep every overlapping row at its original position, within its class."""
    assert len(np.unique(original)) == len(np.unique(fresh)) == len(original)
    assert np.array_equal(labels[original], labels[fresh])
    shared = set(original.tolist()) & set(fresh.tolist())
    result = np.full_like(original, -1)
    for k, row in enumerate(original):
        if int(row) in shared:
            result[k] = row
    for label in np.unique(labels[original]):
        positions = np.flatnonzero((labels[original] == label) & (result < 0))
        choices = [r for r in fresh if labels[r] == label and int(r) not in shared]
        assert len(positions) == len(choices)
        result[positions] = choices
    assert set(result.tolist()) == set(fresh.tolist())
    assert np.array_equal(labels[result], labels[original])
    return result


class SelectionData(TaskData):
    def __init__(self, plan, source, device):
        super().__init__(plan, source, device)
        self.fresh_indices = np.load(plan['fresh_calibration'])
        self.selection_records = {}

    def fresh_pool(self, task):
        original = self.indices[task]
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
            rows, labels = row_map(h, task, 'calibration')
            labels = labels.astype(np.int64)
            assert array_hash(labels) == self.audit[task]['train_labels_hash']
            fresh = align_shared_rows(original, self.fresh_indices[task], labels)
            x = selected_eeg(h, [rows[int(i)] for i in fresh])
        assert np.array_equal(labels[fresh], self.train[task][1].numpy())
        return torch.from_numpy(x), fresh


def gradient_geometry(gradients, epsilon):
    """Fixed diagonal RMS normalization, not a claim of Adam equivalence."""
    assert gradients.ndim == 3 and gradients.shape[0] == 2
    g = gradients.double()
    scale = g.square().mean((0, 1)).sqrt() + epsilon
    difference = (g[0] - g[1]) / (2 * scale)
    gram = difference @ difference.T
    return scale, gram


def choose_signs(gram, initial, max_flips):
    signs = initial.clone().double()
    assert gram.dtype == torch.float64 and torch.all(signs.abs() == 1)
    objectives = [float(signs @ gram @ signs)]
    flips = []
    tolerance = 1e-12 * max(float(gram.diag().sum()), 1.)
    for _ in range(max_flips):
        changes = 4 * (gram.diag() - signs * (gram @ signs))
        k = int(changes.argmin())
        if float(changes[k]) >= -tolerance:
            break
        signs[k] *= -1
        objective = float(signs @ gram @ signs)
        assert objective < objectives[-1]
        objectives.append(objective)
        flips.append(k)
    return signs, {'objectives': objectives, 'flips': flips,
                   'stopping_tolerance': tolerance,
                   'stopped_at_max_flips': len(flips) == max_flips}


def select_arriving(leaf, base, previous, incoming, count, data, task, plan, condition, device):
    assert condition in ('gradient_match', 'random_blocks') and count > 0
    started = time.monotonic()
    original_x, truth = data.train[task]
    fresh_x, fresh_ids = data.fresh_pool(task)
    bs = plan['selection_block_size']
    assert len(truth) == 512 and len(truth) % bs == 0
    blocks = len(truth) // bs
    merger = StreamingMerger(leaf, base, previous, incoming, {}, count, plan['active'], 'old_rows', {})
    initial_coefficients = merger.coefficients.detach().cpu().clone()
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    gradients, losses = [], []
    for x in (original_x, fresh_x):
        batch_gradients, batch_losses = [], []
        for block in range(blocks):
            ids = slice(block * bs, (block + 1) * bs)
            y = truth[ids].to(device)
            logits = merger.logits(wrapper, x[ids].to(device))
            loss = (F.cross_entropy(logits, y, reduction='none') * data.weights[task][y]).mean()
            gradient, = torch.autograd.grad(loss, merger.coefficients)
            assert torch.isfinite(loss) and torch.isfinite(gradient).all()
            batch_gradients.append(gradient.detach().cpu())
            batch_losses.append(float(loss.detach()))
        gradients.append(torch.stack(batch_gradients))
        losses.append(batch_losses)
    gradients = torch.stack(gradients)
    assert torch.equal(initial_coefficients, merger.coefficients.detach().cpu())
    del merger, wrapper, gradient, logits, loss
    scale, gram = gradient_geometry(gradients, plan['selection_epsilon'])
    rng = np.random.default_rng([plan['seed'], plan['tasks'].index(task), 41629])
    initial = torch.from_numpy(rng.choice([-1., 1.], size=blocks))
    if condition == 'gradient_match':
        selected, descent = choose_signs(gram, initial, plan['selection_max_flips'])
    else:
        selected = initial.clone()
        descent = {'objectives': [float(selected @ gram @ selected)], 'flips': [],
                   'stopping_tolerance': None, 'stopped_at_max_flips': False}
    use_fresh = np.repeat(selected.numpy() < 0, bs)
    selected_ids = np.where(use_fresh, fresh_ids, data.indices[task])
    assert len(np.unique(selected_ids)) == 512
    # Each label position stays identical; only EEG rows change.
    x = original_x.clone()
    x[use_fresh] = fresh_x[use_fresh]
    data.train[task] = x, truth
    record = {'task': task, 'stage': count + 1, 'condition': condition,
              'selection_examples': 1024, 'selection_backwards': 2 * blocks,
              'optimizer_updates': 0, 'retained_unique_rows': 512,
              'pool_unique_rows': int(len(np.union1d(data.indices[task], fresh_ids))),
              'shared_rows_at_identical_positions': int((data.indices[task] == fresh_ids).sum()),
              'fresh_blocks': int((selected < 0).sum()),
              'selected_eeg_hash': array_hash(x.numpy()),
              'selected_labels_hash': array_hash(truth.numpy()),
              'selected_indices_hash': array_hash(selected_ids),
              'original_eeg_hash': data.records[task]['calibration_eeg_hash'],
              'fresh_eeg_hash': array_hash(fresh_x.numpy()),
              'initial_coefficients_hash': array_hash(initial_coefficients.numpy()),
              'gradient_hash': array_hash(gradients.numpy()),
              'initial_objective': float(initial @ gram @ initial),
              'selected_objective': float(selected @ gram @ selected),
              'descent': descent, 'block_balanced_ce': losses,
              'elapsed_seconds': time.monotonic() - started}
    data.selection_records[task] = record
    data.records[task]['calibration_eeg_hash'] = record['selected_eeg_hash']
    payload = {'gradients': gradients, 'scale': scale, 'gram': gram,
               'initial_signs': initial, 'selected_signs': selected,
               'original_indices': torch.from_numpy(data.indices[task].copy()),
               'fresh_indices': torch.from_numpy(fresh_ids),
               'selected_indices': torch.from_numpy(selected_ids),
               'initial_coefficients': initial_coefficients}
    return payload, record
