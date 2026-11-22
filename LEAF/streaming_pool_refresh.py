"""Class-matched training-pool resampling for the frozen bilateral merger.

This is a sampling reference, not a novel merging operator. It changes EEG
examples while preserving every label position, loss weight and update rule.
"""
import h5py
import numpy as np
import torch

from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from diagnose_sensor_subspaces import row_map, read_rows
from leaf_datasets.shared import DATA_FOLDER
from run_streaming_strict_history import TaskData
from streaming_crossfit_training import train_stage as original_train_stage


CONDITIONS = ('fixed_fresh_pool', 'stage_fresh_pool')


def matched_indices(labels, template, seed, task_index, stage):
    rng = np.random.default_rng([seed, task_index, stage, 58173])
    result = np.empty(len(template), dtype=np.int64)
    for label in np.unique(template):
        positions = np.flatnonzero(template == label)
        candidates = np.flatnonzero(labels == label)
        assert len(candidates) >= len(positions)
        result[positions] = rng.choice(candidates, size=len(positions), replace=False)
    assert len(np.unique(result)) == len(result)
    assert np.array_equal(labels[result], template)
    return result


class RefreshData(TaskData):
    def __init__(self, plan, source, device):
        super().__init__(plan, source, device)
        self.plan = plan
        self.refresh_records = []

    def refresh(self, tasks, stage, condition):
        assert condition in CONDITIONS and set(tasks) == set(self.train)
        records, payload = {}, {}
        for task in tasks:
            template = self.train[task][1].numpy()
            sampling_stage = stage if condition == 'stage_fresh_pool' else 0
            with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as handle:
                rows, labels = row_map(handle, task, 'calibration')
                labels = labels.astype(np.int64)
                assert array_hash(labels) == self.audit[task]['train_labels_hash']
                ids = matched_indices(labels, template, self.plan['seed'],
                                      self.plan['tasks'].index(task), sampling_stage)
                selected = [rows[int(i)] for i in ids]
                x = selected_eeg(handle, selected)
                # Independent reader verifies actual EEG values, not only row IDs.
                independent = read_rows(handle, selected)
                assert np.array_equal(x, independent)
            self.train[task] = torch.from_numpy(x), torch.from_numpy(template.copy())
            records[task] = dict(stage=stage, sampling_stage=sampling_stage,
                training_pool_rows=len(labels), selected_rows=len(ids),
                ids_hash=array_hash(ids), labels_hash=array_hash(template),
                eeg_hash=array_hash(x), independent_reader_agreement=True,
                overlap_original512=len(np.intersect1d(ids, self.indices[task])))
            payload[task] = torch.from_numpy(ids)
        self.refresh_records.append(records)
        return payload, records


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device):
    ids, refresh = data.refresh(tasks, count + 1, condition)
    state, proxies, payload, record = original_train_stage(
        leaf, base, previous, incoming, [], [], count, data, tasks, {}, plan,
        'single_full', device)
    assert proxies == []
    payload['training_pool_ids'] = ids
    record['pool_refresh'] = refresh
    return state, payload, record
