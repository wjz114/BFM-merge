"""Create complementary source fine-tuning subsets; labels/row metadata only."""
import json
import time
from pathlib import Path

import h5py
import numpy as np

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER


def main():
    root = Path('runs/crossfit_merge_candidate_v1')
    spec = json.loads((root / 'fold_spec.json').read_text())
    for path, digest in spec['sha256'].items():
        assert sha256(Path(path)) == digest, path
    assert not (root / 'folds.npz').exists() and not (root / 'folds.json').exists()
    calibration = np.load(spec['calibration'])
    coverage = json.loads(Path(spec['coverage_audit']).read_text())['audits']
    arrays, audits = {}, {}
    for task in spec['tasks']:
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
            rows, y = row_map(h, task, 'calibration')
            valid_rows, vy = row_map(h, task, 'validation')
        y, vy = np.asarray(y, dtype=np.int64), np.asarray(vy, dtype=np.int64)
        assert array_hash(y) == coverage[task]['train_labels_hash']
        assert array_hash(vy) == coverage[task]['validation_labels_hash']
        assert not (set(rows) & set(valid_rows))
        fold = np.full(len(y), -1, dtype=np.int8)
        task_index = spec['all_tasks'].index(task)
        for label in np.unique(y):
            ids = np.flatnonzero(y == label)
            rng = np.random.default_rng([spec['fold_seed'], task_index, int(label), 1701])
            ids = rng.permutation(ids)
            fold[ids[::2]], fold[ids[1::2]] = 0, 1
        assert np.isin(fold, [0, 1]).all()
        halves = [np.flatnonzero(fold == k) for k in (0, 1)]
        assert not np.intersect1d(*halves).size
        assert np.array_equal(np.sort(np.concatenate(halves)), np.arange(len(y)))
        cal = calibration[task].astype(np.int64)
        assert len(cal) == len(np.unique(cal)) == 512
        assert all(np.array_equal(np.unique(y[ids]), np.unique(y)) for ids in halves)
        arrays[f'{task}__fold'] = fold
        arrays[f'{task}__calibration_fold'] = fold[cal]
        for k, ids in enumerate(halves):
            arrays[f'{task}__train{k}'] = ids
        audits[task] = {
            'full_train_rows': len(y), 'full_train_labels_hash': array_hash(y),
            'validation_rows': len(vy), 'validation_labels_hash': array_hash(vy),
            'assignment_hash': array_hash(fold), 'calibration_ids_hash': array_hash(cal),
            'calibration_fold_hash': array_hash(fold[cal]),
            'fold_sizes': [len(ids) for ids in halves],
            'fold_class_counts': [np.bincount(y[ids], minlength=len(np.unique(y))).tolist() for ids in halves],
            'calibration_fold_sizes': np.bincount(fold[cal], minlength=2).tolist(),
            'training_index_hashes': [array_hash(ids) for ids in halves],
            'training_label_hashes': [array_hash(y[ids]) for ids in halves],
            'train_validation_rows_disjoint': True,
            'source_exclusion_definition': 'Half k experts see only training rows whose fold assignment is k. Out-of-fold calibration maps each row to expert/proxy 1-k; in-fold maps to k.',
        }
    np.savez(root / 'folds.npz', **arrays)
    result = {'created_unix': time.time(), 'spec_sha256': sha256(root / 'fold_spec.json'),
        'folds_sha256': sha256(root / 'folds.npz'), 'audits': audits,
        'checks': {'complementary_cover': True, 'classes_present_in_both_halves': True,
            'calibration_ids_unique_and_exact': True, 'train_validation_rows_disjoint': True},
        'EEG_examples_read': 0, 'official_test_read': False,
        'scope': 'Class-stratified trial-row folds within the existing task training pool. This is not a subject-level split. No source expert has yet been trained on these folds.'}
    (root / 'folds.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({t: {'fold_sizes': a['fold_sizes'], 'calibration_fold_sizes': a['calibration_fold_sizes']}
                      for t, a in audits.items()}, indent=2))


if __name__ == '__main__':
    main()
