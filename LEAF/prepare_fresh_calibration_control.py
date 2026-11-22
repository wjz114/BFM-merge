"""Select label-position-matched training rows with minimum calibration reuse."""
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from aligned_support import array_hash
from leaf_datasets.shared import DATA_FOLDER
from load_datasets import PREDEFINED_SPLITS, SUBJECT_SPLITS, filter_classes, split_train_valid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_labels(task):
    with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
        if task in PREDEFINED_SPLITS:
            return np.asarray(h['trainY'][:], dtype=np.int64)
        assert task in SUBJECT_SPLITS
        subjects = list(h.keys())[:SUBJECT_SPLITS[task][0]]
        y = np.concatenate([h[subject]['Y'][:] for subject in subjects])
    # Apply the original label filter before the exact deterministic split.
    ids, y = filter_classes(task, np.arange(len(y)), y)
    _, y, _, _ = split_train_valid(ids, y, .2)
    return np.asarray(y, dtype=np.int64)


def main():
    root = Path('runs/paper_merge_v2')
    out = root/'inputs/fresh512_data'
    out.mkdir(exist_ok=False)
    reference_path = root/'seed0/full_control_lr1e5_balanced_ce/run.json'
    reference = json.loads(reference_path.read_text())
    plan = json.loads((root/'inputs/full_control_coverage_replay_plan.json').read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    old_indices = np.load(plan['calibration'])
    schedule = np.load(plan['schedule'])
    selected, audits = {}, {}
    for k, task in enumerate(reference['datasets']):
        y = train_labels(task)
        assert array_hash(y) == coverage['audits'][task]['train_labels_hash']
        old = old_indices[task]
        assert len(old) == len(np.unique(old)) == 512
        labels = y[old]
        count = np.bincount(labels)
        assert count.tolist() == reference['dataset_audits'][task]['class_counts']
        rng = np.random.default_rng([20261115, k, 719])
        ids = np.empty_like(old)
        minimum_overlap = 0
        for c, n in enumerate(count):
            previous = old[labels == c]
            fresh = np.setdiff1d(np.flatnonzero(y == c), old)
            take = min(len(fresh), int(n))
            chosen = rng.choice(fresh, take, replace=False)
            missing = int(n)-take
            if missing:
                chosen = np.concatenate([chosen, rng.choice(previous, missing, replace=False)])
            rng.shuffle(chosen)
            ids[np.flatnonzero(labels == c)] = chosen
            minimum_overlap += max(0, 2*int(n)-int(np.sum(y == c)))
        assert len(np.unique(ids)) == 512 and np.array_equal(y[ids], labels)
        overlap = len(np.intersect1d(ids, old))
        assert overlap == minimum_overlap
        local_rows = schedule['local_indices'][schedule['task'] == k]
        assert np.array_equal(y[ids[local_rows]], y[old[local_rows]])
        selected[task] = ids
        audits[task] = {
            'train_size': len(y), 'class_counts': count.tolist(),
            'minimum_overlap': minimum_overlap, 'observed_overlap': overlap,
            'new_to_merger_calibration': 512-overlap, 'fully_disjoint': overlap == 0,
            'selected_label_hash': array_hash(y[ids]),
            'source_train_label_hash': array_hash(y),
            'exact_original_batch_labels': True,
        }
    np.savez(out/'training_indices.npz', **selected)
    record = {
        'unix': time.time(), 'datasets': audits,
        'reference': str(reference_path), 'reference_sha256': sha(reference_path),
        'original_calibration_sha256': plan['calibration_sha256'],
        'schedule_sha256': plan['schedule_sha256'],
        'selected_indices_sha256': sha(out/'training_indices.npz'),
        'fully_disjoint_tasks': [t for t, a in audits.items() if a['fully_disjoint']],
        'total_new_rows': sum(a['new_to_merger_calibration'] for a in audits.values()),
        'code_sha256': sha(__file__), 'test_data_read': False,
        'model_inference_performed': False,
        'scope': 'Only original expert-training labels read for selection. Matches the exact old512 label at every local position and therefore every replayed batch. Minimizes overlap with the calibration used to fit the initial merger. These examples are not new to the individual experts. This is a fixed sample-reuse diagnostic, not an independent test or new method.',
    }
    (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    plan.update({
        'fixed_unix': time.time(), 'training_pool': 'fresh512',
        'training_indices': str(out/'training_indices.npz'),
        'training_indices_sha256': sha(out/'training_indices.npz'),
        'fresh_data_audit': str(out/'run.json'),
        'fresh_data_audit_sha256': sha(out/'run.json'),
        'design': str(root/'inputs/fresh512_control_design.json'),
        'design_sha256': sha(root/'inputs/fresh512_control_design.json'),
        'scope': record['scope'],
    })
    for filename in ['train_fresh_calibration_control.py', 'prepare_fresh_calibration_control.py']:
        plan['source_sha256'][filename] = sha(filename)
    plan_path = root/'inputs/fresh512_control_plan.json'
    with plan_path.open('x') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(json.dumps({'plan': str(plan_path), 'plan_sha256': sha(plan_path),
                      'fully_disjoint_tasks': len(record['fully_disjoint_tasks']),
                      'total_new_rows': record['total_new_rows'],
                      'overlap': {t: a['observed_overlap'] for t, a in audits.items()}}, indent=2))


if __name__ == '__main__':
    main()
