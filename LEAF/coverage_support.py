"""Original train/validation partitions and fixed-size shuffled sampling."""
import h5py
import numpy as np

from leaf_datasets.shared import DATA_FOLDER
from load_datasets import PREDEFINED_SPLITS, SUBJECT_SPLITS, filter_classes, load_by_index, split_train_valid


def training_and_validation(task):
    if task in PREDEFINED_SPLITS:
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            tx, ty, vx, vy = h['trainX'][:], h['trainY'][:], h['validX'][:], h['validY'][:]
    else:
        assert task in SUBJECT_SPLITS
        x, y = filter_classes(task, *load_by_index(task, range(SUBJECT_SPLITS[task][0])))
        tx, ty, vx, vy = split_train_valid(x, y, .2)
    return np.asarray(tx, dtype=np.float32), np.asarray(ty, dtype=np.int64), np.asarray(vx, dtype=np.float32), np.asarray(vy, dtype=np.int64)


def shuffled_schedule(task_order, pool_sizes, seed, batch_size=128):
    rngs = [np.random.default_rng([seed, k, 593]) for k in range(len(pool_sizes))]
    permutations = [rng.permutation(n) for rng, n in zip(rngs, pool_sizes)]
    offsets = [0]*len(pool_sizes)
    indices = []
    for task_index in task_order:
        k = int(task_index)
        assert pool_sizes[k] >= batch_size
        if offsets[k]+batch_size > pool_sizes[k]:
            permutations[k] = rngs[k].permutation(pool_sizes[k])
            offsets[k] = 0
        indices.append(permutations[k][offsets[k]:offsets[k]+batch_size])
        offsets[k] += batch_size
    return np.stack(indices)
