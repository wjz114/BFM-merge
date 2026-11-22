"""Run the frozen reference trainer with train/validation-only dataset I/O.

The training, optimizer, loss, evaluation and export code are unchanged.
Only the discarded official-test arrays are replaced by empty arrays.
"""
import numpy as np

import train_full_model_control as reference
from coverage_support import training_and_validation


def train_validation_only(task):
    tx, ty, vx, vy = training_and_validation(task)
    trial_time = 10 if task in {'EMO_FACED', 'ADHD_AliMotie'} else 4
    return trial_time, tx, ty, vx, vy, np.empty((0, *tx.shape[1:]), dtype=np.float32), np.empty(0, dtype=np.int64)


if __name__ == '__main__':
    reference.load_dataset = train_validation_only
    reference.main()
