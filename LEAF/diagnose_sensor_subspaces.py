"""Measure actual channel subspaces using original calibration, without models."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from leaf_datasets.shared import DATA_FOLDER
from load_datasets import PREDEFINED_SPLITS, SUBJECT_SPLITS


def row_map(handle, task, split):
    if task in PREDEFINED_SPLITS:
        xkey, ykey = ('trainX', 'trainY') if split=='calibration' else ('validX', 'validY')
        return [(xkey, int(i)) for i in range(len(handle[ykey]))], handle[ykey][:]
    train_end = SUBJECT_SPLITS[task][0]
    rows, targets = [], []
    for subject in list(handle.keys())[:train_end]:
        y = handle[subject]['Y'][:]
        for i, value in enumerate(y):
            if task=='MI_HighGamma' and value==2:
                continue
            rows.append((subject+'/X', i))
            targets.append(value)
    keep = np.arange(len(rows)) % 5 != 0 if split=='calibration' else np.arange(len(rows)) % 5 == 0
    return [row for row, selected in zip(rows, keep) if selected], np.asarray(targets)[keep]


def read_rows(handle, rows):
    output = np.empty((len(rows), *handle[rows[0][0]].shape[1:]), dtype=np.float32)
    for key in dict.fromkeys(row[0] for row in rows):
        positions = np.asarray([i for i, row in enumerate(rows) if row[0]==key])
        indices = np.asarray([rows[i][1] for i in positions])
        order = np.argsort(indices)
        output[positions[order]] = handle[key][indices[order]]
    return output


def covariance(handle, rows, basis=None):
    total = np.zeros((65, 65), dtype=np.float64)
    projection_error, energy, samples = 0.0, 0.0, 0
    for start in range(0, len(rows), 32):
        x = read_rows(handle, rows[start:start+32]).astype(np.float64)
        flat = x.transpose(1, 0, 2).reshape(65, -1)
        total += flat@flat.T
        energy += float(np.sum(flat*flat))
        samples += flat.shape[1]
        if basis is not None:
            projection_error += float(np.sum((basis.T@flat)**2))
    return total/samples, np.sqrt(projection_error/energy) if basis is not None else None


def main():
    root = Path('runs/paper_merge_v2')
    plan = json.loads((root/'inputs/sensor_subspace_diagnostic_plan.json').read_text())
    assert sha256(Path(__file__)) == plan['code_sha256']
    output = root/'seed0/sensor_subspace_diagnostic'
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    calibration = np.load(Path(plan['source'])/'calibration_indices.npz')
    reference = json.loads((root/'seed0/smoke_full_control_balanced_ce/run.json').read_text())['dataset_audits']
    records, null_bases = {}, {}
    for task in source['datasets']:
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            train_rows, train_y = row_map(handle, task, 'calibration')
            selected = [train_rows[int(i)] for i in calibration[task]]
            x = read_rows(handle, selected)
            assert array_hash(x) == reference[task]['eeg_hash'], task
            teacher = torch.load(root/f'seed0/ri_cache/{task}.pt', map_location='cpu', weights_only=True)
            assert np.array_equal(train_y[calibration[task]], teacher['truth'][:len(selected)].numpy())
            del x, teacher
            cov, _ = covariance(handle, selected)
            eig, vectors = np.linalg.eigh(cov)
            assert eig.min() > -1e-12*eig.max()
            rank = int(np.sum(eig > plan['relative_eigenvalue_cutoff']*eig.max()))
            basis = vectors[:, :65-rank]
            projector = basis@basis.T
            assert np.linalg.norm(projector@projector-projector) < 1e-10
            _, cal_error = covariance(handle, selected, basis)
            val_rows, _ = row_map(handle, task, 'validation')
            val_cov, val_error = covariance(handle, val_rows, basis)
            np.savez(output/f'{task}.npz', covariance=cov, validation_covariance=val_cov,
                     eigenvalues=eig, null_basis=basis, null_projector=projector)
            null_bases[task] = basis
            records[task] = {'rank': rank, 'nullity': 65-rank, 'calibration_examples': len(selected),
                             'validation_examples': len(val_rows), 'relative_calibration_null_energy_root': cal_error,
                             'relative_validation_null_energy_root': val_error,
                             'relative_eigenvalues': (eig/eig.max()).tolist(),
                             'calibration_eeg_hash': reference[task]['eeg_hash']}
            (output/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
            print(task, 'rank', rank, 'nullity', 65-rank, 'held-out residual', val_error, flush=True)
    overlap = {a: {b: float(np.linalg.norm(null_bases[a].T@null_bases[b], 'fro')**2)
                    for b in records} for a in records}
    result = {'records': records, 'null_projector_inner_products': overlap, 'plan': plan,
              'seconds': time.time()-start, 'code_sha256': sha256(Path(__file__)),
              'scope': 'Uncentered channel covariance of exact original512 calibration trials, checked against full exposed validation. No model evaluation, fitting or official-test reads. Empirical numerical rank is not proof of preprocessing provenance or unseen-data support.'}
    (output/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
