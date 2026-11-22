"""Same frozen sensor-subspace calculation with sequential, cached H5 reads."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import covariance, read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER


def main():
    root = Path('runs/paper_merge_v2')
    plan_path = root/'inputs/sensor_subspace_memory_plan.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    original_plan = json.loads(Path(plan['original_plan']).read_text())
    output = root/'seed0/sensor_subspace_memory'
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    source = json.loads((Path(original_plan['source'])/'run.json').read_text())
    calibration = np.load(Path(original_plan['source'])/'calibration_indices.npz')
    reference = json.loads((root/'seed0/smoke_full_control_balanced_ce/run.json').read_text())['dataset_audits']
    records, null_bases, replay_checks = {}, {}, {}
    for task in source['datasets']:
        task_start = time.time()
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            train_rows, train_y = row_map(handle, task, 'calibration')
            selected = [train_rows[int(i)] for i in calibration[task]]
            val_rows, _ = row_map(handle, task, 'validation')
            # The math functions and row order are unchanged. Only I/O differs:
            # read each needed H5 array once, then index its in-memory copy.
            keys = dict.fromkeys(row[0] for row in selected+val_rows)
            memory = {key: handle[key][:] for key in keys}
        read_seconds = time.time()-task_start
        x = read_rows(memory, selected)
        assert array_hash(x) == reference[task]['eeg_hash'], task
        teacher = torch.load(root/f'seed0/ri_cache/{task}.pt', map_location='cpu', weights_only=True)
        assert np.array_equal(train_y[calibration[task]], teacher['truth'][:len(selected)].numpy())
        del x, teacher
        cov, _ = covariance(memory, selected)
        eig, vectors = np.linalg.eigh(cov)
        assert eig.min() > -1e-12*eig.max()
        rank = int(np.sum(eig > original_plan['relative_eigenvalue_cutoff']*eig.max()))
        basis = vectors[:, :65-rank]
        projector = basis@basis.T
        assert np.linalg.norm(projector@projector-projector) < 1e-10
        _, cal_error = covariance(memory, selected, basis)
        val_cov, val_error = covariance(memory, val_rows, basis)
        del memory
        arrays = {'covariance': cov, 'validation_covariance': val_cov, 'eigenvalues': eig,
                  'null_basis': basis, 'null_projector': projector}
        old = root/f'seed0/sensor_subspace_diagnostic/{task}.npz'
        if old.exists():
            previous = np.load(old)
            replay_checks[task] = {k: float(np.max(np.abs(arrays[k]-previous[k]))) if arrays[k].size else 0.0
                                  for k in ['covariance', 'validation_covariance', 'eigenvalues', 'null_projector']}
            assert all(value==0 for value in replay_checks[task].values()), replay_checks[task]
        np.savez(output/f'{task}.npz', **arrays)
        null_bases[task] = basis
        records[task] = {'rank': rank, 'nullity': 65-rank, 'calibration_examples': len(selected),
                        'validation_examples': len(val_rows), 'relative_calibration_null_energy_root': cal_error,
                        'relative_validation_null_energy_root': val_error,
                        'relative_eigenvalues': (eig/eig.max()).tolist(),
                        'calibration_eeg_hash': reference[task]['eeg_hash'],
                        'sequential_read_seconds': read_seconds, 'seconds': time.time()-task_start}
        (output/'progress.json').write_text(json.dumps({'records': records, 'replay_checks': replay_checks}, indent=2)+'\n')
        print(task, 'rank', rank, 'residual', val_error, 'read seconds', read_seconds,
              'total seconds', time.time()-task_start, 'exact replay', task in replay_checks, flush=True)
    overlap = {a: {b: float(np.linalg.norm(null_bases[a].T@null_bases[b], 'fro')**2)
                   for b in records} for a in records}
    assert len(replay_checks) >= plan['minimum_completed_replays']
    result = {'records': records, 'null_projector_inner_products': overlap, 'replay_checks': replay_checks,
              'plan': original_plan, 'io_plan': plan, 'io_plan_sha256': sha256(plan_path),
              'seconds': time.time()-start, 'code_sha256': sha256(Path(__file__)),
              'scope': 'Same exact original calibration IDs, row order, covariance functions, cutoff and full validation as the original frozen diagnostic. Sequential cached reads replace repeated H5 fancy indexing. All completed original datasets replay bit-exactly. No official test reads or model inference.'}
    (output/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
