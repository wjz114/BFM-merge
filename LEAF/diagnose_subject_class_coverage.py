"""Locate missing subject/class cells and decompose cached validation gains."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    subject_path = root/'analysis/coverage_subject_sensitivity.json'
    source_path = root/'seed0/full_pool_span_diagnostic/run.json'
    fit_path = root/'seed0/original_fp32_tensor_balanced_ce/run.json'
    cp = Path('runs/aligned_merge_v1/seed0/tensor_balanced_ce/calibration_indices.npz')
    sp = root/'inputs/full_control_coverage_data/full_training_schedule.npz'
    subjects, source, fit = [json.loads(p.read_text()) for p in [subject_path, source_path, fit_path]]
    for p in [source_path, cp, sp]:
        assert sha256(p) == subjects['input_sha256'][str(p)]
    assert sha256(fit_path) == source['plan']['input_sha256'][str(fit_path)]
    calibration, schedule = np.load(cp), np.load(sp)
    hashes = {str(p): sha256(p) for p in [subject_path, source_path, fit_path, cp, sp, Path('diagnose_sensor_subspaces.py')]}
    records = {}
    for task, subject_record in subjects['records'].items():
        path = source_path.with_name(f'{task}_logits.pt')
        assert sha256(path) == subjects['input_sha256'][str(path)]
        hashes[str(path)] = sha256(path)
        cached = torch.load(path, map_location='cpu', weights_only=True)
        n = cached['n_calibration']
        k = cached['logits']['initial'].shape[-1]
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            train_rows, train_y = row_map(h, task, 'calibration')
            val_rows, val_y = row_map(h, task, 'validation')
        assert np.array_equal(val_y, cached['truth'][n:].numpy())
        assert array_hash(np.asarray(val_y, dtype=np.int64)) == subject_record['validation_label_hash']
        names = list(subject_record['per_subject'])
        ids = {s: i for i, s in enumerate(names)}
        ts = np.array([ids[key.removesuffix('/X')] for key, _ in train_rows])
        vs = np.array([ids[key.removesuffix('/X')] for key, _ in val_rows])
        train_cell, val_cell = ts*k+train_y, vs*k+val_y
        task_index = fit['datasets'].index(task)
        full_ids = schedule['local_indices'][schedule['task'] == task_index].reshape(-1)
        available = np.bincount(train_cell, minlength=len(names)*k)
        old = np.bincount(train_cell[calibration[task]], minlength=len(names)*k)
        full = np.bincount(train_cell[np.unique(full_ids)], minlength=len(names)*k)
        val_counts = np.bincount(val_y, minlength=k)
        weights = 1/(k*val_counts[val_y])
        assert abs(weights.sum()-1) < 1e-12
        masks = {'cell_seen_in_original512': old[val_cell] > 0,
                 'cell_unseen_in_original512': old[val_cell] == 0}
        correct = {mode: cached['logits'][mode][n:].argmax(-1).numpy() == val_y
                   for mode in ['initial', 'full_update', 'orthogonal_only']}
        strata = {}
        for key, mask in masks.items():
            mass = weights[mask].sum()
            contributions = {m: float(weights[mask]@v[mask]) for m, v in correct.items()}
            strata[key] = {'validation_rows': int(mask.sum()), 'balanced_metric_mass': float(mass),
                           'metric_contributions': contributions,
                           'normalized_weighted_accuracy': {m: v/float(mass) if mass else None for m, v in contributions.items()},
                           'full_repair_gain_contribution_pp': (contributions['full_update']-contributions['initial'])*100,
                           'outside_span_gain_contribution_pp': (contributions['orthogonal_only']-contributions['initial'])*100,
                           'full_repair_conditional_gain_pp': (contributions['full_update']-contributions['initial'])*100/float(mass) if mass else None}
        for mode in correct:
            reconstructed = sum(s['metric_contributions'][mode] for s in strata.values())
            assert abs(reconstructed-source['evaluations'][mode][task]['validation']['BAcc']) < 1e-12
        records[task] = {'subject_groups': len(names), 'classes': k, 'available_subject_class_cells': int((available > 0).sum()),
                         'original512_covered_cells': int((old > 0).sum()), 'full_sampled_covered_cells': int((full > 0).sum()),
                         'cells_added_by_full_sample': int(((old == 0)&(full > 0)).sum()),
                         'original512_cell_counts': old.reshape(len(names), k).tolist(),
                         'full_unique_cell_counts': full.reshape(len(names), k).tolist(),
                         'subject_names': names, 'strata': strata}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'records': records, 'seconds': time.time()-started,
              'new_model_inference': False, 'optimizer_updates': 0, 'eeg_arrays_read': False,
              'official_test_labels_or_arrays_read': False,
              'scope': 'All11subject-mappable tasks, same bank0 original512 and full-repair sample stream. Strata are determined solely by whether the training calibration has any example from the same subject/class cell. Conditional weighted accuracy renormalizes fixed whole-task class-balanced row weights within a stratum; it is not a new task metric or causal estimate. Contributions, without renormalization, sum exactly to the original task BAcc and its repair gain. Seen/unseen cells differ in difficulty and sampling probability, so conditional differences do not isolate coverage causally. All original calibration subjects were already present; no new-subject generalization claim.'}
    with (root/'analysis/subject_class_coverage.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({t: {k: v for k, v in r.items() if k not in ['original512_cell_counts', 'full_unique_cell_counts', 'subject_names']} for t, r in records.items()}, indent=2))


if __name__ == '__main__':
    main()
