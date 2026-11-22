"""Subject-group sensitivity of cached validation gains; no new model inference."""
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
from load_datasets import SUBJECT_SPLITS


def class_statistics(y, prediction, classes):
    count = np.bincount(y, minlength=classes)
    correct = np.bincount(y, weights=(prediction == y).astype(np.int64), minlength=classes)
    return count.astype(np.int64), correct.astype(np.int64)


def bacc(count, correct):
    assert np.all(count > 0)
    return float(np.mean(correct/count))


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    source_path = root/'seed0/full_pool_span_diagnostic/run.json'
    coverage_path = root/'inputs/full_control_coverage_data/run.json'
    source, coverage = [json.loads(p.read_text()) for p in [source_path, coverage_path]]
    calibration_path = Path('runs/aligned_merge_v1/seed0/tensor_balanced_ce/calibration_indices.npz')
    schedule_path = root/'inputs/full_control_coverage_data/full_training_schedule.npz'
    calibration, schedule = np.load(calibration_path), np.load(schedule_path)
    for path in [coverage_path, calibration_path]:
        assert sha256(path) == source['plan']['input_sha256'][str(path)]
    control_path = Path(source['plan']['source'])/'run.json'
    control = json.loads(control_path.read_text())
    assert sha256(control_path) == source['plan']['input_sha256'][str(control_path)]
    assert sha256(schedule_path) == control['plan']['schedule_sha256']
    for name, expected in source['plan']['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    hashes = {str(p): sha256(p) for p in [source_path, coverage_path, calibration_path, schedule_path,
                                         control_path, Path('diagnose_sensor_subspaces.py'), Path('load_datasets.py')]}
    tasks = control['datasets']
    modes = ['initial', 'full_update', 'parallel_only', 'orthogonal_only']
    pairs = [('full_update', 'initial'), ('orthogonal_only', 'initial'),
             ('parallel_only', 'initial'), ('full_update', 'orthogonal_only')]
    records, omitted, replay = {}, {}, {}
    for task_index, task in enumerate(tasks):
        path = source_path.with_name(f'{task}_logits.pt')
        hashes[str(path)] = sha256(path)
        data = torch.load(path, map_location='cpu', weights_only=True)
        n = data['n_calibration']
        assert n == 512
        truth = data['truth'][n:].numpy()
        classes = data['logits']['initial'].shape[-1]
        predictions = {m: data['logits'][m][n:].argmax(-1).numpy() for m in modes}
        all_stats = {m: class_statistics(truth, predictions[m], classes) for m in modes}
        replay[task] = {}
        for mode, (counts, correct) in all_stats.items():
            value = bacc(counts, correct)
            assert abs(value-source['evaluations'][mode][task]['validation']['BAcc']) < 1e-12
            replay[task][mode] = value
        if task not in SUBJECT_SPLITS:
            omitted[task] = 'Predefined array split; the existing loader/prediction cache supplies no subject-to-row map. No subject identities inferred.'
            continue
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            train_rows, train_y = row_map(h, task, 'calibration')
            valid_rows, valid_y = row_map(h, task, 'validation')
        assert np.array_equal(truth, valid_y)
        assert array_hash(np.asarray(valid_y, dtype=np.int64)) == coverage['audits'][task]['validation_labels_hash']
        assert array_hash(np.asarray(train_y, dtype=np.int64)) == coverage['audits'][task]['train_labels_hash']
        assert not set(train_rows).intersection(valid_rows)
        subject = np.array([key.removesuffix('/X') for key, _ in valid_rows])
        train_subject = np.array([key.removesuffix('/X') for key, _ in train_rows])
        names = list(dict.fromkeys(subject.tolist()))
        assert len(names) == SUBJECT_SPLITS[task][0]
        actual_full_ids = schedule['local_indices'][schedule['task'] == task_index].reshape(-1)
        unique_full_ids = np.unique(actual_full_ids)
        old_subjects = set(train_subject[calibration[task]])
        full_subjects = set(train_subject[unique_full_ids])
        per_subject = {}
        task_contrasts = {}
        for name in names:
            mask = subject == name
            stats = {m: class_statistics(truth[mask], prediction[mask], classes) for m, prediction in predictions.items()}
            counts = stats['initial'][0]
            remaining = all_stats['initial'][0]-counts
            assert np.all(remaining > 0), (task, name, remaining)
            contrasts = {}
            for treatment, baseline in pairs:
                total_delta = all_stats[treatment][1]-all_stats[baseline][1]
                delta = stats[treatment][1]-stats[baseline][1]
                contrasts[treatment+'_vs_'+baseline] = {
                    'contribution_to_task_gain_pp': float(np.mean(delta/all_stats['initial'][0])*100),
                    'subject_bacc_gain_pp': float(np.mean(delta/counts)*100) if np.all(counts > 0) else None,
                    'task_gain_without_subject_pp': float(np.mean((total_delta-delta)/remaining)*100)}
            tm = train_subject == name
            per_subject[name] = {'validation_rows': int(mask.sum()), 'validation_class_counts': counts.tolist(),
                                 'all_classes_present': bool(np.all(counts > 0)),
                                 'original_calibration_unique_rows': int((train_subject[calibration[task]] == name).sum()),
                                 'full_repair_unique_rows': int((train_subject[unique_full_ids] == name).sum()),
                                 'full_repair_exposures': int((train_subject[actual_full_ids] == name).sum()),
                                 'available_training_rows': int(tm.sum()), 'contrasts': contrasts}
        for treatment, baseline in pairs:
            key = treatment+'_vs_'+baseline
            gain = (replay[task][treatment]-replay[task][baseline])*100
            components = np.array([r['contrasts'][key]['contribution_to_task_gain_pp'] for r in per_subject.values()])
            assert abs(components.sum()-gain) < 1e-12
            loo = [r['contrasts'][key]['task_gain_without_subject_pp'] for r in per_subject.values()]
            within = [r['contrasts'][key]['subject_bacc_gain_pp'] for r in per_subject.values()]
            valid_within = [v for v in within if v is not None]
            task_contrasts[key] = {'task_gain_pp': gain,
                'subject_contribution_positive': int((components > 1e-14).sum()),
                'subject_contribution_negative': int((components < -1e-14).sum()),
                'subject_contribution_zero': int((np.abs(components) <= 1e-14).sum()),
                'leave_one_subject_out_range_pp': [min(loo), max(loo)],
                'every_subject_deletion_gain_positive': bool(min(loo) > 0),
                'equal_subject_mean_bacc_gain_pp_all_class_subjects': float(np.mean(valid_within)) if valid_within else None,
                'subjects_with_all_classes': len(valid_within)}
        records[task] = {'validation_rows': len(truth), 'subject_groups': len(names),
                         'validation_label_hash': array_hash(np.asarray(valid_y, dtype=np.int64)),
                         'train_validation_row_disjoint': True,
                         'full_training_and_validation_subject_groups_identical': set(train_subject) == set(subject),
                         'original512_subject_groups': len(old_subjects), 'sampled_full_repair_subject_groups': len(full_subjects),
                         'new_subject_groups_in_full_repair': sorted(full_subjects-old_subjects),
                         'per_subject': per_subject, 'contrasts': task_contrasts}
    summary = {}
    for treatment, baseline in pairs:
        key = treatment+'_vs_'+baseline
        summary[key] = {'subject_mappable_task_macro_gain_pp': float(np.mean([r['contrasts'][key]['task_gain_pp'] for r in records.values()])),
            'tasks_positive': sum(r['contrasts'][key]['task_gain_pp'] > 0 for r in records.values()),
            'tasks_every_subject_deletion_positive': sum(r['contrasts'][key]['every_subject_deletion_gain_positive'] for r in records.values()),
            'subject_groups_positive_contribution': sum(r['contrasts'][key]['subject_contribution_positive'] for r in records.values()),
            'subject_groups_negative_contribution': sum(r['contrasts'][key]['subject_contribution_negative'] for r in records.values()),
            'subject_groups_zero_contribution': sum(r['contrasts'][key]['subject_contribution_zero'] for r in records.values())}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'all16_task_bacc_replayed': replay, 'records': records, 'omitted_subject_mapping': omitted,
              'subject_mappable_tasks': len(records), 'task_local_subject_groups': sum(r['subject_groups'] for r in records.values()),
              'summary': summary, 'new_subject_groups_by_task': {t: r['new_subject_groups_in_full_repair'] for t, r in records.items()},
              'seconds': time.time()-started, 'new_model_inference': False, 'new_optimizer_updates': 0,
              'eeg_arrays_read': False, 'official_test_labels_or_arrays_read': False,
              'scope': 'Descriptive subject-group sensitivity of existing bank0 full-weight CE and fixed expert-span decomposition predictions on exposed validation. Validation uses every fifth retained row of the same first training subject groups for these11tasks; this is not unseen-subject evaluation. Labels and row mappings are verified against cached truth and original split hashes. Subject contributions sum exactly to task balanced-accuracy change; within-subject BAcc uses all-class subjects only and is a different estimator. Leave-one-subject-out ranges are sensitivity analyses, not confidence intervals. Task-local groups can overlap across datasets, so406groups are not claimed406unique people. No bootstrap independence assumption, new fitting, native inference or new-method claim.'}
    with (root/'analysis/coverage_subject_sensitivity.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'tasks': len(records), 'subject_groups': result['task_local_subject_groups'],
                      'summary': summary, 'new_subject_groups_by_task': result['new_subject_groups_by_task'], 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
