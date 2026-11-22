"""Audit the original training pools and freeze their sampled example IDs."""
import json
import time
from pathlib import Path

import numpy as np

from aligned_support import array_hash
from coverage_support import shuffled_schedule, training_and_validation
from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    out = root/'inputs/full_control_coverage_data'
    out.mkdir(exist_ok=False)
    started = time.time()
    reference_path = root/'seed0/full_control_lr1e5_balanced_ce/run.json'
    reference = json.loads(reference_path.read_text())
    source = json.loads((Path(reference['plan']['source_fit'])/'run.json').read_text())
    joint = json.loads(Path('runs/poc16/seed0/joint/run.json').read_text())
    tasks = reference['datasets']
    calibration = np.load(reference['plan']['calibration'])
    original_schedule = np.load(reference['plan']['schedule'])
    task_order = original_schedule['task']
    replayed = shuffled_schedule(task_order, [len(calibration[t]) for t in tasks], 0)
    assert np.array_equal(replayed, original_schedule['local_indices'])
    pool_sizes = [joint['split_sizes'][t]['train'] for t in tasks]
    full_indices = shuffled_schedule(task_order, pool_sizes, 0)
    np.savez(out/'full_training_schedule.npz', task=task_order, local_indices=full_indices)
    audits = {}
    for k, task in enumerate(tasks):
        tx, ty, vx, vy = training_and_validation(task)
        assert len(ty) == pool_sizes[k] and len(vy) == joint['split_sizes'][task]['validation']
        calx, caly = tx[calibration[task]], ty[calibration[task]]
        assert array_hash(calx) == reference['dataset_audits'][task]['eeg_hash']
        nclasses = len(source['ce_class_weights'][task])
        assert np.array_equal(np.unique(ty), np.arange(nclasses))
        assert np.array_equal(np.unique(vy), np.arange(nclasses))
        counts = np.bincount(ty, minlength=nclasses)
        rows = full_indices[task_order == k]
        assert rows.shape == (100,128) and rows.min() >= 0 and rows.max() < len(ty)
        assert all(len(np.unique(row)) == 128 for row in rows)
        exposures = np.bincount(rows.flatten(), minlength=len(ty))
        unique = np.flatnonzero(exposures)
        np.save(out/f'{task}_exposure_counts.npy', exposures)
        audits[task] = {'train_size':len(ty),'validation_size':len(vy),'shape':list(tx.shape),
            'train_eeg_hash':array_hash(tx),'train_labels_hash':array_hash(ty),
            'validation_eeg_hash':array_hash(vx),'validation_labels_hash':array_hash(vy),
            'calibration_eeg_hash':array_hash(calx),'calibration_labels_hash':array_hash(caly),
            'train_class_counts':counts.tolist(),'calibration_class_counts':np.bincount(caly,minlength=nclasses).tolist(),
            'training_exposures':int(exposures.sum()),'unique_rows_sampled':len(unique),
            'sampled_fraction_of_full_pool':len(unique)/len(ty),
            'unique_original_calibration_rows_sampled':int(np.isin(unique,calibration[task]).sum()),
            'original_calibration_training_exposures':int(exposures[calibration[task]].sum()),
            'maximum_exposures_per_row':int(exposures.max()),'sampled_class_counts':np.bincount(ty[rows.flatten()],minlength=nclasses).tolist()}
        (out/'progress.json').write_text(json.dumps(audits,indent=2)+'\n')
        print(task,'pool',len(ty),'unique sampled',len(unique),flush=True)
        del tx,ty,vx,vy,calx,caly
    result = {'unix':time.time(),'reference':str(reference_path),'reference_sha256':sha256(reference_path),
        'original_schedule_sha256':sha256(Path(reference['plan']['schedule'])),
        'full_schedule_sha256':sha256(out/'full_training_schedule.npz'),
        'original512_schedule_exact_replay':True,'audits':audits,
        'source_sha256':{p:sha256(Path(p)) for p in ['prepare_coverage_control.py','coverage_support.py','aligned_support.py','load_datasets.py']},
        'seconds':time.time()-started,'test_data_read':False,'model_inference_performed':False,
        'scope':'Original expert-training partitions only. Increased unique labeled-data access at the same1600updates/204800 exposures; class weights reflect each full pool. Original512 EEG and schedule replay exactly. No validation/test examples enter training.'}
    (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed coverage audit',sum(a['unique_rows_sampled'] for a in audits.values()),'unique rows',flush=True)


if __name__ == '__main__':
    main()
