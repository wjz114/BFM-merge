"""Pair class/count-matched calibration pools and a common development remainder."""
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
    root=Path('runs/calibration_domain_pilot_v1')
    path=root/'pool_spec.json'
    plan=json.loads(path.read_text())
    for p,h in plan['sha256'].items():
        assert sha256(Path(p))==h,p
    assert not (root/'pools.npz').exists() and not (root/'pools.json').exists()
    coverage=json.loads(Path(plan['coverage_audit']).read_text())['audits']
    arrays,audits={},{}
    with np.load(plan['original_calibration']) as original:
        for task in plan['pilot_tasks']:
            with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as handle:
                rows,y=row_map(handle,task,'calibration')
                vrows,vy=row_map(handle,task,'validation')
            y,vy=np.asarray(y,dtype=np.int64),np.asarray(vy,dtype=np.int64)
            assert array_hash(y)==coverage[task]['train_labels_hash']
            assert array_hash(vy)==coverage[task]['validation_labels_hash']
            assert not (set(rows)&set(vrows))
            old=original[task].astype(np.int64)
            assert len(old)==len(np.unique(old))==512
            classes=np.unique(y)
            assert np.array_equal(classes,np.arange(len(classes)))
            old_counts=np.bincount(y[old],minlength=len(classes))
            val_counts=np.bincount(vy,minlength=len(classes))
            quota=min(plan['maximum_calibration_rows']//len(classes),int(old_counts.min()),int((val_counts//2).min()))
            assert quota>0
            train_ids,dev_ids=[],[]
            index=plan['all_tasks'].index(task)
            for label in classes:
                train_rng=np.random.default_rng([plan['seed'],index,int(label),4861])
                dev_rng=np.random.default_rng([plan['seed'],index,int(label),4862])
                train_ids.extend(train_rng.permutation(old[y[old]==label])[:quota].tolist())
                dev_ids.extend(dev_rng.permutation(np.flatnonzero(vy==label))[:quota].tolist())
            train_ids,dev_ids=np.asarray(train_ids,dtype=np.int64),np.asarray(dev_ids,dtype=np.int64)
            eval_ids=np.setdiff1d(np.arange(len(vy),dtype=np.int64),dev_ids,assume_unique=True)
            assert np.array_equal(y[train_ids],vy[dev_ids])
            assert len(train_ids)==len(dev_ids)==len(classes)*quota
            assert len(np.unique(train_ids))==len(train_ids) and np.isin(train_ids,old).all()
            assert not np.intersect1d(dev_ids,eval_ids).size
            assert len(dev_ids)+len(eval_ids)==len(vy)
            assert np.array_equal(np.unique(vy[eval_ids]),classes)
            arrays[f'{task}__training_calibration_ids']=train_ids
            arrays[f'{task}__development_calibration_ids']=dev_ids
            arrays[f'{task}__development_evaluation_ids']=eval_ids
            arrays[f'{task}__paired_calibration_labels']=y[train_ids]
            audits[task]={'original_calibration_size':512,'new_calibration_size_per_domain':len(train_ids),
                'class_count':len(classes),'per_class_quota':quota,'calibration_labels_hash':array_hash(y[train_ids]),
                'training_calibration_ids_hash':array_hash(train_ids),'development_calibration_ids_hash':array_hash(dev_ids),
                'development_evaluation_ids_hash':array_hash(eval_ids),'development_evaluation_rows':len(eval_ids),
                'development_evaluation_class_counts':np.bincount(vy[eval_ids],minlength=len(classes)).tolist(),
                'full_train_labels_hash':array_hash(y),'full_development_labels_hash':array_hash(vy),
                'identical_label_sequence_between_domains':True,'class_weights':[1.]*len(classes),
                'training_calibration_subset_of_original512':True,'new_development_calibration_evaluation_disjoint':True}
    np.savez(root/'pools.npz',**arrays)
    result={'created_unix':time.time(),'spec_sha256':sha256(path),'pools_sha256':sha256(root/'pools.npz'),
        'audits':audits,'EEG_examples_read':0,'official_test_read':False,
        'scope':'Both calibration domains have identical class/count/label-slot sequences. At most half of every old-development class is allocated to calibration; all other development rows form the common evaluation subset. This pool has already been exposed and used for source checkpoint selection. It is not a new blind test.',
        'sampling_to_use':'Identical with-replacement index schedules across domains so small pools can support the fixed12-round,batch16 protocol. Equal CE exposures do not imply equal unique examples versus the historical512 baseline.',
        'historical_reference':'Re-evaluate frozen original512 single-full and source-soup checkpoints on this same remaining-development subset; do not compare new-subset scores directly to old full-validation scores.'}
    with (root/'pools.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps({t:{k:a[k] for k in ['new_calibration_size_per_domain','per_class_quota','development_evaluation_rows']}
                      for t,a in audits.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
