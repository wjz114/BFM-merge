"""Hash new pool EEG with the independent row reader, before any model fitting."""
import json
import time
from pathlib import Path

import h5py
import numpy as np

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import row_map, read_rows
from leaf_datasets.shared import DATA_FOLDER


def main():
    root=Path('runs/calibration_domain_pilot_v1')
    path=root/'data_spec.json'
    spec=json.loads(path.read_text())
    assert not (root/'data_audit.json').exists()
    for p,h in spec['sha256'].items():
        assert sha256(Path(p))==h,p
    metadata=json.loads((root/'pools.json').read_text())['audits']
    records={}
    total=0
    started=time.monotonic()
    with np.load(root/'pools.npz') as pools:
        for task in spec['tasks']:
            records[task]={}
            with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as handle:
                for name in ('training_calibration','development_calibration','development_evaluation'):
                    rows,y=row_map(handle,task,'calibration' if name=='training_calibration' else 'validation')
                    y=np.asarray(y,dtype=np.int64)
                    expected='full_train_labels_hash' if name=='training_calibration' else 'full_development_labels_hash'
                    assert array_hash(y)==metadata[task][expected]
                    ids=pools[f'{task}__{name}_ids']
                    assert array_hash(ids)==metadata[task][f'{name}_ids_hash']
                    x=read_rows(handle,[rows[int(i)] for i in ids])
                    truth=y[ids]
                    assert x.shape[0]==len(truth) and x.shape[1]==65 and np.isfinite(x).all()
                    if name!='development_evaluation':
                        assert array_hash(truth)==metadata[task]['calibration_labels_hash']
                    records[task][name]={'rows':len(ids),'ids_hash':array_hash(ids),'eeg_hash':array_hash(x),
                        'labels_hash':array_hash(truth),'shape':list(x.shape),
                        'class_counts':np.bincount(truth,minlength=metadata[task]['class_count']).tolist()}
                    total+=len(ids)
                    del x
    result={'status':'passed','spec_sha256':sha256(path),'pools_sha256':sha256(root/'pools.npz'),
        'tasks':records,'EEG_rows_read':total,'model_forward_examples':0,'optimizer_updates':0,
        'official_test_read':False,'elapsed_seconds':time.monotonic()-started,
        'scope':'Independent read_rows EEG hashes for each fixed row subset; future runners using selected_eeg must match these hashes. Development calibration and evaluation are complementary subsets of previously exposed development data.'}
    with (root/'data_audit.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2),flush=True)


if __name__=='__main__':
    main()
