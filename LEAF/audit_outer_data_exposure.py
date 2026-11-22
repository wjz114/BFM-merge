"""Audit existing held-out metadata and historical result exposure; no EEG inference."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import h5py


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('runs/paper_merge_v2')
    names=['MI_Weibo2014','MI_Dreyer2023']
    hits={name:[] for name in names}
    scanned=[]
    for path in sorted(Path('runs').rglob('results*.csv')):
        with path.open() as f:rows=list(csv.DictReader(f))
        scanned.append({'path':str(path),'sha256':sha(path)})
        for name in names:
            matched=[row for row in rows if row.get('dataset')==name and row.get('BAcc')]
            if matched:
                assert all(math.isfinite(float(row['BAcc'])) for row in matched)
                hits[name].append({'path':str(path),'sha256':sha(path),
                    'metric_rows':len(matched),'models':[row['model'] for row in matched],
                    'splits':sorted({row['split'] for row in matched}),
                    'sample_counts':sorted({int(row['n']) for row in matched})})
    metadata={}
    for name in names:
        path=Path('/home/jingzhou/data/LEAF/Downstream')/(name+'.h5')
        stat=path.stat()
        with h5py.File(path,'r',locking=False) as f:
            metadata[name]={'path':str(path),'file_size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                'root_attributes':{k:str(v) for k,v in f.attrs.items()},
                'groups':{key:{'attributes':{k:str(v) for k,v in group.attrs.items()},
                    'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in group.items()}}
                    for key,group in f.items()}}
    sources=[Path('leaf_datasets/MI_Weibo2014.py'),Path('leaf_datasets/MI_Dreyer2023.py'),
             Path('zero_shot_heldout.py'),root/'analysis/outer_evaluation_provenance_status.json']
    weibo_source=Path('/media/datasets/EEG_Dataset/MI_Weibo2014')
    assert not weibo_source.exists()
    assert hits['MI_Weibo2014']
    result={'unix':time.time(),'code_sha256':sha(Path(__file__)),
        'source_sha256':{str(p):sha(p) for p in sources},'scanned_result_csv':scanned,
        'historical_metric_exposure':hits,'metadata_only_observations':metadata,
        'primary_documentation':['https://leaf-bci.github.io/docs/datasets/',
                                 'https://leaf-bci.github.io/docs/datasets/held-out.html'],
        'decisions':{
            'MI_Weibo2014':{'fresh_independent_confirmation_eligible':False,
                'reason':'Prior result CSVs already contain measured outcomes for all10subjects across multiple checkpoints and seeds. Exclusion from instruction tuning is different from being unexposed in this research process.',
                'label_mapping_source':'Repository builder selects raw event codes1/2 and subtracts1; public documentation and evaluator identify0=Left,1=Right.',
                'raw_reproduction_available_at_builder_path':False},
            'MI_Dreyer2023':{'fresh_independent_confirmation_eligible':False,
                'reason':'The local builder/evaluator explicitly mark0=Left,1=Right as assumed because the delivered HDF5 lacks a label legend. Public held-out status does not resolve this local mapping.',
                'metric_exposure_in_scanned_results':bool(hits['MI_Dreyer2023']),
                'exposure_search_scope':'Only the listed runs/**/results*.csv files; no claim about all external histories.'}},
        'model_inference_performed':False,'eeg_arrays_materialized':False,
        'scope':'Read existing result CSVs, source code and HDF5 structure/attributes only. No new predictions, EEG-array loading, label-order guessing, model selection or modification of previous provenance records. Instruction-tuning exclusion does not prove absence from self-supervised pretraining.'}
    out=root/'analysis/outer_data_exposure_audit.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(out),'sha256':sha(out),'scanned_csv':len(scanned),
                     'historical_metric_rows':{k:sum(r['metric_rows'] for r in v) for k,v in hits.items()}},indent=2))


if __name__=='__main__':main()
