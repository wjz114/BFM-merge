"""Matched-batch native source replay after the Cho source-equivalence gate failed."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT,PrototypeClassifier,load_leaf,prepare_text,sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config,load_yaml
from qformer_residual_rotation import rotate_qformer_state


@torch.no_grad()
def main():
    root=Path('runs/paper_merge_v2');output=root/'analysis/query_rotation_numerics.json';assert not output.exists()
    plan=json.loads((root/'inputs/query_rotation_merges_plan_v2.json').read_text())
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha256(Path(p))==h,p
    start=time.time();device=configure(0,0);torch.cuda.set_per_process_memory_fraction(.16,device)
    task='MI_Cho2017';source=json.loads(Path(plan['source_fit']).read_text());targets=json.loads(Path(plan['target_manifest']).read_text())
    ids=np.load(plan['calibration'])[task]
    with h5py.File(DATA_FOLDER/f'{task}.h5','r',locking=False) as handle:
        rows,y=row_map(handle,task,'calibration');x=torch.from_numpy(selected_eeg(handle,[rows[int(i)] for i in ids]));truth=torch.tensor(y[ids],dtype=torch.int64)
    assert array_hash(x)==plan['original512_eeg_hash'][task]
    ref=targets['records'][task];assert sha256(Path(ref['target_path']))==ref['target_sha256']
    cached=torch.load(ref['target_path'],map_location='cpu',weights_only=True);pos=np.searchsorted(cached['pool_indices'].numpy(),ids)
    assert np.array_equal(cached['pool_indices'][pos].numpy(),ids)
    config=build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'));_,_,inst,proto=prepare_text(config,task)
    assert sha256(Path(source['experts'][task]))==plan['expert_sha256'][task]
    state=torch.load(source['experts'][task],map_location='cpu',weights_only=True)
    model=load_leaf(config,Path(source['experts'][task])).to(device).eval().requires_grad_(False)
    rotations={'canonical':torch.eye(256,dtype=torch.float64),
               'response':torch.load(plan['response_rotations'],map_location='cpu',weights_only=True)[task]['fit_all'],
               'parameter':torch.load(plan['parameter_rotations'],map_location='cpu',weights_only=True)[task]}
    logits={}
    for kind in ['original',*rotations]:
        model.load_state_dict(state if kind=='original' else rotate_qformer_state(state,rotations[kind]),strict=True)
        wrapper=PrototypeClassifier(model,inst,proto).to(device).eval()
        logits[kind]=torch.cat([wrapper(x[i:i+64].to(device)).cpu() for i in range(0,len(x),64)])
    result={'unix':time.time(),'task':task,'code_sha256':sha256(Path(__file__)),
            'against_cached':{k:compare(cached['teacher_logits'][pos],v,truth) for k,v in logits.items()},
            'against_matched_native':{k:compare(logits['original'],logits[k],truth) for k in rotations},
            'forward_examples':2048,'optimizer_updates':0,'test_data_read':False,'candidate_validation_examples':0,
            'seconds':time.time()-start}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
