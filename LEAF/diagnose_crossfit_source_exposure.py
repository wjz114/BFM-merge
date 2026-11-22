"""Measure paired in/out-of-source-training behavior on the fixed calibration rows.

CPU-only diagnostic. Row-wise source selection is used only for measurement, not
as an inference method. No validation or official-test EEG is loaded.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier, classification_metrics, load_leaf, sha256
from run_streaming_strict_history import TaskData


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    args=parser.parse_args()
    root=Path('runs/crossfit_merge_candidate_v1')
    plan_path=root/'source_exposure_diagnostic_plan.json'
    plan=json.loads(plan_path.read_text())
    assert args.task in plan['tasks']
    for p,h in plan['sha256'].items():
        assert sha256(Path(p))==h,p
    output=root/'source_exposure_diagnostic'/args.task
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.manual_seed(0)
    started=time.monotonic()
    device=torch.device('cpu')
    source=json.loads(Path(plan['source_fit']).read_text())
    data=TaskData(plan,source,device)
    data.arrive(args.task)
    x,y=data.train[args.task]
    with np.load(plan['folds']) as archive:
        fold=torch.from_numpy(archive[f'{args.task}__calibration_fold'].astype(np.int64))
    fold_audit=json.loads(Path(plan['fold_audit']).read_text())['audits'][args.task]
    assert array_hash(data.indices[args.task].astype(np.int64))==fold_audit['calibration_ids_hash']
    assert array_hash(fold.numpy().astype(np.int8))==fold_audit['calibration_fold_hash']
    paths=[Path(source['experts'][args.task])]
    identities={str(paths[0]):sha256(paths[0])}
    assert identities[str(paths[0])]==plan['full_expert_sha256'][args.task]
    for k in (0,1):
        folder=root/'half_sources'/f'{args.task}_fold{k}'
        completion=json.loads((folder/'subset_completion.json').read_text())
        assert completion['status']=='completed' and completion['plan_sha256']==sha256(root/'source_scientific_plan.json')
        path=folder/'best.ckpt'
        assert sha256(path)==completion['checkpoint_sha256']
        paths.append(path)
        identities[str(path)]=completion['checkpoint_sha256']
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).eval().requires_grad_(False)
    classifier=PrototypeClassifier(leaf,*data.readouts[args.task]).eval()
    outputs=[]
    with torch.no_grad():
        for path in paths:
            leaf.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            outputs.append(torch.cat([classifier(x[i:i+32]).cpu() for i in range(0,len(y),32)]))
    halves=torch.stack(outputs[1:])
    rows=torch.arange(len(y))
    cases={'full_training_source':outputs[0],'in_fold_source':halves[fold,rows],
           'out_of_fold_source':halves[1-fold,rows]}
    metrics,losses={},{}
    for name,logits in cases.items():
        losses[name]=F.cross_entropy(logits,y,reduction='none')*data.weights[args.task][y]
        metrics[name]=classification_metrics(logits,y)
        metrics[name]['class_weighted_CE']=float(losses[name].mean())
    gap=losses['out_of_fold_source']-losses['in_fold_source']
    result={'task':args.task,'plan_sha256':sha256(plan_path),'source_sha256':identities,
        'source_training_rows_per_fold':fold_audit['fold_sizes'],
        'calibration_rows':len(y),'calibration_eeg_hash':data.records[args.task]['calibration_eeg_hash'],
        'calibration_labels_hash':data.records[args.task]['calibration_labels_hash'],
        'calibration_fold_sizes':torch.bincount(fold,minlength=2).tolist(),'metrics':metrics,
        'out_minus_in_class_weighted_CE':float(gap.mean()),
        'out_minus_in_BAcc_pp':100*(metrics['out_of_fold_source']['BAcc']-metrics['in_fold_source']['BAcc']),
        'per_class_out_minus_in_weighted_CE':{str(k):float(gap[y==k].mean()) for k in y.unique().tolist()},
        'source_exclusion_definition':'Out-of-fold means excluded from supervised fine-tuning of that task source expert. This is not exclusion from foundation pretraining or the adaptive merge coefficient history.',
        'device':'cpu','precision':'float32','diagnostic_forward_examples':3*len(y),
        'optimizer_updates':0,'validation_EEG_read':False,'official_test_read':False,
        'row_wise_routing_used_only_for_diagnostic':True,'pilot_algorithm_or_settings_changed':False,
        'elapsed_seconds':time.monotonic()-started}
    torch.save({'full_logits':outputs[0],'half0_logits':outputs[1],'half1_logits':outputs[2],
                'truth':y,'training_fold':fold},output/'logits.pt')
    result['logits_sha256']=sha256(output/'logits.pt')
    with (output/'comparison.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')
    print(json.dumps({k:result[k] for k in ['task','metrics','out_minus_in_class_weighted_CE',
        'out_minus_in_BAcc_pp','diagnostic_forward_examples','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':
    main()
