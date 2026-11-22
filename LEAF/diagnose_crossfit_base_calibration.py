"""Measure the unchanged common base on pilot calibration rows, CPU only."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import PrototypeClassifier, classification_metrics, load_leaf, sha256
from run_streaming_strict_history import TaskData


def main():
    root=Path('runs/crossfit_merge_candidate_v1')
    path=root/'base_calibration_plan.json'
    plan=json.loads(path.read_text())
    assert not (root/'base_calibration.json').exists()
    for p,h in plan['sha256'].items():
        assert sha256(Path(p))==h,p
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.manual_seed(0)
    source=json.loads(Path(plan['source_fit']).read_text())
    data=TaskData(plan,source,torch.device('cpu'))
    leaf=load_leaf(data.config,Path(source['base_checkpoint'])).eval().requires_grad_(False)
    started=time.monotonic()
    records,outputs={},{}
    with torch.no_grad():
        for task in plan['tasks']:
            data.arrive(task)
            x,y=data.train[task]
            classifier=PrototypeClassifier(leaf,*data.readouts[task]).eval()
            logits=torch.cat([classifier(x[i:i+32]) for i in range(0,len(y),32)])
            outputs[task]={'logits':logits,'truth':y}
            metrics=classification_metrics(logits,y)
            metrics['class_weighted_CE']=float((F.cross_entropy(logits,y,reduction='none')*data.weights[task][y]).mean())
            historical=Path(source['experts'][task]).parent/'run.json'
            assert sha256(historical)==plan['initial_validation_source_sha256'][task]
            record=json.loads(historical.read_text())
            assert record['source_checkpoint_sha256']==sha256(Path(source['base_checkpoint']))
            records[task]={'base_calibration':metrics,'recorded_initial_development_validation':record['source_val'][task],
                'calibration_minus_recorded_validation_BAcc_pp':100*(metrics['BAcc']-record['source_val'][task]['BAcc']),
                'calibration_audit':data.records[task],'validation_metric_source':str(historical)}
            print(json.dumps({'task':task,**records[task]}),flush=True)
    torch.save(outputs,root/'base_calibration_logits.pt')
    result={'plan_sha256':sha256(path),'records':records,'logits_sha256':sha256(root/'base_calibration_logits.pt'),
        'diagnostic_forward_examples':2048,'optimizer_updates':0,'device':'cpu','precision':'float32',
        'validation_EEG_read':False,'official_test_read':False,'elapsed_seconds':time.monotonic()-started,
        'scope':'Current CPU calibration measurements versus recorded initial GPU development validation of the same common checkpoint. No new validation inference. A score gap is not proof of exact checkpoint training membership; use training provenance separately. The current fold exclusion covers expert supervised fine-tuning only.'}
    with (root/'base_calibration.json').open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')


if __name__=='__main__':
    main()
