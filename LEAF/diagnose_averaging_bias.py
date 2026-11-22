"""Calibration-only constant logit-offset diagnostic for checkpoint averaging."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from d_downstream_qformer_ce import classification_metrics, sha256


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/averaging_bias_diagnostic_design.json'
    design = json.loads(design_path.read_text())
    source = Path(design['source'])
    assert sha256(source/'run.json') == design['source_sha256']
    original = json.loads((source/'run.json').read_text())
    tasks = list(original['comparisons'])
    records = {}
    for task in tasks:
        data = torch.load(source/f'{task}_logits.pt', map_location='cpu', weights_only=True)
        n, y = data['n_calibration'], data['truth']
        w, e = [data['logits'][k].double() for k in ['weight_mean', 'logit_mean']]
        residual = (w-w.mean(-1, keepdim=True))-(e-e.mean(-1, keepdim=True))
        bias = residual[:n].mean(0)
        conditions = {'weight_mean': w, 'logit_mean': e, 'logit_mean_plus_bias': e+bias, 'weight_mean_minus_bias': w-bias}
        row = {'bias': bias.tolist(), 'artifact_sha256': sha256(source/f'{task}_logits.pt'), 'splits': {}}
        for split, ids in [('calibration', slice(0,n)), ('validation', slice(n,None))]:
            r = residual[ids]
            report = {'residual_mean': r.mean(0).tolist(),
                      'residual_rms_before': float(r.square().mean().sqrt()),
                      'residual_rms_after_calibration_bias': float((r-bias).square().mean().sqrt()), 'conditions': {}}
            for name, logits in conditions.items():
                values, truth = logits[ids], y[ids]
                metrics = classification_metrics(values.float(), truth)
                if name in ['weight_mean', 'logit_mean']:
                    assert abs(metrics['BAcc']-original['evaluations'][name][task][split]['BAcc']) < 1e-12
                cm = confusion_matrix(truth.numpy(), values.argmax(-1).numpy(), labels=range(w.shape[1]))
                report['conditions'][name] = {'metrics': metrics, 'confusion_matrix': cm.tolist(),
                    'class_recall': (cm.diagonal()/cm.sum(1)).tolist(),
                    'prediction_agreement_with_weight_mean': float((values.argmax(-1) == w[ids].argmax(-1)).double().mean())}
            row['splits'][split] = report
        records[task] = row
    summaries = {split: {name: float(np.mean([records[t]['splits'][split]['conditions'][name]['metrics']['BAcc'] for t in tasks]))
                        for name in conditions} for split in ['calibration', 'validation']}
    result = {'design': design, 'design_sha256': sha256(design_path), 'code_sha256': sha256(Path(__file__)),
              'records': records, 'summaries': summaries, 'new_model_inference': False, 'official_test_used': False}
    (root/'analysis/averaging_bias_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))
    print('Workload', json.dumps(records['Workload'], indent=2))


if __name__ == '__main__':
    main()
