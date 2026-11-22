"""Describe fixed head-calibration predictions using previously saved native logits."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--expert-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.time()
    torch.set_num_threads(4)
    evaluation = json.loads(args.evaluation.read_text())
    expert_evaluation = json.loads(args.expert_evaluation.read_text())
    tasks = evaluation['datasets']
    assert tasks == expert_evaluation['datasets']
    rows, source_hashes = {}, {str(p): sha256(p) for p in [args.evaluation, args.expert_evaluation]}
    for task in tasks:
        ep = args.evaluation.parent / f'{task}_logits.pt'
        tp = args.expert_evaluation.parent / f'{task}_logits.pt'
        source_hashes.update({str(p): sha256(p) for p in [ep, tp]})
        data = torch.load(ep, map_location='cpu', weights_only=True)
        teacher = torch.load(tp, map_location='cpu', weights_only=True)
        assert torch.equal(data['truth'], teacher['truth'])
        assert data['n_calibration'] == teacher['n_calibration']
        assert torch.equal(data['logits']['initial'].argmax(-1), teacher['logits']['initial'].argmax(-1))
        rows[task] = {}
        n = data['n_calibration']
        for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
            truth = data['truth'][ids].long()
            tl = teacher['original_expert_logits'][ids].double()
            teacher_pred = tl.argmax(-1)
            initial_pred = data['logits']['initial'][ids].argmax(-1)
            counts = torch.bincount(truth, minlength=tl.shape[1]).double()
            assert (counts > 0).all()
            weights = 1 / (len(counts) * counts[truth])
            teacher_correct = teacher_pred == truth
            initial_correct = initial_pred == truth
            recoverable = (~initial_correct) & teacher_correct
            rows[task][split] = {}
            for condition, logits in data['logits'].items():
                logits = logits[ids].double()
                prediction = logits.argmax(-1)
                correct = prediction == truth
                centered_error = (logits-logits.mean(-1, keepdim=True))-(tl-tl.mean(-1, keepdim=True))
                fixes = float(weights[(~initial_correct) & correct].sum())
                breaks = float(weights[initial_correct & (~correct)].sum())
                recovered_mass = float(weights[recoverable & correct].sum())
                recoverable_mass = float(weights[recoverable].sum())
                record = {
                    'bacc': float(weights[correct].sum()),
                    'teacher_bacc': float(weights[teacher_correct].sum()),
                    'balanced_teacher_agreement': float(weights[prediction == teacher_pred].sum()),
                    'balanced_centered_teacher_logit_mse': float((weights * centered_error.square().mean(-1)).sum()),
                    'balanced_changed_prediction_mass': float(weights[prediction != initial_pred].sum()),
                    'fixed_initial_error_mass': fixes,
                    'broke_initial_correct_mass': breaks,
                    'net_bacc_gain_pp': 100 * (fixes-breaks),
                    'initial_wrong_teacher_correct_mass': recoverable_mass,
                    'recovered_teacher_correct_mass': recovered_mass,
                    'recovery_fraction': recovered_mass / recoverable_mass if recoverable_mass else None,
                }
                assert abs(record['bacc']-evaluation['evaluations'][condition][task][split]['BAcc']) < 1e-12
                rows[task][split][condition] = record
    macro = {}
    for split in ['calibration', 'validation']:
        macro[split] = {}
        for condition in evaluation['summaries'][split]:
            keys = rows[tasks[0]][split][condition]
            macro[split][condition] = {
                k: float(np.mean([rows[t][split][condition][k] for t in tasks
                                  if rows[t][split][condition][k] is not None]))
                for k in keys}
    result = {
        'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
        'source_sha256': source_hashes, 'datasets': tasks, 'per_task': rows, 'macro': macro,
        'seconds': time.time()-started, 'model_inference_performed': False,
        'fitting_performed': False, 'test_data_read': False,
        'scope': 'Post-hoc descriptive diagnostic of all fixed conditions, both splits. '
                 'All scores use native saved logits; teacher validation predictions are used only here, '
                 'never in fitting. Class-balanced prediction transitions exactly decompose BAcc changes. '
                 'Recovery fraction is a mean of task fractions, not an overall fraction or capacity bound. '
                 'Cosine-logit MSE differs from the raw-output ridge objective and cannot identify a unique failure mechanism.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
