"""Measure head/full-weight prediction overlap from already saved native logits."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import classification_metrics, sha256


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    sources = {
        'head': root/'seed0/normalized_head_evaluation/run.json',
        'full_span': root/'seed0/full_pool_span_diagnostic/run.json',
        'full_repair': root/'seed0/full_control_coverage_full_balanced_ce/run.json',
        'teacher': root/'seed0/prototype_head_evaluation/run.json',
    }
    runs = {k: json.loads(p.read_text()) for k, p in sources.items()}
    hashes = {str(p): sha256(p) for p in sources.values()}
    tasks = runs['head']['datasets']
    assert tasks == runs['full_repair']['datasets'] == runs['teacher']['datasets']
    rows = {}
    for task in tasks:
        paths = {k: sources[k].parent/f'{task}_logits.pt' for k in ['head', 'full_span', 'teacher']}
        data = {k: torch.load(p, map_location='cpu', weights_only=True) for k, p in paths.items()}
        hashes.update({str(p): sha256(p) for p in paths.values()})
        y, n = data['head']['truth'], data['head']['n_calibration']
        assert all(torch.equal(v['truth'], y) and v['n_calibration'] == n for v in data.values())
        logits = {'initial': data['head']['logits']['initial'],
                  'full_repair': data['full_span']['logits']['full_update'],
                  'teacher': data['teacher']['original_expert_logits']}
        logits.update({c: data['head']['logits'][c] for c in ['prototype_contrast', 'full_feature_ridge', 'rotated_contrast_control']})
        assert torch.equal(logits['initial'].argmax(-1), data['full_span']['logits']['initial'].argmax(-1))
        rows[task] = {}
        for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
            truth = y[ids].long()
            preds = {k: v[ids].argmax(-1) for k, v in logits.items()}
            correct = {k: v == truth for k, v in preds.items()}
            counts = torch.bincount(truth, minlength=logits['initial'].shape[1]).double()
            assert (counts > 0).all()
            weights = 1/(len(counts)*counts[truth])
            scores = {k: float(weights[c].sum()) for k, c in correct.items()}
            expected = runs['full_repair']['average_validation' if split == 'validation' else 'average_calibration'][task]['BAcc']
            assert abs(scores['full_repair']-expected) < 1e-12
            assert all(abs(scores[c]-runs['head']['evaluations'][c][task][split]['BAcc']) < 1e-12
                       for c in ['initial', 'prototype_contrast', 'full_feature_ridge', 'rotated_contrast_control'])
            rows[task][split] = {}
            initial, full, teacher = (correct[k] for k in ['initial', 'full_repair', 'teacher'])
            for condition in ['prototype_contrast', 'full_feature_ridge', 'rotated_contrast_control']:
                head = correct[condition]
                masks = {
                    'head_correct_full_wrong': head & ~full,
                    'full_correct_head_wrong': full & ~head,
                    'both_correct': head & full,
                    'both_wrong': ~head & ~full,
                    'head_exclusive_new_fix': ~initial & ~full & head,
                    'head_spares_full_damage': initial & ~full & head,
                    'full_exclusive_new_fix': ~initial & full & ~head,
                    'full_spares_head_damage': initial & full & ~head,
                    'both_fix_initial_error': ~initial & full & head,
                    'teacher_correct_both_wrong': teacher & ~full & ~head,
                    'predictions_differ': preds[condition] != preds['full_repair'],
                }
                row = {k+'_pp': 100*float(weights[v].sum()) for k, v in masks.items()}
                row['head_minus_full_pp'] = 100*(scores[condition]-scores['full_repair'])
                assert abs(row['head_correct_full_wrong_pp']-row['full_correct_head_wrong_pp']-row['head_minus_full_pp']) < 1e-10
                assert abs(row['head_exclusive_new_fix_pp']+row['head_spares_full_damage_pp']-row['head_correct_full_wrong_pp']) < 1e-10
                row['native_bacc'] = {k: scores[k] for k in ['initial', 'full_repair', condition, 'teacher']}
                rows[task][split][condition] = row
    macro = {split: {c: {k: float(np.mean([rows[t][split][c][k] for t in tasks]))
                         for k in rows[tasks[0]][split][c] if k != 'native_bacc'}
                     for c in ['prototype_contrast', 'full_feature_ridge', 'rotated_contrast_control']}
             for split in ['calibration', 'validation']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'datasets': tasks, 'macro': macro, 'per_task': rows,
              'new_model_inference': False, 'fitting_performed': False, 'test_data_read': False,
              'scope': 'Descriptive class-balanced overlap of saved native predictions. Ground-truth masks identify exclusive corrections and damage, not a realizable selector, ensemble, combined checkpoint or recoverable accuracy bound. Exposed validation is diagnostic only; no candidate fitting or task weighting is based on these masks.'}
    output = root/'analysis/head_repair_complementarity.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
