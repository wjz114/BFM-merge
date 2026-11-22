"""Paired full-validation RI analysis under the frozen pre-adaptation groups."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def mean(values):
    return float(np.mean(list(values)))


def main():
    root = Path('runs/paper_merge_v2')
    original_dir, adapted_dir = root/'seed0/shared_mi_transfer', root/'seed0/shared_mi_transfer_ri'
    original = json.loads((original_dir/'run.json').read_text())
    adapted = json.loads((adapted_dir/'run.json').read_text())
    retention = json.loads((root/'seed0/ri_full_retention/run.json').read_text())
    tasks = original['tasks']
    assert adapted['tasks'] == tasks and len(tasks) == 6
    assert len(retention['records']) == 16
    assert original['source_calibration_sha256'] == adapted['source_calibration_sha256']
    fixed_checks = {}
    for task in tasks:
        before = np.load(original_dir/f'{task}_logits.npz')
        after = np.load(adapted_dir/f'{task}_logits.npz')
        assert np.array_equal(before['truth'], after['truth'])
        assert np.array_equal(before['calibration_indices'], after['calibration_indices'])
        assert before['n_calibration'] == after['n_calibration']
        errors = {}
        for model in ['base', 'prior_candidate']:
            assert original['checkpoint_hashes'][model] == adapted['checkpoint_hashes'][model]
            errors[model] = float(np.abs(before[model]-after[model]).max())
            assert errors[model] <= 1e-5, (task, model, errors[model])
        full_own = torch.load(root/f'seed0/ri_full_retention/{task}_logits.pt',
                              map_location='cpu', weights_only=True)
        n = int(before['n_calibration'])
        assert np.array_equal(full_own['truth'].numpy(), before['truth'][n:])
        for model, values in [('original', before[task][n:]), ('adapted', after[task][n:])]:
            errors[f'own_{model}'] = float(np.abs(full_own[model].numpy()-values).max())
            assert errors[f'own_{model}'] <= 1e-5
        fixed_checks[task] = errors
    pairs = []
    for target in tasks:
        for expert in tasks:
            if expert == target:
                continue
            calibration_gain = original['domains'][target]['splits']['calibration'][expert]['bacc_gain_vs_base']
            category = 'helpful' if calibration_gain > 0 else 'harmful' if calibration_gain < 0 else 'neutral'
            row = {'expert': expert, 'target': target, 'category_from_original_calibration': category,
                   'original_calibration_gain_pp': 100*calibration_gain, 'splits': {}}
            for split in ['calibration', 'validation']:
                before = original['domains'][target]['splits'][split][expert]
                after = adapted['domains'][target]['splits'][split][expert]
                row['splits'][split] = {
                    'before_bacc_gain_vs_base_pp': 100*before['bacc_gain_vs_base'],
                    'after_bacc_gain_vs_base_pp': 100*after['bacc_gain_vs_base'],
                    'adaptation_bacc_change_pp': 100*(after['BAcc']-before['BAcc']),
                    'before_balanced_help_pp': 100*before['balanced_help_rate'],
                    'after_balanced_help_pp': 100*after['balanced_help_rate'],
                    'before_balanced_harm_pp': 100*before['balanced_harm_rate'],
                    'after_balanced_harm_pp': 100*after['balanced_harm_rate'],
                }
            pairs.append(row)
    summary = {}
    for category in ['all', 'helpful', 'harmful', 'neutral']:
        selected = [p for p in pairs if category == 'all' or p['category_from_original_calibration'] == category]
        summary[category] = {'count': len(selected), 'splits': {}}
        if selected:
            for split in ['calibration', 'validation']:
                summary[category]['splits'][split] = {
                    key: mean(p['splits'][split][key] for p in selected)
                    for key in selected[0]['splits'][split]
                }
    own = retention['records']
    result = {'full_own_retention': own, 'own_macro_gain_pp': retention['macro_gain_pp'],
              'own_leave_one_task_out_gain_pp': {
                  task: mean(row['gain_pp'] for other, row in own.items() if other != task) for task in own},
              'mi_pairs': pairs, 'mi_category_summary': summary, 'repeated_forward_checks': fixed_checks,
              'scope': 'Full development-exposed validation, paired frozen RI and original experts. No official test or checkpoint selection.',
              'interpretation': 'Category membership uses pre-adaptation calibration only. Expert/domain pairs are dependent; no independent significance test. Cross-domain restoration and retention do not establish better final merged performance.',
              'source_run_sha256': {str(folder): sha256(folder/'run.json') for folder in [
                  original_dir, adapted_dir, root/'seed0/ri_full_retention']},
              'code_sha256': sha256(Path(__file__))}
    (root/'analysis/ri_full_validation_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'own_macro_gain_pp': result['own_macro_gain_pp'], 'mi_category_summary': summary}, indent=2))


if __name__ == '__main__':
    main()
