"""Verify source subsets, unchanged source training settings and actual batch counts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert not args.output.exists()
    for p, h in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == h, p
    folds = json.loads(Path(plan['fold_audit']).read_text())['audits']
    coverage = json.loads(Path(plan['coverage_audit']).read_text())['audits']
    merge = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=True)
    fixed = set(base) - set(plan['active'])
    checks = {}
    new_exposures = 0
    old_exposures = 0
    for task in plan['tasks']:
        original = Path(merge['experts'][task]).parent
        reference = json.loads((original / 'run.json').read_text())
        n, b, steps = reference['split_sizes'][task]['train'], reference['batch_size'], reference['actual_optimizer_steps']
        passes, remaining = divmod(steps, (n + b - 1) // b)
        old_exposures += passes * n + remaining * b
        checks[task] = []
        for fold in (0, 1):
            folder = Path(plan['project_source_root']) / f'{task}_fold{fold}'
            run = json.loads((folder / 'run.json').read_text())
            completion = json.loads((folder / 'subset_completion.json').read_text())
            data_path = Path(plan['audit_root']) / f'{task}_fold{fold}_data.json'
            data = json.loads(data_path.read_text())
            assert completion['status'] == 'completed' and completion['plan_sha256'] == sha256(args.plan)
            for key, p in [('run_sha256', folder / 'run.json'), ('checkpoint_sha256', folder / 'best.ckpt'),
                           ('data_audit_sha256', data_path), ('batch_size_trace_sha256', folder / 'training_batch_sizes.npy')]:
                assert completion[key] == sha256(p)
            for key in ['source_checkpoint_sha256', 'datasets', 'labels', 'instruction', 'conditioning',
                        'seed', 'mode', 'rounds', 'total_steps', 'warmup_steps', 'eval_every_rounds',
                        'lr_tokenizer', 'lr_rest', 'weight_decay', 'lr_schedule', 'precision',
                        'batch_size', 'cudnn_benchmark', 'batchnorm', 'schedule', 'selection',
                        'model_config', 'training_config', 'actual_optimizer_steps']:
                assert run[key] == reference[key], (task, fold, key)
            assert run['split_sizes'][task]['train'] == folds[task]['fold_sizes'][fold]
            assert run['split_sizes'][task]['validation'] == reference['split_sizes'][task]['validation']
            assert data['training_index_hash'] == folds[task]['training_index_hashes'][fold]
            assert data['training_labels_hash'] == folds[task]['training_label_hashes'][fold]
            assert data['validation_eeg_hash'] == coverage[task]['validation_eeg_hash']
            assert data['validation_labels_hash'] == coverage[task]['validation_labels_hash']
            assert not data['official_test_read'] and not completion['official_test_read']
            n = data['training_rows']
            sizes = np.load(folder / 'training_batch_sizes.npy')
            cycle = [b] * (n // b) + ([n % b] if n % b else [])
            expected = np.resize(np.asarray(cycle, dtype=np.int64), steps)
            assert np.array_equal(sizes, expected)
            assert int(sizes.sum()) == completion['observed_training_example_exposures'] == completion['reconstructed_training_example_exposures']
            new_exposures += int(sizes.sum())
            state = torch.load(folder / 'best.ckpt', map_location='cpu', weights_only=True)
            assert state.keys() == base.keys()
            assert all(torch.equal(state[k], base[k]) for k in fixed)
            assert all(torch.isfinite(state[k]).all() for k in plan['active'])
            with np.load(original / 'text_readout.npz') as a, np.load(folder / 'text_readout.npz') as z:
                assert set(a.files) == set(z.files)
                assert all(np.array_equal(a[k], z[k]) for k in a.files)
            checks[task].append({'fold': fold, 'training_rows': n,
                'observed_training_examples': int(sizes.sum()), 'best_round': run['best_round'],
                'best_validation_BAcc': run['best_val_macro_BAcc'],
                'checkpoint': str(folder / 'best.ckpt'), 'checkpoint_sha256': completion['checkpoint_sha256']})
    assert len(checks) == 4 and new_exposures == 2952020
    result = {'status': 'passed', 'plan_sha256': sha256(args.plan), 'tasks': checks,
        'new_source_experts': 8, 'new_source_optimizer_steps': 24000,
        'new_source_observed_training_examples': new_exposures,
        'existing_full_source_experts': 4, 'existing_source_optimizer_steps': 12000,
        'existing_source_reconstructed_training_examples': old_exposures,
        'total_source_optimizer_steps': 36000, 'total_source_training_examples': old_exposures + new_exposures,
        'original_source_protocol_matched_except_training_subset': True,
        'all_source_checkpoint_readouts_and_fixed_parameters_match': True,
        'folds_disjoint': True, 'official_test_read_by_new_sources': False,
        'scope': 'Trial-row folds, not subject folds. Development validation selects all source checkpoints by the existing rule. Existing full-source exposure counts are reconstructed; new counts are observed from batch traces.'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
