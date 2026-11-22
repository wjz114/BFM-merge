"""Inspect task-gradient agreement from existing original512 gradient caches."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def summarize(gram, tasks):
    norms = gram.diag().sqrt()
    cosine = gram/(norms[:, None]*norms[None, :])
    pair_mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    pairs = cosine[pair_mask]
    average_inner = gram.mean(1)
    average_norm_sq = float(gram.mean())
    assert average_norm_sq > 0
    return {'gram': gram.tolist(), 'cosine': cosine.tolist(),
            'gradient_norms': dict(zip(tasks, norms.tolist())),
            'negative_pairs': int((pairs < 0).sum()), 'total_pairs': len(pairs),
            'mean_pair_cosine': float(pairs.mean()), 'min_pair_cosine': float(pairs.min()),
            'max_pair_cosine': float(pairs.max()),
            'uniform_mean_gradient_inner_product_per_task': dict(zip(tasks, average_inner.tolist())),
            'tasks_with_negative_mean_gradient_inner_product': [t for t, v in zip(tasks, average_inner) if v < 0],
            'between_task_variance_over_mean_gradient_norm_sq': float(gram.diag().mean()/average_norm_sq-1),
            'mean_gradient_norm': average_norm_sq**.5}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    folders = {'initial': 'local_gradient_cache',
               'old512_full_weight_step800': 'trajectory_gradients_full_weight_old512_step800',
               'old512_full_weight_step1600': 'trajectory_gradients_full_weight_old512_step1600'}
    stages, hashes, initial = {}, {}, None
    for stage, folder in folders.items():
        path = root/f'seed0/{folder}/run.json'
        cache = json.loads(path.read_text())
        hashes[str(path)] = sha256(path)
        if initial is None:
            initial = cache
        assert cache['datasets'] == initial['datasets'] and cache['parameter_names'] == initial['parameter_names']
        tasks = cache['datasets']
        data = []
        for task in tasks:
            row = cache['records'][task]
            for key in ['eeg_hash', 'labels_hash', 'instruction_hash', 'prototypes_hash']:
                assert row[key] == initial['records'][task][key]
            p = Path(row['path'])
            assert sha256(p) == row['sha256']
            hashes[str(p)] = row['sha256']
            data.append(torch.load(p, map_location='cpu', weights_only=True))
        gram = {c: torch.zeros((len(tasks), len(tasks)), dtype=torch.float64) for c in ['tower', 'qformer', 'head']}
        for name in cache['parameter_names']:
            values = torch.stack([g[name].flatten().double() for g in data])
            product = values@values.T
            component = name.split('.')[0]
            assert component in ['tower', 'qformer']
            gram[component] += product
            if name.startswith('qformer.output_embed.2.'):
                gram['head'] += product
        gram['all'] = gram['tower']+gram['qformer']
        errors = {}
        for component in ['tower', 'qformer']:
            calculated = gram[component].diag().sqrt()
            reference = torch.tensor([cache['records'][t]['component_gradient_norm'][component] for t in tasks], dtype=torch.float64)
            error = float(((calculated-reference).abs()/reference).max())
            assert error < 1e-5, (stage, component, error)
            errors[component] = error
        stages[stage] = {'components': {c: summarize(g, tasks) for c, g in gram.items()},
                         'stored_gradient_norm_max_relative_errors': errors,
                         'mean_weighted_calibration_ce': float(np.mean([cache['records'][t]['weighted_ce'] for t in tasks]))}
        print(stage, {c: {'negative_pairs': v['negative_pairs'], 'mean_pair_cosine': v['mean_pair_cosine'],
                         'mean_gradient_harmed_tasks': v['tasks_with_negative_mean_gradient_inner_product']} for c, v in stages[stage]['components'].items()}, flush=True)
        del data
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'datasets': initial['datasets'], 'stages': stages, 'seconds': time.time()-started,
              'new_model_inference': False, 'new_backpropagation': False, 'fitting_performed': False,
              'validation_used_in_statistics': False, 'test_data_read': False,
              'scope': 'Original512 rows/task and their empirical class weights only, at initialization and two existing old512 full-weight trajectory checkpoints. Full383-parameter gradients, not just head gradients. Negative inner products describe infinitesimal Euclidean CE disagreement; Adam, minibatch noise, finite steps and generalization can differ. Between-task variance concerns cached task means only. No full-pool or independent-confirmation claim, no candidate trained or selected.'}
    output = root/'analysis/parameter_gradient_conflicts.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
