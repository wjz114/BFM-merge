"""Read existing checkpoints/gradients to audit the historical tokenizer LR split."""
import json
import math
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def component(name):
    if name.startswith('tower.tokenizer.'):
        return 'tokenizer'
    if name.startswith('tower.'):
        return 'tower_rest'
    if name.startswith('qformer.output_embed.2.'):
        return 'readout'
    assert name.startswith('qformer.')
    return 'qformer_rest'


@torch.no_grad()
def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    sources = {}

    def read(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return json.loads(path.read_text())

    fit = read(root/'seed0/original_fp32_tensor_balanced_ce/run.json')
    joint_plan = read(root/'inputs/bank_joint_fp32_validation_plan.json')
    joint_run = read('runs/poc16/seed0/joint/run.json')
    cache = read(root/'seed0/local_gradient_cache/run.json')
    names = fit['groups']
    assert names == cache['parameter_names'] and len(names) == 383
    specs = {
        'base': (fit['base_checkpoint'], fit['base_checkpoint_sha256']),
        'initial_merger': (str(root/'seed0/original_fp32_tensor_balanced_avg/merged.ckpt'),
                           'abd0aaf09d606387f12395f6ebc7a9de54c3178699f951a80a3cf413c2c6ee01'),
        'joint': (joint_plan['banks']['0']['checkpoint'], joint_plan['banks']['0']['sha256']),
    }
    for label, folder in {
        'merged_lr1e5': 'full_control_coverage_full_balanced_ce',
        'merged_lr1e4': 'full_control_coverage_lr1e4_balanced_ce',
        'base_lr1e5': 'base_initialization_ce_lr1e5',
        'base_lr1e4': 'base_initialization_ce_lr1e4',
    }.items():
        run = read(root/f'seed0/{folder}/run.json')
        specs[label] = (str(root/f'seed0/{folder}/merged.ckpt'), run['checkpoint_sha256'])
    states = {}
    for label, (path, expected) in specs.items():
        assert sha256(Path(path)) == expected, path
        sources[path] = expected
        states[label] = torch.load(path, map_location='cpu', weights_only=True)
        assert set(states[label]) == set(states['base'])
        assert all(torch.isfinite(states[label][n]).all() for n in names)
    groups = {c: [n for n in names if component(n) == c]
              for c in ['tokenizer', 'tower_rest', 'qformer_rest', 'readout']}
    groups['all'] = names
    assert sum(len(groups[c]) for c in list(groups)[:-1]) == len(names)

    def vector(state, selected):
        return torch.cat([state[n].flatten().double() for n in selected])

    def describe(delta, target):
        energy, target_energy = float(delta.square().sum()), float(target.square().sum())
        dot = float(delta@target)
        return {
            'norm': math.sqrt(energy), 'rms': math.sqrt(energy/len(delta)),
            'norm_over_joint_displacement': math.sqrt(energy/target_energy),
            'cosine_to_joint_displacement': dot/math.sqrt(energy*target_energy) if energy else None,
            'projection_fraction_of_joint_displacement': dot/target_energy,
            'remaining_squared_distance_fraction': float((target-delta).square().sum())/target_energy,
        }

    rows = {}
    for c, selected in groups.items():
        vs = {key: vector(state, selected) for key, state in states.items()}
        base, merged, joint = [vs[key] for key in ['base', 'initial_merger', 'joint']]
        rows[c] = {
            'parameters': len(base), 'tensors': len(selected),
            'base_norm': float(base.norm()),
            'joint_from_base_norm': float((joint-base).norm()),
            'joint_from_merged_norm': float((joint-merged).norm()),
            'initial_merger_from_base': describe(merged-base, joint-base),
            'repairs': {key: describe(vs[key]-(merged if key.startswith('merged') else base),
                                     joint-(merged if key.startswith('merged') else base))
                        for key in ['merged_lr1e5', 'merged_lr1e4', 'base_lr1e5', 'base_lr1e4']},
        }

    mean_gradient = {n: torch.zeros_like(states['base'][n], dtype=torch.float64) for n in names}
    individual_energy = {c: 0.0 for c in groups}
    for task in cache['datasets']:
        rec = cache['records'][task]
        assert sha256(Path(rec['path'])) == rec['sha256']
        sources[rec['path']] = rec['sha256']
        grad = torch.load(rec['path'], map_location='cpu', weights_only=True)
        assert set(grad) == set(names)
        for name in names:
            value = grad[name].double()
            mean_gradient[name] += value/len(cache['datasets'])
            energy = float(value.square().sum())/len(cache['datasets'])
            individual_energy[component(name)] += energy
            individual_energy['all'] += energy
    for c, selected in groups.items():
        g = vector(mean_gradient, selected)
        mean_energy = float(g.square().sum())
        rows[c]['initial_original512_gradient'] = {
            'mean_gradient_norm': math.sqrt(mean_energy),
            'mean_task_gradient_squared_norm': individual_energy[c],
            'between_task_variance_over_mean_norm_sq': individual_energy[c]/mean_energy-1,
            'directional_derivatives': {
                key: float(g@(vector(states[key], selected)-vector(states['initial_merger'], selected)))
                for key in ['joint', 'merged_lr1e5', 'merged_lr1e4']},
        }
    expected_ratio = joint_run['lr_tokenizer']/joint_run['lr_rest']
    assert abs(expected_ratio-10) < 1e-12
    result = {
        'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': sources,
        'historical_joint_group_rates': {k: joint_run[k] for k in ['lr_tokenizer', 'lr_rest', 'lr_schedule', 'weight_decay', 'precision']},
        'historical_tokenizer_to_rest_lr_ratio': expected_ratio,
        'components': rows, 'seconds': time.time()-started,
        'new_model_inference': False, 'new_backpropagation': False, 'optimizer_updates': 0,
        'new_EEG_data_read': False, 'validation_metrics_used': False,
        'scope': 'Coordinate-dependent checkpoint displacement and existing original512 balanced-CE gradient diagnostic. Joint weights are a diagnostic oracle only; no candidate is constructed from them. Different distance scales/directions do not establish a bottleneck or a causal explanation of BAcc. The historical LR ratio is configuration evidence, not a value fit to Joint weights. Cached gradients omit Adam moments, full-pool sampling and finite-step effects.',
    }
    output = root/'analysis/optimizer_group_geometry.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'seconds': result['seconds'],
                      'components': {c: {'parameters': v['parameters'], 'joint_from_base_norm': v['joint_from_base_norm'],
                                       'merged_1e5': v['repairs']['merged_lr1e5'], 'base_1e5': v['repairs']['base_lr1e5']}
                                     for c, v in rows.items()}}, indent=2))


if __name__ == '__main__':
    main()
