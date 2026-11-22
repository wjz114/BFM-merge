"""Measure frozen initial-space coverage at predeclared trajectory checkpoints."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from diagnose_ri_span import project


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == expected, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    basis = json.loads(Path(plan['basis_run']).read_text())
    source = json.loads(Path(basis['plan']['source_fit']).read_text())
    reference = json.loads(Path(plan['initial_capture_reference']).read_text())
    tasks, names = source['datasets'], source['groups']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {}
    for task in tasks:
        path = Path(source['experts'][task])
        assert sha256(path) == basis['plan']['expert_sha256'][task]
        experts[task] = torch.load(path, map_location='cpu', weights_only=True)
    extensions = {}
    for kind, row in basis['outputs'].items():
        assert sha256(Path(row['path'])) == row['sha256']
        extensions[kind] = torch.load(row['path'], map_location='cpu', weights_only=True)
    caches, gradients, cache_sources = {}, {}, {}
    for label, path in plan['gradient_caches'].items():
        path = Path(path)
        cache = json.loads(path.read_text())
        assert cache['datasets'] == tasks and cache['parameter_names'] == names
        assert cache['optimizer_updates'] == 0 and not cache['test_data_read']
        assert cache.get('checkpoint_state_exactly_unchanged', cache.get('initial_state_exactly_unchanged'))
        caches[label] = cache
        cache_sources[label] = {'path': str(path), 'sha256': sha256(path)}
        gradients[label] = {}
        for task in tasks:
            row = cache['records'][task]
            assert sha256(Path(row['path'])) == row['sha256']
            gradients[label][task] = torch.load(row['path'], map_location='cpu', weights_only=True)
            assert set(gradients[label][task]) == set(names)
            if label != 'initial':
                initial = caches['initial']['records'][task]
                for key in ['eeg_hash', 'labels_hash', 'instruction_hash', 'prototypes_hash']:
                    assert row[key] == initial[key]
    labels = list(caches)
    assert labels[0] == 'initial'
    width = len(tasks)+1
    total = torch.zeros((len(labels), width), dtype=torch.float64)
    dots = torch.zeros_like(total)
    residuals = {k: torch.zeros_like(total) for k in ['original_only', 'gradient', 'random']}
    records, components = {}, {}
    for index, name in enumerate(names):
        original = torch.stack([(experts[t][name]-base[name]).flatten() for t in tasks]).double()
        raw = []
        for label in labels:
            values = torch.stack([gradients[label][t][name].flatten() for t in tasks]).double()
            raw.append(torch.cat([values, values.mean(0, keepdim=True)]))
        values = torch.stack(raw)
        energy = values.square().sum(2)
        total += energy
        dots += (values*values[:1]).sum(2)
        raw = values.reshape(len(labels)*width, -1)
        component = name.split('.')[0]
        comp = components.setdefault(component, {
            'total': torch.zeros_like(total),
            'residuals': {k: torch.zeros_like(total) for k in residuals}})
        comp['total'] += energy
        row = {'raw_energy': energy.tolist(), 'spaces': {}}
        for kind in residuals:
            matrix = original if kind == 'original_only' else torch.cat([
                original, extensions[kind][name].reshape(16, -1).double()])
            projection, residual, _, solver = project(matrix, raw)
            residue = residual.square().sum(1).reshape(len(labels), width)
            residuals[kind] += residue
            comp['residuals'][kind] += residue
            error = float((projection.square().sum()+residual.square().sum()-raw.square().sum()).abs()
                          /raw.square().sum().clamp_min(1e-30))
            assert error <= 1e-5
            row['spaces'][kind] = {'rank': solver['rank'], 'residual_energy': residue.tolist(),
                                   'relative_pythagorean_error': error}
        records[name] = row
        if (index+1) % 50 == 0:
            print('Measured', index+1, '/', len(names), flush=True)
    assert (total > 0).all()
    summaries = {}
    for i, label in enumerate(labels):
        summaries[label] = {
            'energy_weighted_captured_fraction': {
                k: float(1-v[i, :-1].sum()/total[i, :-1].sum()) for k, v in residuals.items()},
            'task_mean_gradient_captured_fraction': {
                k: float(1-v[i, -1]/total[i, -1]) for k, v in residuals.items()},
            'total_task_gradient_energy': float(total[i, :-1].sum()),
            'task_gradient_energy_ratio_to_initial': float(total[i, :-1].sum()/total[0, :-1].sum()),
            'mean_task_gradient_cosine_to_initial': float((dots[i, :-1]/(total[i, :-1]*total[0, :-1]).sqrt()).mean()),
            'task_mean_gradient_cosine_to_initial': float(dots[i, -1]/(total[i, -1]*total[0, -1]).sqrt()),
            'per_task': {t: {
                'gradient_norm': float(total[i, j].sqrt()),
                'cosine_to_initial': float(dots[i, j]/(total[i, j]*total[0, j]).sqrt()),
                'captured_fraction': {k: float(1-v[i, j]/total[i, j]) for k, v in residuals.items()}}
                for j, t in enumerate(tasks)},
            'components': {c: {
                'energy_weighted_captured_fraction': {
                    k: float(1-v[i, :-1].sum()/comp['total'][i, :-1].sum()) for k, v in comp['residuals'].items()},
                'gradient_energy': float(comp['total'][i, :-1].sum())}
                for c, comp in components.items()},
            'macro_calibration_bacc': float(np.mean([caches[label]['records'][t]['calibration_metrics']['BAcc'] for t in tasks])),
            'macro_weighted_ce': float(np.mean([caches[label]['records'][t]['weighted_ce'] for t in tasks]))}
    for kind, value in summaries['initial']['energy_weighted_captured_fraction'].items():
        assert abs(value-reference['energy_weighted_captured_fraction'][kind]) < 1e-10
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'tasks': tasks, 'state_labels': labels, 'gradient_row_labels': tasks+['task_mean'],
              'summaries': summaries, 'tensor_records': records, 'cache_sources': cache_sources,
              'initial_capture_replayed': True, 'all_calibration_and_text_hashes_match_initial': True,
              'seconds': time.time()-started, 'model_inference_performed': False,
              'optimizer_updates': 0, 'new_EEG_data_read': False,
              'scope': 'Fixed initial-space projection of empirical original512 balancedCE gradients at predeclared existing checkpoints. Same data and weights across states. Coordinate-dependent geometric diagnostic, not an accuracy/novelty claim or optimizer-independent capacity bound. No validation/test gradients or Joint information.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {n: v for n, v in row.items() if n not in ['per_task', 'components']} for k, row in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
