"""Measure initial task-gradient coverage of the three frozen pilot spaces."""
import argparse
import json
import time
from pathlib import Path

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
    for filename, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(filename)) == expected, filename
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    basis = json.loads(Path(plan['basis_run']).read_text())
    source = json.loads(Path(basis['plan']['source_fit']).read_text())
    cache = json.loads(Path(basis['gradient_cache']).read_text())
    assert sha256(Path(basis['gradient_cache'])) == basis['gradient_cache_sha256']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts, gradients, extensions = {}, {}, {}
    for task in source['datasets']:
        path = Path(source['experts'][task])
        assert sha256(path) == basis['plan']['expert_sha256'][task]
        experts[task] = torch.load(path, map_location='cpu', weights_only=True)
        record = cache['records'][task]
        assert sha256(Path(record['path'])) == record['sha256']
        gradients[task] = torch.load(record['path'], map_location='cpu', weights_only=True)
    for name, record in basis['outputs'].items():
        assert sha256(Path(record['path'])) == record['sha256']
        extensions[name] = torch.load(record['path'], map_location='cpu', weights_only=True)
    tasks = source['datasets']
    total = torch.zeros(len(tasks), dtype=torch.float64)
    residuals = {name: torch.zeros_like(total) for name in ['original_only', 'gradient', 'random']}
    records = {}
    for index, name in enumerate(source['groups']):
        original = torch.stack([(experts[t][name]-base[name]).flatten() for t in tasks]).double()
        raw = torch.stack([gradients[t][name].flatten() for t in tasks]).double()
        raw_energy = raw.square().sum(1)
        total += raw_energy
        row = {'raw_gradient_energy_per_task': raw_energy.tolist(), 'spaces': {}}
        for kind in residuals:
            matrix = original if kind == 'original_only' else torch.cat([
                original, extensions[kind][name].reshape(16, -1).double()])
            projection, residual, _, solver = project(matrix, raw)
            energy = residual.square().sum(1)
            residuals[kind] += energy
            pythagorean_error = float((projection.square().sum()+residual.square().sum()-raw.square().sum()).abs()
                                      /raw.square().sum().clamp_min(1e-30))
            assert pythagorean_error <= 1e-5
            row['spaces'][kind] = {'rank': solver['rank'], 'residual_energy_per_task': energy.tolist(),
                                   'relative_pythagorean_error': pythagorean_error}
        records[name] = row
        if (index+1) % 50 == 0:
            print('Measured', index+1, '/', len(source['groups']), flush=True)
    assert (total > 0).all()
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'tasks': tasks, 'tensor_records': records,
        'total_gradient_energy_per_task': dict(zip(tasks, total.tolist())),
        'energy_weighted_captured_fraction': {
            kind: float(1-energy.sum()/total.sum()) for kind, energy in residuals.items()},
        'per_task_captured_fraction': {
            task: {kind: float(1-energy[i]/total[i]) for kind, energy in residuals.items()}
            for i, task in enumerate(tasks)},
        'seconds': time.time()-started, 'model_inference_performed': False,
        'optimizer_updates': 0, 'validation_data_read': False, 'test_data_read': False,
        'scope': 'FP64 least-squares projection of cached initial task gradients onto actual FP32 expert/basis artifacts. Coordinate-dependent local representability; gradient coverage is largely guaranteed by construction. It does not imply a matched full-weight Adam update, later-trajectory coverage, generalization, or novelty. No new data or model inference.',
    }
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', result['energy_weighted_captured_fraction'], flush=True)


if __name__ == '__main__':
    main()
