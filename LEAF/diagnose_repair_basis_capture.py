"""Measure how frozen pilot spaces represent existing successful repair updates."""
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
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    experts, extensions, updates, metadata = {}, {}, {}, {}
    for task in source['datasets']:
        path = Path(source['experts'][task])
        assert sha256(path) == basis['plan']['expert_sha256'][task]
        experts[task] = torch.load(path, map_location='cpu', weights_only=True)
    for kind, record in basis['outputs'].items():
        assert sha256(Path(record['path'])) == record['sha256']
        extensions[kind] = torch.load(record['path'], map_location='cpu', weights_only=True)
    for label, path in plan['repair_runs'].items():
        meta = json.loads(Path(path).read_text())
        assert meta['plan']['initial_checkpoint_sha256'] == sha256(Path(plan['initial_checkpoint']))
        assert meta['datasets'] == source['datasets'] and meta['updates'] == 1600
        checkpoint = Path(path).with_name('merged.ckpt')
        assert sha256(checkpoint) == meta['checkpoint_sha256']
        updates[label] = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert all(torch.equal(updates[label][n], initial[n]) for n in initial if n not in source['groups'])
        metadata[label] = meta
    labels = list(updates)
    totals = torch.zeros(len(labels), dtype=torch.float64)
    residuals = {kind: torch.zeros_like(totals) for kind in ['original_only', 'gradient', 'random']}
    records = {}
    for index, name in enumerate(source['groups']):
        original = torch.stack([(experts[t][name]-base[name]).flatten() for t in source['datasets']]).double()
        targets = torch.stack([(updates[label][name].double()-initial[name].double()).flatten()
                               for label in labels])
        energy = targets.square().sum(1)
        totals += energy
        row = {'update_energy': dict(zip(labels, energy.tolist())), 'spaces': {}}
        for kind in residuals:
            matrix = original if kind == 'original_only' else torch.cat([
                original, extensions[kind][name].reshape(16, -1).double()])
            parallel, residual, _, solver = project(matrix, targets)
            residual_energy = residual.square().sum(1)
            residuals[kind] += residual_energy
            error = float((parallel.square().sum()+residual.square().sum()-targets.square().sum()).abs()
                          /targets.square().sum().clamp_min(1e-30))
            assert error <= 1e-5
            row['spaces'][kind] = {'rank': solver['rank'], 'residual_energy': dict(zip(labels, residual_energy.tolist())),
                                   'relative_pythagorean_error': error}
        records[name] = row
        if (index+1) % 50 == 0:
            print('Measured', index+1, '/', len(source['groups']), flush=True)
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'repair_labels': labels, 'tensor_records': records,
        'captured_fraction': {label: {kind: float(1-residuals[kind][i]/totals[i]) for kind in residuals}
                              for i, label in enumerate(labels)},
        'repair_update_energy': dict(zip(labels, totals.tolist())),
        'source_fixed_average_macro_bacc': {label: metadata[label]['average_validation_macro_bacc'] for label in labels},
        'seconds': time.time()-started, 'model_inference_performed': False, 'optimizer_updates': 0,
        'validation_data_read': False, 'test_data_read': False,
        'scope': 'Coordinate-dependent FP64 least-squares geometry of two existing averaged full-weight repair updates against the actual frozen original/gradient/random bases. Original512 repair matches the candidate fitting pool; full-pool repair has broader access. No fitting or new model inference. Low captured energy is not a capacity bound and does not determine projected-model accuracy; functional intervention would be needed for that claim.',
    }
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', result['captured_fraction'], flush=True)


if __name__ == '__main__':
    main()
