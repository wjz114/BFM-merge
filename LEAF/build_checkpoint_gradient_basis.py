"""Construct one refreshed gradient basis and its matched random control at a fixed checkpoint."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256
from diagnose_ri_span import project


def orthonormal_rows(matrix, cutoff):
    values, vectors = torch.linalg.eigh(matrix@matrix.T)
    keep = values > max(float(values[-1]), 0.)*cutoff
    basis = (vectors[:, keep].T@matrix)/values[keep, None].sqrt()
    return basis, values, keep


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
    source = json.loads(Path(plan['source_fit']).read_text())
    cache_path = Path(plan['gradient_cache'])/'run.json'
    cache = json.loads(cache_path.read_text())
    assert cache['plan_sha256'] == plan['gradient_plan_sha256']
    assert cache['optimizer_updates'] == 0 and cache['checkpoint_state_exactly_unchanged']
    tasks, names = source['datasets'], source['groups']
    assert tasks == cache['datasets'] and names == cache['parameter_names']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts, gradients = {}, {}
    for task in tasks:
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        experts[task] = torch.load(path, map_location='cpu', weights_only=True)
        record = cache['records'][task]
        assert sha256(Path(record['path'])) == record['sha256']
        gradients[task] = torch.load(record['path'], map_location='cpu', weights_only=True)
    gradient_basis, random_basis, geometry = {}, {}, {}
    for column, name in enumerate(names):
        original = torch.stack([(experts[t][name]-base[name]).flatten() for t in tasks]).double()
        raw = torch.stack([gradients[t][name].flatten() for t in tasks]).double()
        _, residual, _, solver = project(original, raw)
        assert solver['rank'] == 16
        informed, eigenvalues, keep = orthonormal_rows(residual, plan['gradient_relative_eigenvalue_cutoff'])
        rank = int(keep.sum())
        output_shape = (16, *base[name].shape)
        informed_output = torch.zeros((16, original.shape[1]), dtype=torch.float64)
        random_output = torch.zeros_like(informed_output)
        if rank:
            # Equal singular values and total norm: both extensions have the
            # same rank and Gram matrix, with only their parameter directions differing.
            scale = float(original.norm())/math.sqrt(rank)
            generator = torch.Generator().manual_seed(plan['random_seed']+104729*column)
            noise = torch.randn((rank, original.shape[1]), generator=generator, dtype=torch.float64)
            _, noise, _, _ = project(original, noise)
            random, _, random_keep = orthonormal_rows(noise, 1e-10)
            assert int(random_keep.sum()) == rank, name
            informed_output[:rank] = informed*scale
            random_output[:rank] = random*scale
        else:
            scale = 0.
        informed_output, random_output = informed_output.float(), random_output.float()
        target_gram = torch.zeros((16, 16), dtype=torch.float64)
        target_gram[:rank, :rank] = torch.eye(rank, dtype=torch.float64)*scale**2
        checks = {}
        for label, candidate in [('gradient', informed_output), ('random', random_output)]:
            candidate = candidate.double()
            gram_error = float((candidate@candidate.T-target_gram).norm()/target_gram.norm().clamp_min(1e-30))
            cross_error = float((original@candidate.T).norm()/(original.norm()*candidate.norm()).clamp_min(1e-30))
            assert gram_error <= 1e-5 and cross_error <= 1e-6, (name, label, gram_error, cross_error)
            checks[label] = {'relative_gram_error_after_fp32': gram_error,
                             'relative_cross_gram_after_fp32': cross_error}
        gradient_basis[name] = informed_output.reshape(output_shape)
        random_basis[name] = random_output.reshape(output_shape)
        raw_energy, residual_energy = float(raw.square().sum()), float(residual.square().sum())
        geometry[name] = {'rank': rank, 'original_rank': solver['rank'], 'row_scale': scale,
                          'raw_gradient_energy': raw_energy, 'outside_gradient_energy': residual_energy,
                          'residual_gram_eigenvalues': eigenvalues.tolist(), 'checks': checks}
        if (column+1) % 50 == 0:
            print('Built', column+1, '/', len(names), flush=True)
    outputs = {}
    for label, tensors in [('gradient', gradient_basis), ('random', random_basis)]:
        path = args.outdir/f'{label}_basis.pt'
        torch.save(tensors, path)
        outputs[label] = {'path': str(path), 'sha256': sha256(path)}
    raw_energy = sum(v['raw_gradient_energy'] for v in geometry.values())
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'gradient_cache': str(cache_path), 'gradient_cache_sha256': sha256(cache_path),
              'outputs': outputs, 'geometry': geometry, 'nominal_added_coefficients': 16*len(names),
              'effective_added_coefficients': sum(v['rank'] for v in geometry.values()),
              'outside_gradient_energy_fraction': sum(v['outside_gradient_energy'] for v in geometry.values())/raw_energy,
              'model_inference_performed': False, 'fitting_performed': False, 'test_data_read': False,
              'seconds': time.time()-started,
              'scope': 'Per tensor: remove the original16-expert span from16 task gradients, retain residual Gram eigenvalues above fixed1e-8 relative cutoff, whiten retained rows and scale total Frobenius norm to the original expert basis. The random extension is orthogonal to that same expert span and matches rank/Gram/norm. No validation or Joint information enters construction. This creates a candidate search space, not a performance or novelty result.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed refreshed gradient/random bases', result['effective_added_coefficients'], flush=True)


if __name__ == '__main__':
    main()
