"""Build a fixed published SWUDI-A baseline, preserving the native LEAF state.

Reference: arXiv2606.07289v1, Algorithm1; official ASWUDI default at
WalkerWorldPeace/MLLMerging commit dee33ad04fd15439fb937bc9d9ce9e1a93610962.
This is an adaptation of an existing method, not a new merging algorithm.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, load_leaf, sha256
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES


@torch.no_grad()
def spectral_delta(vectors):
    vectors = vectors.float()
    count, rows, columns = vectors.shape
    norm_sq = vectors.flatten(1).norm(dim=1).square()
    if norm_sq.max() < 1e-12:
        return vectors.mean(0), {'operation': 'tiny-update mean', 'rank': None}
    inverse_norm = norm_sq.clamp_min(1e-12).reciprocal()
    gram = torch.einsum('nab,nac->nbc', vectors, vectors)
    operator = torch.einsum('n,nbc->bc', inverse_norm, gram)
    rhs = torch.einsum('n,nac->ac', inverse_norm, torch.einsum('nab,nbc->nac', vectors, gram))
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    safe = eigenvalues.clamp_min(1e-8)
    descending_roots = safe.flip(0).sqrt()
    rank = int(torch.ceil(descending_roots.sum().square()/descending_roots.square().sum().clamp_min(1e-30)))
    rank = max(1, min(rank, columns))
    basis = eigenvectors[:, -rank:]
    anchor = vectors.sum(0)
    fitted = ((rhs@basis)/safe[-rank:])@basis.T
    retained_anchor = (anchor@basis)@basis.T
    result = anchor-retained_anchor+fitted
    assert torch.isfinite(result).all()
    return result, {'operation': 'SWUDI-A participation_sqrt, hard filter, sum anchor',
                    'rank': rank, 'input_dimension': columns, 'rank_fraction': rank/columns,
                    'minimum_raw_eigenvalue': float(eigenvalues.min()),
                    'maximum_eigenvalue': float(eigenvalues.max()),
                    'discarded_anchor_energy': float((anchor-retained_anchor).double().square().sum()),
                    'result_energy': float(result.double().square().sum()),
                    'mean_delta_energy': float(vectors.mean(0).double().square().sum())}


def analytic_checks():
    # One expert is an exact fixed point even if its spectrum is truncated.
    generator = torch.Generator().manual_seed(173)
    single = torch.randn(1, 7, 5, generator=generator)
    recovered, _ = spectral_delta(single)
    single_error = float((recovered-single[0]).abs().max())
    assert single_error < 2e-5
    # Orthogonal, equally weighted column supports have an exact sum solution.
    disjoint = torch.zeros(2, 2, 2)
    disjoint[0, 0, 0] = 1
    disjoint[1, 1, 1] = 1
    recovered, info = spectral_delta(disjoint)
    assert info['rank'] == 2 and torch.equal(recovered, torch.eye(2))
    tiny = torch.full((3, 2, 4), 1e-9)
    recovered, info = spectral_delta(tiny)
    assert info['rank'] is None and torch.equal(recovered, tiny.mean(0))
    return {'single_expert_fixed_point_max_error': single_error,
            'orthogonal_support_sum_exact': True, 'tiny_update_fallback_exact': True}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, digest in plan['source_sha256'].items():
        assert sha256(Path(filename)) == digest, filename
    assert sha256(Path(plan['source_fit'])) == plan['source_fit_sha256']
    source = json.loads(Path(plan['source_fit']).read_text())
    for path, digest in plan['checkpoint_sha256'].items():
        assert sha256(Path(path)) == digest, path
    outdir = Path(plan['outdir'])
    outdir.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    checks = analytic_checks()
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).eval()
    base = leaf.state_dict()
    active = [name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)]
    fixed = [name for name in base if name not in active]
    assert len(active) == 383 and len(fixed) == 13
    affine = set()
    for prefix, module in leaf.named_modules():
        if isinstance(module, torch.nn.Linear):
            affine.add(prefix+'.weight')
        elif isinstance(module, torch.nn.MultiheadAttention):
            affine.update(prefix+'.'+name for name in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
                          if getattr(module, name, None) is not None)
    non_affine = [name for name in active if base[name].ndim == 2 and name not in affine]
    assert non_affine == ['qformer.base_queries']
    del leaf
    experts = [torch.load(path, map_location='cpu', weights_only=True) for path in source['experts'].values()]
    assert len(experts) == 16
    assert all(set(state) == set(base) and all(torch.equal(state[n], base[n]) for n in fixed) for state in experts)
    result = {name: value.clone() for name, value in base.items()}
    diagnostics, query_mean = {}, None
    for i, name in enumerate(active):
        vectors = torch.stack([state.pop(name).float()-base[name].float() for state in experts])
        if base[name].ndim == 2:
            delta, info = spectral_delta(vectors)
        else:
            delta, info = vectors.mean(0), {'operation': 'arithmetic mean for non-2D parameter'}
        result[name] = base[name]+delta
        diagnostics[name] = info
        assert torch.isfinite(result[name]).all()
        if name == 'qformer.base_queries':
            query_mean = base[name]+vectors.mean(0)
        if (i+1) % 30 == 0:
            print('Built', i+1, 'of', len(active), flush=True)
    assert query_mean is not None and all(torch.equal(result[name], base[name]) for name in fixed)
    all_path = outdir/'all_2d.ckpt'
    torch.save(result, all_path)
    original_query = result['qformer.base_queries']
    result['qformer.base_queries'] = query_mean
    affine_path = outdir/'affine_only.ckpt'
    torch.save(result, affine_path)
    states = [torch.load(path, map_location='cpu', weights_only=True) for path in [all_path, affine_path]]
    differences = [name for name in base if not torch.equal(states[0][name], states[1][name])]
    assert differences == ['qformer.base_queries']
    for state in states:
        restored = load_leaf(config, Path(source['base_checkpoint']))
        restored.load_state_dict(state, strict=True)
        assert all(torch.equal(value, state[name]) for name, value in restored.state_dict().items())
    record = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': source['datasets'], 'checks': checks, 'decompositions': diagnostics,
              'fixed_tensors_exact': True, 'active_parameters': sum(base[n].numel() for n in active),
              'native_state_reload_exact': True, 'variant_differences': differences,
              'query_policy_max_difference': float((original_query-query_mean).abs().max()),
              'checkpoints': {key: {'path': str(path), 'sha256': sha256(path)}
                              for key, path in [('all_2d', all_path), ('affine_only', affine_path)]},
              'seconds': time.time()-started, 'optimizer_updates': 0, 'data_arrays_read': False,
              'test_data_read': False, 'scope': plan['scope']}
    (outdir/'build.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Build complete', record['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
