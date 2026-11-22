"""Resolve inconclusive FFN identity bounds with exact weight-distance matching."""
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch

from d_downstream_qformer_ce import sha256
from diagnose_ffn_permutation_identity import signatures


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    cp = root/'analysis/ffn_permutation_identity.json'
    sp = root/'seed0/original_fp32_tensor_balanced_ce/run.json'
    certificate, source = [json.loads(p.read_text()) for p in [cp, sp]]
    for filename, expected in certificate['input_sha256'].items():
        assert sha256(Path(filename)) == expected, filename
    modules = {name: tuple(certificate['modules'][name]) for name in {r['module'] for r in certificate['uncertified']}}
    paths = {'base': source['base_checkpoint'], **source['experts']}
    rows = {name: signatures(torch.load(path, map_location='cpu', weights_only=True, mmap=True), modules)
            for name, path in paths.items()}
    results = []
    for case in certificate['uncertified']:
        a, b = rows[case['first']][case['module']], rows[case['second']][case['module']]
        cost = a.square().sum(1)[:, None]+b.square().sum(1)[None, :]-2*(a@b.T)
        diagonal = cost.diagonal().clone()
        assert float(diagonal.min()) >= -1e-12
        off = cost.clone()
        off.fill_diagonal_(float('inf'))
        margins = off.min(1).values-diagonal
        unique_identity = bool((margins > 0).all())
        if unique_identity:
            permutation = np.arange(len(a))
            algorithm = 'strict_rowwise_minimum_certificate'
        else:
            selected_rows, permutation = linear_sum_assignment(cost.numpy())
            assert np.array_equal(selected_rows, np.arange(len(a)))
            algorithm = 'scipy_linear_sum_assignment'
        selected = cost[torch.arange(len(a)), torch.from_numpy(permutation)]
        gain = float(diagonal.sum()-selected.sum())
        assert gain >= -1e-9
        results.append({'first': case['first'], 'second': case['second'], 'module': case['module'],
                        'algorithm': algorithm, 'strict_unique_identity_certificate': unique_identity,
                        'minimum_offdiagonal_minus_diagonal_squared_distance': float(margins.min()),
                        'moved_units': int(np.sum(permutation != np.arange(len(a)))),
                        'identity_squared_distance': float(diagonal.sum()),
                        'matched_squared_distance': float(selected.sum()),
                        'squared_distance_reduction': gain,
                        'permutation': permutation.tolist() if np.any(permutation != np.arange(len(a))) else None})
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'input_sha256': {str(p): sha256(p) for p in [cp, sp, Path('diagnose_ffn_permutation_identity.py')]},
              'scipy_version': scipy.__version__, 'resolved_cases': results,
              'total_pair_module_checks': certificate['number_of_pair_module_checks'],
              'previous_sufficient_identity_certificates': certificate['certified_pair_module_checks'],
              'new_strict_identity_certificates': sum(r['strict_unique_identity_certificate'] for r in results),
              'cases_with_nonidentity_assignment': sum(r['moved_units'] > 0 for r in results),
              'total_moved_units': sum(r['moved_units'] for r in results),
              'seconds': time.time()-started,
              'data_read': False, 'model_inference_performed': False, 'checkpoint_modified': False,
              'scope': 'Exact FP64 squared-weight-distance matrices for the126 local FFN matching cases left inconclusive by triangle bounds. A strict rowwise minimum certifies identity uniquely; otherwise an exact linear assignment is solved. All external residual coordinates remain fixed. This does not cover residual-stream permutations, attention gauges, activation/loss-aware matching or interpolation barriers. No model trained or evaluated.'}
    with (root/'analysis/ffn_permutation_exact_assignments.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['total_pair_module_checks', 'previous_sufficient_identity_certificates',
                                            'new_strict_identity_certificates', 'cases_with_nonidentity_assignment',
                                            'total_moved_units', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
