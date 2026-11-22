"""Sufficient identity-matching certificates for local FFN hidden permutations."""
import itertools
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def signatures(state, modules):
    result = {}
    for name, (first, second) in modules.items():
        incoming, bias, outgoing = [state[k].double() for k in [first+'.weight', first+'.bias', second+'.weight']]
        assert incoming.shape[0] == len(bias) == outgoing.shape[1]
        result[name] = torch.cat([incoming, bias[:, None], outgoing.T], dim=1)
    return result


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    source_path = root/'seed0/original_fp32_tensor_balanced_ce/run.json'
    cache_path = root/'seed0/full_model_distillation_targets/run.json'
    source, cache = [json.loads(p.read_text()) for p in [source_path, cache_path]]
    bp = Path(source['base_checkpoint'])
    assert sha256(bp) == source['base_checkpoint_sha256']
    base = torch.load(bp, map_location='cpu', weights_only=True)
    modules = {}
    for key in base:
        if key.endswith('.linear1.weight'):
            prefix = key.removesuffix('.linear1.weight')
            modules[prefix] = (prefix+'.linear1', prefix+'.linear2')
        elif key.endswith('.ff.0.weight'):
            prefix = key.removesuffix('.ff.0.weight')
            modules[prefix] = (prefix+'.ff.0', prefix+'.ff.2')
    assert len(modules) == 28
    for first, second in modules.values():
        assert {first+'.weight', first+'.bias', second+'.weight'} <= set(source['groups'])
    reference = signatures(base, modules)
    separations = {}
    for name, rows in reference.items():
        distances = torch.cdist(rows, rows, compute_mode='use_mm_for_euclid_dist')
        distances.fill_diagonal_(float('inf'))
        separations[name] = distances.min(1).values
        assert (separations[name] > 0).all()
    shifts = {'base': {name: torch.zeros(len(rows), dtype=torch.float64) for name, rows in reference.items()}}
    hashes = {str(p): sha256(p) for p in [source_path, cache_path, bp, Path('LEAF.py')]}
    for task in source['datasets']:
        path = Path(source['experts'][task])
        actual = sha256(path)
        assert actual == cache['records'][task]['expert_sha256']
        hashes[str(path)] = actual
        state = torch.load(path, map_location='cpu', weights_only=True)
        candidate = signatures(state, modules)
        shifts[task] = {name: (candidate[name]-rows).norm(dim=1) for name, rows in reference.items()}
        del state, candidate
    pair_checks, failed = {}, []
    for a, b in itertools.combinations(shifts, 2):
        checks = {}
        for name in modules:
            da, db, separation = shifts[a][name], shifts[b][name], separations[name]
            # For row i: own distance <= da_i+db_i, cross distance >= separation_i-da_i-max(db).
            # The reverse bound independently checks the other direction.
            margin_ab = separation-(2*da+db+db.max())
            margin_ba = separation-(2*db+da+da.max())
            margin = torch.minimum(margin_ab, margin_ba)
            certified = bool((margin > 0).all())
            checks[name] = {'certified_identity_unique_minimum': certified,
                            'hidden_units': len(margin),
                            'minimum_distance_margin_lower_bound': float(margin.min()),
                            'uncertified_rows': int((margin <= 0).sum())}
            if not certified:
                failed.append({'first': a, 'second': b, 'module': name, **checks[name]})
        pair_checks[a+'__'+b] = checks
    cells = [cell for row in pair_checks.values() for cell in row.values()]
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'modules': {k: list(v) for k, v in modules.items()},
              'number_of_checkpoints': len(shifts), 'number_of_pairs': len(pair_checks),
              'number_of_pair_module_checks': len(cells),
              'all_local_identity_assignments_certified': not failed,
              'certified_pair_module_checks': sum(v['certified_identity_unique_minimum'] for v in cells),
              'minimum_margin_lower_bound': min(v['minimum_distance_margin_lower_bound'] for v in cells),
              'uncertified': failed, 'pair_checks': pair_checks,
              'per_checkpoint_max_row_shift': {task: {name: float(v.max()) for name, v in row.items()} for task, row in shifts.items()},
              'seconds': time.time()-started,
              'inference_performed': False, 'optimization_performed': False, 'data_read': False,
              'checkpoint_modified': False,
              'scope': 'Sufficient triangle-inequality certificates, using FP64 weight signatures [incoming row, hidden bias, outgoing column]. For each of28 FFNs, external residual coordinates stay fixed. A positive bound proves identity is the unique closest-row assignment and hence unique squared-weight-distance minimizer for this local hidden permutation. A nonpositive bound is inconclusive. Does not cover jointly permuting residual streams, attention heads/internal gauges, activation matching, loss-aware permutations or linear mode connectivity. No model trained, merged, exported or evaluated.'}
    path = root/'analysis/ffn_permutation_identity.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['number_of_checkpoints', 'number_of_pairs', 'number_of_pair_module_checks',
                                            'all_local_identity_assignments_certified', 'certified_pair_module_checks',
                                            'minimum_margin_lower_bound', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
