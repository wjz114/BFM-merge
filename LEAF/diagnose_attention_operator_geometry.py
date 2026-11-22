"""Attention head correspondence and factor cross terms, without data or fitting."""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from d_downstream_qformer_ce import sha256


def factors(state, name):
    w = state[name+'.in_proj_weight'].double()
    b = state[name+'.in_proj_bias'].double()
    assert w.shape == (768, 256) and b.shape == (768,)
    q, k, v = w.reshape(3, 8, 32, 256)
    qb, kb, vb = b.reshape(3, 8, 32)
    o = state[name+'.out_proj.weight'].double().reshape(256, 8, 32).permute(1, 0, 2)
    return {'q': torch.cat([q, qb[:, :, None]], -1), 'k': k,
            'v': torch.cat([v, vb[:, :, None]], -1), 'o': o,
            'kb': kb, 'ob': state[name+'.out_proj.bias'].double()}


def operators(f):
    return {'qk': f['q'].transpose(-1, -2)@f['k']/math.sqrt(32), 'vo': f['o']@f['v']}


def quadratic(d):
    return {'qk': d['q'].transpose(-1, -2)@d['k']/math.sqrt(32), 'vo': d['o']@d['v']}


def signature(f):
    return torch.cat([f[k].reshape(8, -1) for k in ['q', 'k', 'kb', 'v', 'o']], 1)


def algebra_check(f, generator):
    x, y = [torch.randn(n, 256, generator=generator, dtype=torch.float64) for n in [5, 7]]
    xa, ya = [torch.cat([t, torch.ones(len(t), 1, dtype=t.dtype)], -1) for t in [x, y]]
    op = operators(f)
    before, after, score_error = [], [], []
    for h in range(8):
        score = (xa@f['q'][h].T)@(y@f['k'][h].T+f['kb'][h]).T/math.sqrt(32)
        canonical = xa@op['qk'][h]@y.T
        score_error.append(float(((score-score.mean(-1, keepdim=True))-(canonical-canonical.mean(-1, keepdim=True))).abs().max()))
        before.append(score.softmax(-1)@(ya@f['v'][h].T)@f['o'][h].T)
        after.append(canonical.softmax(-1)@(ya@op['vo'][h].T))
    error = float((torch.stack(before).sum(0)-torch.stack(after).sum(0)).abs().max())
    assert error < 1e-10 and max(score_error) < 1e-10
    return {'max_abs_output_error': error, 'max_abs_centered_score_error': max(score_error)}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    source_path = root/'seed0/original_fp32_tensor_balanced_avg/run.json'
    cache_path = root/'seed0/full_model_distillation_targets/run.json'
    source, cache = [json.loads(p.read_text()) for p in [source_path, cache_path]]
    base_path, merged_path = Path(source['base_checkpoint']), source_path.with_name('merged.ckpt')
    assert sha256(base_path) == source['base_checkpoint_sha256']
    assert str(merged_path) == cache['plan']['initial_checkpoint']
    hashes = {str(p): sha256(p) for p in [source_path, cache_path, base_path, merged_path, Path('LEAF.py')]}
    base, merged = [torch.load(p, map_location='cpu', weights_only=True, mmap=True) for p in [base_path, merged_path]]
    names = [k.removesuffix('.out_proj.weight') for k in base if k.endswith('.out_proj.weight')]
    assert len(names) == 32
    bf, mf = [{n: factors(state, n) for n in names} for state in [base, merged]]
    mean_factors = {n: {k: torch.zeros_like(v) for k, v in bf[n].items()} for n in names}
    mean_operators = {n: {k: torch.zeros_like(v) for k, v in operators(bf[n]).items()} for n in names}
    mean_quadratic = {n: {k: torch.zeros_like(v) for k, v in operators(bf[n]).items()} for n in names}
    alignment = {}
    for task in source['datasets']:
        path = Path(source['experts'][task])
        hashes[str(path)] = sha256(path)
        assert hashes[str(path)] == cache['records'][task]['expert_sha256']
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        checks = {}
        for name in names:
            f = factors(state, name)
            for key, value in f.items():
                mean_factors[name][key] += value/16
            for key, value in operators(f).items():
                mean_operators[name][key] += value/16
            for key, value in quadratic({k: f[k]-bf[name][k] for k in f}).items():
                mean_quadratic[name][key] += value/16
            a, b = signature(bf[name]), signature(f)
            cost = a.square().sum(1)[:, None]+b.square().sum(1)[None, :]-2*a@b.T
            diag = cost.diagonal().clone()
            off = cost.clone()
            off.fill_diagonal_(float('inf'))
            margin = off.min(1).values-diag
            rows, permutation = linear_sum_assignment(cost.numpy())
            checks[name] = {'strict_identity_row_minimum': bool((margin > 0).all()),
                            'minimum_squared_distance_margin': float(margin.min()),
                            'moved_heads': int((permutation != np.arange(8)).sum()),
                            'distance_reduction': float(diag.sum()-cost[torch.from_numpy(rows), torch.from_numpy(permutation)].sum())}
        alignment[task] = checks
    generator = torch.Generator().manual_seed(0)
    reconstruction = {state: {n: algebra_check(f[n], generator) for n in names} for state, f in [('base', bf), ('merger', mf)]}
    records = {}
    covariance_replay_max, expansion_replay_max = 0., 0.
    for name in names:
        base_op, merged_op, uniform_op = [operators(f[name]) for f in [bf, mf, mean_factors]]
        mean_delta = {k: mean_factors[name][k]-bf[name][k] for k in bf[name]}
        merged_delta = {k: mf[name][k]-bf[name][k] for k in bf[name]}
        mean_q, merged_q = quadratic(mean_delta), quadratic(merged_delta)
        linear = {'qk': (bf[name]['q'].transpose(-1, -2)@merged_delta['k']+merged_delta['q'].transpose(-1, -2)@bf[name]['k'])/math.sqrt(32),
                  'vo': bf[name]['o']@merged_delta['v']+merged_delta['o']@bf[name]['v']}
        record = {}
        for key in ['qk', 'vo']:
            target = mean_operators[name][key]
            covariance = target-uniform_op[key]
            replay = mean_quadratic[name][key]-mean_q[key]
            err = float((covariance-replay).norm()/covariance.norm())
            assert err < 1e-8
            covariance_replay_max = max(covariance_replay_max, err)
            delta = merged_op[key]-base_op[key]
            err = float((delta-linear[key]-merged_q[key]).norm()/delta.norm())
            assert err < 1e-10
            expansion_replay_max = max(expansion_replay_max, err)
            singular = torch.linalg.svdvals(target)
            tail = singular[:, 32:].square().sum(-1)
            squared = lambda t: t.square().flatten(1).sum(1)
            covariance2, target2 = squared(covariance), squared(target)
            assert bool((tail <= covariance2+1e-9*target2).all())
            record[key] = {'uniform_factor_covariance_squared_norm': covariance2.tolist(),
                           'uniform_target_squared_norm': target2.tolist(),
                           'uniform_operator_change_from_base_squared_norm': squared(target-base_op[key]).tolist(),
                           'best_rank32_target_squared_error': tail.tolist(),
                           'merged_operator_change_from_base_squared_norm': squared(delta).tolist(),
                           'merged_quadratic_term_squared_norm': squared(merged_q[key]).tolist(),
                           'merged_linear_term_squared_norm': squared(linear[key]).tolist()}
        records[name] = record
    aggregate = {}
    for key in ['qk', 'vo']:
        sums = {metric: sum(sum(row[key][metric]) for row in records.values()) for metric in records[names[0]][key]}
        aggregate[key] = {'squared_norm_sums': sums,
                          'uniform_factor_error_over_operator_target_norm': math.sqrt(sums['uniform_factor_covariance_squared_norm']/sums['uniform_target_squared_norm']),
                          'uniform_factor_error_over_operator_change_norm': math.sqrt(sums['uniform_factor_covariance_squared_norm']/sums['uniform_operator_change_from_base_squared_norm']),
                          'best_rank32_error_over_factor_error': math.sqrt(sums['best_rank32_target_squared_error']/sums['uniform_factor_covariance_squared_norm']),
                          'merged_quadratic_over_actual_operator_change_norm': math.sqrt(sums['merged_quadratic_term_squared_norm']/sums['merged_operator_change_from_base_squared_norm'])}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'modules': names, 'heads_per_module': 8, 'head_rank': 32,
              'base_expert_head_alignment': alignment,
              'all512_head_matchings_strict_identity': all(c['strict_identity_row_minimum'] for row in alignment.values() for c in row.values()),
              'total_moved_heads': sum(c['moved_heads'] for row in alignment.values() for c in row.values()),
              'synthetic_operator_reconstruction': reconstruction,
              'covariance_identity_max_relative_error': covariance_replay_max,
              'merger_delta_expansion_max_relative_error': expansion_replay_max,
              'per_module': records, 'aggregate': aggregate, 'seconds': time.time()-started,
              'real_data_read': False, 'full_model_inference_performed': False, 'checkpoint_modified': False,
              'scope': 'QK uses augmented query and unaugmented key weights, divided by sqrt32; key bias contributes only a per-query constant removed by softmax. VO includes value bias; output bias is a separate affine constant. Reconstruction is checked with synthetic inputs on actual base/merger weights. Uniform expert operator means are diagnostic targets, not the learned merger objective or a teacher-output ensemble. Current learned-merger quadratic terms are exact coordinate-dependent expansions about the base, not error attribution. Frobenius and rank32 errors are unweighted by activations and do not bound classification accuracy. This is not SLOA reproduction, a candidate export, or a novelty claim.'}
    with (root/'analysis/attention_operator_geometry.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'all_head_matchings_identity': result['all512_head_matchings_strict_identity'],
                      'total_moved_heads': result['total_moved_heads'], 'aggregate': aggregate, 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
