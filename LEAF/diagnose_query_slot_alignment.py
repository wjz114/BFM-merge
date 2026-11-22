"""Check query-slot weight and cached-response matching before any alignment trial."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from LEAF import TextConditionedQFormer


PERMUTATIONS = np.asarray(list(itertools.permutations(range(8))), dtype=np.int64)


def assignment(cost):
    c = cost.cpu().numpy()
    assert c.shape == (8, 8) and np.isfinite(c).all()
    totals = c[np.arange(8)[None, :], PERMUTATIONS].sum(1)
    order = np.argsort(totals, kind='stable')
    best, second = int(order[0]), int(order[1])
    assert np.array_equal(PERMUTATIONS[0], np.arange(8))
    off = c.copy()
    np.fill_diagonal(off, np.inf)
    return {'permutation': PERMUTATIONS[best].tolist(), 'identity_optimal': best == 0,
            'identity_cost': float(totals[0]), 'optimal_cost': float(totals[best]),
            'identity_minus_optimal': float(totals[0]-totals[best]),
            'relative_cost_reduction': float((totals[0]-totals[best])/totals[0]) if totals[0] > 0 else None,
            'runner_up_cost_gap': float(totals[second]-totals[best]),
            'strict_identity_row_margin': float((off.min(1)-np.diag(c)).min()),
            'permutation_cost_median': float(np.median(totals)), 'cost_matrix': c.tolist()}


def weight_signature(state):
    q = state['qformer.base_queries'].double()
    w = state['qformer.output_embed.2.weight'].double()
    assert q.shape == (8, 256) and w.shape == (768, 2048)
    return torch.cat([q, w.reshape(768, 8, 256).permute(1, 0, 2).reshape(8, -1)], 1)


def toy_symmetry():
    torch.manual_seed(0)
    model = TextConditionedQFormer(num_q=3, dim_q=4, dim_kv=4, n_heads=2,
                                  n_layers=2, text_dim=6, max_tokens=8).double().eval()
    perm = torch.tensor([2, 0, 1])
    changed = copy.deepcopy(model)
    changed.base_queries.copy_(model.base_queries[perm])
    changed.output_embed[2].weight.copy_(model.output_embed[2].weight.reshape(6, 3, 4)[:, perm].reshape(6, 12))
    tokens, instruction = torch.randn(2, 5, 4, dtype=torch.float64), torch.randn(2, 6, dtype=torch.float64)
    original, transformed = model(tokens, instruction), changed(tokens, instruction)
    error = float((original-transformed).abs().max())
    assert torch.allclose(original, transformed, rtol=1e-12, atol=1e-12)
    return {'maximum_absolute_output_error': error, 'synthetic_qformer_forward_examples': 4,
            'learned_queries_and_output_blocks_permuted_together': True}


def response_costs(initial, teacher, weights, batch_size):
    assert initial.shape == teacher.shape and initial.shape[1] == 2048
    ma = torch.zeros(8, 256, dtype=torch.float64)
    mb = torch.zeros_like(ma)
    aa = torch.zeros(8, dtype=torch.float64)
    bb = torch.zeros_like(aa)
    cross = torch.zeros(8, 8, dtype=torch.float64)
    mass = float(weights.sum())
    assert mass > 0
    for start in range(0, len(initial), batch_size):
        sl = slice(start, start+batch_size)
        a = initial[sl].double().reshape(-1, 8, 256)
        b = teacher[sl].double().reshape(-1, 8, 256)
        w = weights[sl]
        ma += (a*w[:, None, None]).sum(0)
        mb += (b*w[:, None, None]).sum(0)
        aa += (a.square()*w[:, None, None]).sum((0, 2))
        bb += (b.square()*w[:, None, None]).sum((0, 2))
        cross += torch.einsum('nqd,nrd,n->qr', a, b, w)
    ma, mb, aa, bb, cross = ma/mass, mb/mass, aa/mass, bb/mass, cross/mass
    raw = aa[:, None]+bb[None, :]-2*cross
    mean_distance = (ma[:, None, :]-mb[None, :, :]).square().sum(-1)
    centered = raw-mean_distance
    return raw/256, centered/256


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    output = Path(plan['output'])
    assert not output.exists()
    started = time.time()
    torch.set_num_threads(4)
    toy = toy_symmetry()
    source = json.loads(Path(plan['source_fit']).read_text())
    manifest = json.loads(Path(plan['target_manifest']).read_text())
    expert_manifest = json.loads(Path(plan['expert_manifest']).read_text())
    schedule = np.load(plan['schedule'])
    reference_signatures = {}
    for name, path in [('base', plan['base_checkpoint']), ('initial_merger', plan['initial_checkpoint'])]:
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        reference_signatures[name] = weight_signature(state)
    records, cached_rows = {}, 0
    for ti, task in enumerate(plan['datasets']):
        ref = manifest['records'][task]
        cp, ep = Path(ref['target_path']), Path(source['experts'][task])
        assert sha256(cp) == ref['target_sha256']
        assert sha256(ep) == expert_manifest['records'][task]['expert_sha256']
        expert = torch.load(ep, map_location='cpu', weights_only=True, mmap=True)
        sig = weight_signature(expert)
        weight_costs = {name: assignment(torch.cdist(reference, sig).square())
                        for name, reference in reference_signatures.items()}
        cache = torch.load(cp, map_location='cpu', weights_only=True)
        ids = cache['pool_indices'].numpy()
        stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert source['datasets'][ti] == task and array_hash(stream) == ref['stream_indices_hash']
        positions = np.searchsorted(ids, stream)
        assert np.array_equal(ids[positions], stream)
        count = torch.bincount(torch.from_numpy(positions), minlength=len(ids)).double()
        assert int(count.sum()) == ref['fitting_exposures']
        assert int((count > 0).sum()) == ref['unique_fitting_rows']
        weights = count*torch.tensor(ref['class_weights'], dtype=torch.float64)[cache['truth']]
        raw, centered = response_costs(cache['initial_prehead'], cache['teacher_prehead'], weights, plan['batch_size'])
        records[task] = {'weight_matching': weight_costs, 'raw_response_matching': assignment(raw),
                         'centered_response_matching': assignment(centered),
                         'target_sha256': ref['target_sha256'],
                         'expert_sha256': expert_manifest['records'][task]['expert_sha256'],
                         'cached_union_rows': len(ids), 'unique_fitting_rows': int((count > 0).sum()),
                         'fitting_exposures': int(count.sum())}
        cached_rows += len(ids)
        print(task, {k: records[task][k]['permutation'] for k in ['raw_response_matching', 'centered_response_matching']}, flush=True)
        del cache, expert, sig
    assert cached_rows == plan['budget']['cached_union_rows']
    result = {'unix': time.time(), 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'toy_symmetry': toy, 'records': records,
              'cached_union_rows': cached_rows, 'new_eeg_reads': 0, 'new_leaf_forward_examples': 0,
              'optimizer_updates': 0, 'checkpoints_modified': False, 'seconds': time.time()-started,
              'scope': plan['scope']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print('Completed', json.dumps({'output': str(output), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
