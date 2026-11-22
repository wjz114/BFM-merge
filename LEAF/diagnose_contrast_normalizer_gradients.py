"""Measure cosine-normalization gradients outside task prototype contrasts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256


def geometry(raw, prototypes, truth, basis):
    radius = raw.norm(dim=-1, keepdim=True)
    unit = raw/radius
    logits = unit@prototypes.T
    v = (logits.softmax(-1)-F.one_hot(truth, len(prototypes)))@prototypes
    alpha = (unit*v).sum(-1, keepdim=True)
    gradient = (v-alpha*unit)/radius
    projected = (gradient@basis.T)@basis
    normalizer = gradient-projected
    predicted_normalizer = -alpha*(unit-(unit@basis.T)@basis)/radius
    return gradient, projected, normalizer, predicted_normalizer, logits


def toy_check():
    torch.manual_seed(0)
    p = F.normalize(torch.randn(4, 9, dtype=torch.float64), dim=-1)
    _, s, vh = torch.linalg.svd(p-p.mean(0), full_matrices=False)
    basis = vh[s > s[0]*1e-8]
    z = torch.randn(7, 9, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    native, = torch.autograd.grad(F.cross_entropy(F.normalize(z, dim=-1)@p.T, y, reduction='sum'), z)
    analytic, _, outside, predicted, _ = geometry(z.detach(), p, y, basis)
    error = float((native-analytic).abs().max())
    outside_error = float((outside-predicted).abs().max())
    assert error < 1e-12 and outside_error < 1e-12
    return {'analytic_vs_autograd_max_error': error, 'normalizer_identity_max_error': outside_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    output = Path(plan['output'])
    assert not output.exists()
    started = time.time()
    torch.set_num_threads(4)
    toy = toy_check()
    targets = json.loads(Path(plan['target_manifest']).read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    gradients = json.loads(Path(plan['gradient_manifest']).read_text())
    schedule, calibration = np.load(plan['schedule']), np.load(plan['calibration'])
    records, original_heads, projected_heads, cached_rows = {}, [], [], 0
    for ti, task in enumerate(plan['datasets']):
        ref = targets['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        p = cache['prototypes'].double()
        _, singular, vh = torch.linalg.svd(p-p.mean(0), full_matrices=False)
        basis = vh[singular > singular[0]*plan['rank_relative_cutoff']]
        assert 1 <= len(basis) <= len(p)-1
        ids = cache['pool_indices'].numpy()
        stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert source['datasets'][ti] == task and array_hash(stream) == ref['stream_indices_hash']
        positions = np.searchsorted(ids, stream)
        assert np.array_equal(ids[positions], stream)
        multiplicity = torch.bincount(torch.from_numpy(positions), minlength=len(ids)).double()
        classes = torch.tensor(ref['class_weights'], dtype=torch.float64)
        head = torch.zeros(768, 2049, dtype=torch.float64)
        total_energy = outside_energy = identity_error2 = outside_energy_unweighted = 0.
        max_logit_error = 0.
        for start in range(0, len(ids), plan['batch_size']):
            sl = slice(start, start+plan['batch_size'])
            truth = cache['truth'][sl]
            g, gp, gn, expected, logits = geometry(cache['initial_raw'][sl].double(), p, truth, basis)
            max_logit_error = max(max_logit_error, float((logits-cache['initial_logits'][sl]).abs().max()))
            count, weight = multiplicity[sl], classes[truth]
            objective_weight = count*weight
            head[:, :-1] += (g*objective_weight[:, None]).T@cache['initial_prehead'][sl].double()
            head[:, -1] += (g*objective_weight[:, None]).sum(0)
            energy_weight = count*weight.square()
            total_energy += float((energy_weight*g.square().sum(-1)).sum())
            outside_energy += float((energy_weight*gn.square().sum(-1)).sum())
            identity_error2 += float((count*(gn-expected).square().sum(-1)).sum())
            outside_energy_unweighted += float((count*gn.square().sum(-1)).sum())
        assert max_logit_error < plan['initial_logit_replay_atol']
        head /= multiplicity.sum()
        projected_head = basis.T@(basis@head)
        outside_head = head-projected_head
        original_heads.append(head.flatten())
        projected_heads.append(projected_head.flatten())
        # Independent native-FP32 reference from the original512 calibration.
        old_ids = calibration[task]
        old_positions = np.searchsorted(ids, old_ids)
        assert np.array_equal(ids[old_positions], old_ids)
        y = cache['truth'][old_positions]
        g, _, _, _, _ = geometry(cache['initial_raw'][old_positions].double(), p, y, basis)
        w = torch.tensor(source['ce_class_weights'][task], dtype=torch.float64)[y]
        replay_head = (g*w[:, None]).T@cache['initial_prehead'][old_positions].double()/len(old_ids)
        replay_bias = (g*w[:, None]).mean(0)
        old = gradients['records'][task]
        assert sha256(Path(old['path'])) == old['sha256']
        native = torch.load(old['path'], map_location='cpu', weights_only=True)
        replay = {}
        for name, value in [('qformer.output_embed.2.weight', replay_head), ('qformer.output_embed.2.bias', replay_bias)]:
            reference = native[name].double()
            replay[name] = float((value-reference).norm()/reference.norm())
            assert replay[name] < plan['native_head_gradient_replay_rtol']
        records[task] = {'contrast_rank': len(basis), 'class_count': len(p),
            'per_exposure_weighted_gradient_outside_fraction': outside_energy/total_energy,
            'aggregate_head_gradient_outside_fraction': float(outside_head.square().sum()/head.square().sum()),
            'aggregate_head_gradient_norm': float(head.norm()),
            'projected_head_gradient_norm': float(projected_head.norm()),
            'normalizer_identity_relative_error': (identity_error2/max(outside_energy_unweighted, 1e-300))**.5,
            'initial_logit_max_error': max_logit_error, 'native_head_gradient_replay_relative_error': replay,
            'target_sha256': ref['target_sha256'], 'gradient_reference_sha256': old['sha256'],
            'unique_fitting_rows': int((multiplicity > 0).sum()), 'fitting_exposures': int(multiplicity.sum())}
        cached_rows += len(ids)
        print(task, records[task]['aggregate_head_gradient_outside_fraction'], flush=True)
        del cache, native
    assert cached_rows == plan['budget']['cached_union_rows']
    original = torch.stack(original_heads)
    projected = torch.stack(projected_heads)
    norms, projected_norms = original.norm(dim=1), projected.norm(dim=1)
    original_cosine = (original@original.T)/(norms[:, None]*norms[None, :])
    cross_cosine = (original@projected.T)/(norms[:, None]*projected_norms[None, :])
    off = ~torch.eye(len(original), dtype=torch.bool)
    mean_gradient = original.mean(0)
    original_dd = -(original@mean_gradient)/norms
    projected_dd = -(projected@mean_gradient)/projected_norms
    result = {'unix': time.time(), 'plan': plan, 'plan_sha256': sha256(args.plan),
        'code_sha256': sha256(Path(__file__)), 'toy': toy, 'records': records,
        'original_gradient_cosine': original_cosine.tolist(),
        'original_loss_vs_projected_update_cosine': cross_cosine.tolist(),
        'matrix_convention': 'row is recipient original CE objective; column is source task whose negative gradient supplies the update.',
        'ordered_off_diagonal_pair_count': int(off.sum()),
        'original_negative_transfer_pairs': int((original_cosine[off] < 0).sum()),
        'projected_negative_transfer_pairs': int((cross_cosine[off] < 0).sum()),
        'macro_original_ce_directional_derivative_unit_head_step': dict(zip(plan['datasets'], original_dd.tolist())),
        'macro_original_ce_directional_derivative_projected_unit_head_step': dict(zip(plan['datasets'], projected_dd.tolist())),
        'projected_better_macro_direction_tasks': int((projected_dd < original_dd).sum()),
        'new_eeg_reads': 0, 'new_model_forward_examples': 0, 'optimizer_updates': 0,
        'cached_union_rows': cached_rows, 'seconds': time.time()-started, 'scope': plan['scope']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print('Completed', json.dumps({k: result[k] for k in ['seconds', 'original_negative_transfer_pairs',
        'projected_negative_transfer_pairs', 'projected_better_macro_direction_tasks']}), flush=True)


if __name__ == '__main__':
    main()
