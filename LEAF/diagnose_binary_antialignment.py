"""Check whether binary merger errors concentrate near zero-gradient antipodes."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_decision_disagreements import quantiles
from d_downstream_qformer_ce import sha256


def describe(logits, teacher_logits, truth, weights, prototypes, raw=None):
    weights = weights.double()/weights.sum()
    difference = prototypes[1].double()-prototypes[0].double()
    radius = difference.norm()
    signed = (2*truth-1)*(logits[:, 1].double()-logits[:, 0].double())/radius
    assert float(signed.abs().max()) <= 1+1e-5
    tangent = (1-signed.square()).clamp_min(0).sqrt()
    correct, tc = logits.argmax(-1) == truth, teacher_logits.argmax(-1) == truth
    groups = {'correct': correct, 'wrong': ~correct,
              'teacher_correct_student_wrong': tc & ~correct, 'teacher_wrong_student_wrong': ~tc & ~correct}
    result = {'weighted_accuracy': float(weights[correct].sum()), 'groups': {}}
    for name, mask in groups.items():
        mass = float(weights[mask].sum())
        result['groups'][name] = {'weighted_mass': mass, 'exposures': int(mask.sum()),
            'signed_alignment_p10_p50_p90': quantiles(signed[mask], weights[mask]) if mass else None,
            'head_tangent_factor_p10_p50_p90': quantiles(tangent[mask], weights[mask]) if mass else None,
            'antialignment_tails': {str(c): {'weighted_mass_all_rows': float(weights[mask & (signed < -c)].sum()),
                'fraction_within_group': float(weights[mask & (signed < -c)].sum()/mass) if mass else None}
                for c in [.9, .99, .999]}}
    if raw is not None:
        h = raw.double()
        norm = h.norm(dim=-1)
        unit = h/norm[:, None]
        gap = unit@difference
        analytic = (gap.sigmoid()-truth)[:, None]*(difference[None]-unit*gap[:, None])/norm[:, None]
        result['raw_feature_gradient_norm_p10_p50_p90'] = {
            name: quantiles(analytic[mask].norm(dim=-1), weights[mask]) if bool(mask.any()) else None for name, mask in groups.items()}
        result['raw_feature_norm_p10_p50_p90'] = {
            name: quantiles(norm[mask], weights[mask]) if bool(mask.any()) else None for name, mask in groups.items()}
        probe = h[:32].detach().clone().requires_grad_()
        loss = F.cross_entropy(F.normalize(probe, dim=-1)@prototypes.double().T, truth[:32], reduction='sum')
        actual, = torch.autograd.grad(loss, probe)
        error = float((actual-analytic[:32]).abs().max())
        assert error < 1e-12
        result['analytic_gradient_autograd_max_abs_error'] = error
        result['reconstructed_logit_gap_max_abs_error'] = float((gap-(logits[:, 1]-logits[:, 0]).double()).abs().max())
    return result


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    destination = root/'analysis/binary_antialignment.json'
    assert not destination.exists()
    cache_path = root/'seed0/full_model_distillation_targets/run.json'
    reference_path = root/'analysis/decision_disagreements.json'
    cache, reference = [json.loads(p.read_text()) for p in [cache_path, reference_path]]
    schedule_path = Path(cache['plan']['schedule'])
    schedule = np.load(schedule_path)
    sources = {str(p): sha256(p) for p in [cache_path, reference_path, schedule_path, Path(__file__), Path('diagnose_decision_disagreements.py')]}
    rows = {}
    for ti, task in enumerate(cache['datasets']):
        spec = cache['records'][task]
        if len(spec['class_weights']) != 2:
            continue
        path = Path(spec['target_path'])
        assert sha256(path) == spec['target_sha256']
        sources[str(path)] = spec['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64))
        pos = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][pos], ids)
        truth = data['truth'][pos]
        weights = torch.tensor(spec['class_weights'], dtype=torch.float64)[truth]
        full = describe(data['initial_logits'][pos], data['teacher_logits'][pos], truth, weights, data['prototypes'], data['initial_raw'][pos])
        assert abs(full['weighted_accuracy']-reference['full_stream'][task]['student_weighted_accuracy']) < 1e-12
        subset_path = root/f'seed0/input_response_v3_shard{ti%2}/{task}.pt'
        subset_run = root/f'seed0/input_response_v3_shard{ti%2}/run.json'
        meta = json.loads(subset_run.read_text())['records'][task]
        assert sha256(subset_path) == meta['sha256']
        sources[str(subset_path)] = meta['sha256']
        subset = torch.load(subset_path, map_location='cpu', weights_only=True)
        n = subset['n_original']
        pools = {}
        for pool, section in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            y = subset['truth'][section]
            w = (n/(2*torch.bincount(y).double()))[y]
            pools[pool] = {m: describe(subset['logits'][m][section], subset['logits']['teacher'][section], y, w, data['prototypes'])
                           for m in ['initial', 'ce', 'kl']}
        rows[task] = {'full_initial': full, 'subsets': pools}
    assert len(rows) == 8
    summaries = {}
    selections = {'full_initial': [r['full_initial'] for r in rows.values()]}
    selections.update({f'{pool}_{model}': [r['subsets'][pool][model] for r in rows.values()]
                       for pool in ['original', 'replacement'] for model in ['initial', 'ce', 'kl']})
    for name, items in selections.items():
        summaries[name] = {}
        for group in ['wrong', 'teacher_correct_student_wrong']:
            entries = [r['groups'][group] for r in items]
            mass = float(np.mean([r['weighted_mass'] for r in entries]))
            summaries[name][group] = {'macro_weighted_mass': mass,
                'tails': {str(c): {'macro_weighted_mass_all_rows': float(np.mean([r['antialignment_tails'][str(c)]['weighted_mass_all_rows'] for r in entries])),
                    'fraction_of_macro_error_mass': float(np.mean([r['antialignment_tails'][str(c)]['weighted_mass_all_rows'] for r in entries])/mass) if mass else None}
                    for c in [.9, .99, .999]}}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'source_sha256': sources,
              'datasets': list(rows), 'per_task': rows, 'summaries': summaries, 'seconds': time.time()-started,
              'new_model_inference': False, 'optimizer_updates': 0, 'test_data_read': False,
              'scope': 'Binary tasks only, existing fitting multiset and dependent128-row original/replacement subsets. Tail cutoffs0.9/0.99/0.999 are descriptive geometric levels, not a selected training rule. The factor sqrt(1-cosine^2) describes the normalized readout Jacobian; parameter Jacobians and Adam dynamics are not measured. Small factors neither prove a trapping basin nor establish a remedy. Full-stream actual raw-feature gradients are analytic, unweighted per-example CE gradients, summarized with the original class weights.'}
    with destination.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', destination, sha256(destination), 'seconds', result['seconds'])
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
