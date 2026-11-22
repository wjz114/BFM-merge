"""Class-balanced decision disagreements and margins from existing training caches."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def quantiles(x, weights):
    order = x.argsort(stable=True)
    cdf = weights[order].cumsum(0)/weights.sum()
    positions = torch.searchsorted(cdf, torch.tensor([.1, .5, .9], dtype=torch.float64)).clamp_max(len(x)-1)
    return x[order[positions]].tolist()


def margins(logits, prototypes):
    values, indices = logits.topk(2, dim=1)
    bound = (prototypes[indices[:, 0]]-prototypes[indices[:, 1]]).norm(dim=1)
    assert (bound > 0).all()
    normalized = (values[:, 0]-values[:, 1])/bound
    assert float(normalized.max()) < 1+1e-5
    return normalized


def describe(teacher, student, truth, weights, prototypes):
    teacher, student, weights = teacher.double(), student.double(), weights.double()
    weights = weights/weights.sum()
    tpred, spred = teacher.argmax(-1), student.argmax(-1)
    tc, sc = tpred == truth, spred == truth
    masks = {'both_correct': tc & sc, 'teacher_correct_student_wrong': tc & ~sc,
             'teacher_wrong_student_correct': ~tc & sc,
             'both_wrong_same_prediction': ~tc & ~sc & (tpred == spred),
             'both_wrong_different_prediction': ~tc & ~sc & (tpred != spred)}
    tm, sm = margins(teacher, prototypes.double()), margins(student, prototypes.double())
    tlp, slp = teacher.log_softmax(-1), student.log_softmax(-1)
    kl = (tlp.exp()*(tlp-slp)).sum(-1)
    entropy = -(tlp.exp()*tlp).sum(-1)/np.log(teacher.shape[1])
    groups = {}
    for label, mask in masks.items():
        mass = float(weights[mask].sum())
        groups[label] = {'weighted_mass': mass, 'examples': int(mask.sum()),
                         'teacher_margin_quantiles_p10_p50_p90': quantiles(tm[mask], weights[mask]) if mass else None,
                         'student_margin_quantiles_p10_p50_p90': quantiles(sm[mask], weights[mask]) if mass else None,
                         'teacher_normalized_entropy_mean': float((weights[mask]*entropy[mask]).sum()/mass) if mass else None,
                         'weighted_kl_contribution': float((weights[mask]*kl[mask]).sum())}
    assert abs(sum(g['weighted_mass'] for g in groups.values())-1) < 1e-12
    tacc, sacc = float(weights[tc].sum()), float(weights[sc].sum())
    assert abs(tacc-sacc-groups['teacher_correct_student_wrong']['weighted_mass']+groups['teacher_wrong_student_correct']['weighted_mass']) < 1e-12
    order = tm.argsort(stable=True)
    # Equal row-count quintiles are descriptive; all original class weights are retained.
    quintiles = []
    for ids in torch.tensor_split(order, 5):
        w = weights[ids]
        quintiles.append({'examples': len(ids), 'weighted_mass': float(w.sum()),
                          'normalized_teacher_margin_range': [float(tm[ids].min()), float(tm[ids].max())],
                          'teacher_correct_fraction': float(w[tc[ids]].sum()/w.sum()),
                          'student_correct_fraction': float(w[sc[ids]].sum()/w.sum()),
                          'teacher_correct_student_wrong_fraction': float(w[(tc & ~sc)[ids]].sum()/w.sum()),
                          'weighted_kl_contribution': float((w*kl[ids]).sum())})
    return {'teacher_weighted_accuracy': tacc, 'student_weighted_accuracy': sacc,
            'teacher_student_accuracy_gap': tacc-sacc,
            'weighted_disagreement': float(weights[tpred != spred].sum()),
            'weighted_teacher_kl': float((weights*kl).sum()),
            'groups': groups, 'teacher_margin_quintiles_low_to_high': quintiles}


def transitions(before, after, teacher, truth, weights):
    w = weights.double()/weights.sum()
    bc, ac, tc = [logits.argmax(-1) == truth for logits in [before, after, teacher]]
    masks = {'recovered_teacher_correct': ~bc & ac & tc,
             'recovered_teacher_wrong': ~bc & ac & ~tc,
             'lost_teacher_correct': bc & ~ac & tc,
             'lost_teacher_wrong': bc & ~ac & ~tc,
             'persistent_wrong_teacher_correct': ~bc & ~ac & tc,
             'persistent_wrong_teacher_wrong': ~bc & ~ac & ~tc,
             'persistent_correct_teacher_correct': bc & ac & tc,
             'persistent_correct_teacher_wrong': bc & ac & ~tc}
    masses = {k: float(w[v].sum()) for k, v in masks.items()}
    assert abs(sum(masses.values())-1) < 1e-12
    delta = sum(v for k, v in masses.items() if k.startswith('recovered'))-sum(v for k, v in masses.items() if k.startswith('lost'))
    assert abs(delta-float(w[ac].sum()-w[bc].sum())) < 1e-12
    return {'weighted_accuracy_change': delta, 'groups': masses}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    target = root/'analysis/decision_disagreements.json'
    assert not target.exists()
    cache_path = root/'seed0/full_model_distillation_targets/run.json'
    input_path = root/'analysis/input_response_summary.json'
    geometry_path = root/'analysis/full_distillation_radial_and_label_geometry.json'
    cache, diagnostic, geometry = [json.loads(p.read_text()) for p in [cache_path, input_path, geometry_path]]
    hashes = {str(p): sha256(p) for p in [cache_path, input_path, geometry_path]}
    assert cache['datasets'] == diagnostic['datasets']
    schedule = np.load(cache['plan']['schedule'])
    hashes[cache['plan']['schedule']] = sha256(Path(cache['plan']['schedule']))
    full, subsets = {}, {}
    for ti, task in enumerate(cache['datasets']):
        spec = cache['records'][task]
        path = Path(spec['target_path'])
        assert sha256(path) == spec['target_sha256']
        hashes[str(path)] = spec['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64))
        positions = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][positions], ids) and len(ids) == 12800
        truth = data['truth'][positions]
        w = torch.tensor(spec['class_weights'], dtype=torch.float64)[truth]
        full[task] = describe(data['teacher_logits'][positions], data['initial_logits'][positions], truth, w, data['prototypes'])
        reference = geometry['per_task'][task]
        assert abs(full[task]['teacher_weighted_accuracy']-reference['weighted_teacher_correct_fraction']) < 1e-12
        assert abs(full[task]['student_weighted_accuracy']-reference['weighted_initial_correct_fraction']) < 1e-12
        subset_path = root/f'seed0/input_response_v3_shard{ti%2}/{task}.pt'
        source_run = root/f'seed0/input_response_v3_shard{ti%2}/run.json'
        subset_record = json.loads(source_run.read_text())['records'][task]
        assert sha256(subset_path) == subset_record['sha256']
        hashes[str(subset_path)] = subset_record['sha256']
        saved = torch.load(subset_path, map_location='cpu', weights_only=True)
        n = saved['n_original']
        subsets[task] = {}
        for pool, select in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            y = saved['truth'][select]
            counts = torch.bincount(y)
            weights = (n/(len(counts)*counts.double()))[y]
            logits = {m: z[select] for m, z in saved['logits'].items()}
            subsets[task][pool] = {
                'models': {m: describe(logits['teacher'], logits[m], y, weights, data['prototypes']) for m in ['initial', 'ce', 'kl']},
                'transitions': {after+'_minus_'+before: transitions(logits[before], logits[after], logits['teacher'], y, weights)
                                for after, before in [('ce', 'initial'), ('kl', 'ce')]}}
    macro_full = {key: float(np.mean([row[key] for row in full.values()])) for key in
                  ['teacher_weighted_accuracy', 'student_weighted_accuracy', 'teacher_student_accuracy_gap', 'weighted_disagreement', 'weighted_teacher_kl']}
    macro_full['group_mass'] = {g: float(np.mean([row['groups'][g]['weighted_mass'] for row in full.values()])) for g in next(iter(full.values()))['groups']}
    macro_subsets = {}
    for pool in ['original', 'replacement']:
        macro_subsets[pool] = {}
        for model in ['initial', 'ce', 'kl']:
            rows = [subsets[t][pool]['models'][model] for t in cache['datasets']]
            macro_subsets[pool][model] = {key: float(np.mean([r[key] for r in rows])) for key in
                                         ['teacher_weighted_accuracy', 'student_weighted_accuracy', 'teacher_student_accuracy_gap', 'weighted_disagreement']}
            macro_subsets[pool][model]['group_mass'] = {g: float(np.mean([r['groups'][g]['weighted_mass'] for r in rows])) for g in rows[0]['groups']}
        macro_subsets[pool]['transitions'] = {pair: {g: float(np.mean([subsets[t][pool]['transitions'][pair]['groups'][g] for t in cache['datasets']]))
                                                   for g in subsets[cache['datasets'][0]][pool]['transitions'][pair]['groups']}
                                             for pair in ['ce_minus_initial', 'kl_minus_ce']}
    result = {'code_sha256': sha256(Path(__file__)), 'source_sha256': hashes, 'datasets': cache['datasets'],
              'full_stream': full, 'subsets': subsets, 'macro_full_stream': macro_full,
              'macro_subsets': macro_subsets, 'seconds': time.time()-started,
              'new_model_inference': False, 'optimizer_updates': 0,
              'scope': 'Post hoc training-only cache analysis. Full fitting multiset uses original full-pool class weights normalized within task; subset weights exactly balance their selected classes. Cached predictions do not locate input-space decision boundaries. Margin quintiles describe existing data, not a new selection or training rule. Replacement rows are not independent evaluation.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target), 'seconds', result['seconds'])
    print('Full stream', json.dumps(macro_full, indent=2))
    print('Subset transitions', json.dumps({p:v['transitions'] for p,v in macro_subsets.items()}, indent=2))


if __name__ == '__main__':
    main()
