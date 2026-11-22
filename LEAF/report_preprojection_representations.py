"""Compare class variation, head-visible subspaces and paired representation structure."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_teacher_output_compression import moments


def residual(x, weights, group=None):
    if group is None:
        fitted = (weights[:, None]*x).sum(0)
    else:
        _, fitted = moments(x, group, weights)
    return x-fitted


def gram(x, weights, group=None):
    centered = residual(x, weights, group)*weights.sqrt()[:, None]
    return centered@centered.T


def cka(a, b):
    denominator = float(a.norm()*b.norm())
    if denominator == 0:
        return None
    value = float((a*b).sum())/denominator
    assert -1e-12 <= value <= 1+1e-12
    return value


def row_space(weight, memo):
    key = array_hash(weight)
    if key not in memo:
        w = weight.double()
        _, singular, vh = torch.linalg.svd(w, full_matrices=False)
        cutoff = torch.finfo(torch.float64).eps*max(w.shape)*singular[0]
        rank = int((singular > cutoff).sum())
        basis = vh[:rank].T.contiguous()
        orth_error = float((basis.T@basis-torch.eye(rank, dtype=torch.float64)).abs().max())
        assert orth_error < 1e-10
        memo[key] = basis, {'rank': rank, 'threshold': float(cutoff),
                           'singular_value_range': [float(singular[-1]), float(singular[0])],
                           'orthogonality_max_abs_error': orth_error, 'weight_hash': key}
    return memo[key]


def subspace_variance(x, weight, basis, weights, group):
    centered = residual(x, weights, group)
    coordinates = centered@basis
    null = centered-coordinates@basis.T
    total = float((weights[:, None]*centered.square()).sum())
    visible = float((weights[:, None]*coordinates.square()).sum())
    hidden = float((weights[:, None]*null.square()).sum())
    assert abs(total-visible-hidden) < 1e-10*max(1., total)
    denominator = float(null.norm()*weight.norm())
    annihilation = float((null@weight.T).norm())/denominator if denominator else 0.
    assert annihilation < 1e-10
    return {'total_variance': total, 'row_space_variance': visible, 'null_space_variance': hidden,
            'null_fraction': hidden/total if total else None, 'relative_null_annihilation': annihilation}


def summary(values, tasks):
    valid = [(t, v) for t,v in zip(tasks, values) if v is not None]
    if not valid:
        return {'defined_tasks': [], 'undefined_tasks': tasks, 'mean': None}
    names, vals = zip(*valid)
    a = np.asarray(vals)
    return {'defined_tasks': list(names), 'undefined_tasks': [t for t in tasks if t not in names],
            'mean': float(a.mean()), 'median': float(np.median(a)), 'range': [float(a.min()), float(a.max())],
            'positive': int((a > 0).sum()), 'negative': int((a < 0).sum()), 'ties': int((a == 0).sum()),
            'task_deletion_means': {t:float(np.delete(a, i).mean()) for i,t in enumerate(names)} if len(a)>1 else {}}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    target = root/'analysis/preprojection_representation_summary.json'
    assert not target.exists()
    design_path = root/'inputs/preprojection_design.json'
    design = json.loads(design_path.read_text())
    source_paths = [design_path, Path(__file__), Path('diagnose_teacher_output_compression.py'), Path('aligned_support.py')]
    sources = {str(p): sha256(p) for p in source_paths}
    compression = json.loads((root/'analysis/teacher_output_compression.json').read_text())
    assert sha256(Path('diagnose_teacher_output_compression.py')) == compression['code_sha256']
    records, runs = {}, []
    for shard in design['shards'].values():
        path = Path(shard['outdir'])/'run.json'
        plan_path = Path(shard['plan'])
        run, plan = [json.loads(p.read_text()) for p in [path, plan_path]]
        assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
        assert plan['design_sha256'] == sha256(design_path)
        assert run['code_sha256'] == sha256(Path(design['worker']))
        for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
            assert sha256(Path(name)) == digest, name
        assert run['all_model_states_unchanged'] and run['optimizer_updates'] == run['backward_passes'] == 0
        assert not run['validation_inference_performed'] and not run['test_data_read']
        records.update(run['records'])
        runs.append(run)
        sources.update({str(p):sha256(p) for p in [path, plan_path]})
    assert sum(r['main_forward_examples'] for r in runs) == design['budget']['main_forward_examples']
    assert set(records) == set(design['all_tasks'])
    memo, results = {}, {}
    for ti, task in enumerate(design['all_tasks']):
        spec = records[task]
        path = Path(spec['path'])
        assert sha256(path) == spec['sha256']
        sources[str(path)] = spec['sha256']
        assert spec['teacher_state_unchanged'] and spec['exact_head_reconstruction']
        for check in spec['native_logit_replay'].values():
            assert check['max_abs_logit_error'] < 2e-5
            assert all(r['margin_within_twice_error'] for r in check['changed_rows'])
        data = torch.load(path, map_location='cpu', weights_only=True)
        n = data['n_original']
        assert n == 128 and len(data['truth']) == 256
        y0 = data['truth'][:n]
        assert torch.equal(y0, data['truth'][n:])
        rng = np.random.default_rng([0, ti, 20261117])
        permutations = []
        for _ in range(32):
            order = np.arange(n)
            for c in y0.unique().tolist():
                ids = np.flatnonzero(y0.numpy() == c)
                order[ids] = rng.permutation(ids)
            assert torch.equal(y0, y0[order])
            permutations.append(torch.from_numpy(order))
        pools = {}
        for pool, selection in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            y = data['truth'][selection]
            weights = 1/(len(y.unique())*torch.bincount(y).double())[y]
            assert abs(float(weights.sum())-1) < 1e-12
            teacher_prediction = data['logits']['teacher'][selection].argmax(-1)
            models, grams = {}, {}
            for model, features in data['features'].items():
                x = {key:value[selection].double() for key,value in features.items()}
                logits = data['logits'][model][selection].double()
                x['centered_logits'] = logits-logits.mean(-1, keepdim=True)
                variance = {group_name: {rep:moments(value, group, weights)[0] for rep,value in x.items()}
                            for group_name,group in [('truth', y), ('teacher_argmax', teacher_prediction)]}
                basis, basis_meta = row_space(data['head_states'][model]['weight'], memo)
                w = data['head_states'][model]['weight'].double()
                subspaces = {name:subspace_variance(x['prehead'], w, basis, weights, group)
                             for name,group in [('global', None), ('within_truth', y), ('within_teacher_argmax', teacher_prediction)]}
                assert abs(subspaces['global']['total_variance']-variance['truth']['prehead']['total_variance']) < 1e-10
                models[model] = {'variance': variance, 'head_row_space': basis_meta, 'prehead_subspaces': subspaces}
                grams[model] = {rep:{'global':gram(value, weights), 'within_truth':gram(value, weights, y)}
                                for rep,value in x.items()}
            agreements = {}
            for model in ['initial', 'ce', 'kl']:
                agreements[model] = {}
                for rep in ['prehead', 'raw', 'unit', 'centered_logits']:
                    a, b = grams['teacher'][rep], grams[model][rep]
                    paired = cka(a['within_truth'], b['within_truth'])
                    null = [cka(a['within_truth'], b['within_truth'][order][:, order]) for order in permutations]
                    assert (paired is None) == any(v is None for v in null)
                    agreements[model][rep] = {'global_cka': cka(a['global'], b['global']), 'within_truth_cka': paired,
                        'within_truth_permutation_cka': null, 'within_truth_permutation_mean': float(np.mean(null)) if paired is not None else None,
                        'paired_minus_permutation_mean': paired-float(np.mean(null)) if paired is not None else None,
                        'teacher_within_gram_frobenius_norm': float(a['within_truth'].norm()),
                        'student_within_gram_frobenius_norm': float(b['within_truth'].norm())}
            pools[pool] = {'class_counts': torch.bincount(y).tolist(), 'models': models, 'teacher_student_cka': agreements}
        results[task] = pools
        print('Analyzed', task, flush=True)
    summaries = {}
    tasks = design['all_tasks']
    for pool in ['original', 'replacement']:
        summaries[pool] = {'teacher_within_variance_fraction': {}, 'teacher_prehead_null_fraction': {},
                           'teacher_student_cka': {}, 'kl_minus_ce_cka': {}}
        for rep in ['prehead', 'raw', 'unit', 'centered_logits']:
            summaries[pool]['teacher_within_variance_fraction'][rep] = summary(
                [results[t][pool]['models']['teacher']['variance']['truth'][rep]['within_fraction'] for t in tasks], tasks)
        for centering in ['global', 'within_truth', 'within_teacher_argmax']:
            summaries[pool]['teacher_prehead_null_fraction'][centering] = summary(
                [results[t][pool]['models']['teacher']['prehead_subspaces'][centering]['null_fraction'] for t in tasks], tasks)
        for model in ['initial', 'ce', 'kl']:
            summaries[pool]['teacher_student_cka'][model] = {rep:{metric:summary(
                [results[t][pool]['teacher_student_cka'][model][rep][metric] for t in tasks], tasks)
                for metric in ['global_cka', 'within_truth_cka', 'paired_minus_permutation_mean']}
                for rep in ['prehead', 'raw', 'unit', 'centered_logits']}
        for rep in ['prehead', 'raw', 'unit', 'centered_logits']:
            summaries[pool]['kl_minus_ce_cka'][rep] = {}
            for metric in ['global_cka', 'within_truth_cka', 'paired_minus_permutation_mean']:
                pairs = [(results[t][pool]['teacher_student_cka']['kl'][rep][metric],
                          results[t][pool]['teacher_student_cka']['ce'][rep][metric]) for t in tasks]
                summaries[pool]['kl_minus_ce_cka'][rep][metric] = summary(
                    [a-b if a is not None and b is not None else None for a,b in pairs], tasks)
    result = {'unix':time.time(), 'code_sha256':sha256(Path(__file__)), 'source_sha256':sources,
              'datasets':tasks, 'per_task':results, 'summaries':summaries, 'unique_head_row_spaces':len(memo),
              'cache_forward_examples':sum(r['main_forward_examples'] for r in runs),
              'cache_seconds':[r['seconds'] for r in runs], 'analysis_seconds':time.time()-started,
              'scope':design['scope']+' Linear CKA and within-class permutation comparisons are descriptive geometry, not measures of semantic knowledge, causal efficacy or independent generalization.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target), 'seconds', result['analysis_seconds'])
    for pool,row in summaries.items():
        print(pool, 'teacher within-class variance fractions', {k:v['mean'] for k,v in row['teacher_within_variance_fraction'].items()})
        print(pool, 'teacher prehead null fractions', {k:v['mean'] for k,v in row['teacher_prehead_null_fraction'].items()})
        print(pool, 'prehead CKA excess', {m:x['prehead']['paired_minus_permutation_mean']['mean'] for m,x in row['teacher_student_cka'].items()})


if __name__ == '__main__':
    main()
