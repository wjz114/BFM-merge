"""Inspect ambient-gradient effects using frozen M0 gradients, on CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def project(v, record):
    """Independent batched implementation of the stored symmetric projector."""
    q = record['q'].double()
    perm = record['permutation']
    b, n = v.shape
    arranged = v if perm is None else v[:, perm]
    arranged = arranged.reshape(b, q.shape[0], q.shape[2])
    coefficients = torch.einsum('grd,bgd->bgr', q, arranged)
    result = torch.einsum('grd,bgr->bgd', q, coefficients).reshape(b, n)
    if perm is not None:
        restored = torch.empty_like(result)
        restored[:, perm] = result
        result = restored
    return result


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--plan', type=Path, required=True)
    args = args.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    plan = json.loads(args.plan.read_text())
    assert sha(__file__) == plan['worker_sha256']
    for path, expected in plan['input_sha256'].items():
        assert sha(path) == expected, path
    cache = json.loads(Path(plan['cache_manifest']).read_text())
    runtime = json.loads(Path(plan['training_plan']).read_text())
    assert cache['plan']['initial_checkpoint'] == runtime['initial_checkpoint']
    assert cache['plan']['input_sha256'][runtime['initial_checkpoint']] == runtime['initial_checkpoint_sha256']
    assert cache['initial_state_exactly_unchanged'] and cache['optimizer_updates'] == 0
    names, tasks = cache['parameter_names'], cache['datasets']
    gradients, sources = [], dict(plan['input_sha256'])
    for task in tasks:
        record = cache['records'][task]
        assert sha(record['path']) == record['sha256']
        sources[record['path']] = record['sha256']
        g = torch.load(record['path'], map_location='cpu', weights_only=True)
        assert list(g) == names
        gradients.append(g)
    labels = tasks + ['uniform_task_mean_gradient']
    manifest = json.loads(Path(plan['basis_manifest']).read_text())
    conditions, norm_errors = {}, {}
    for kind in plan['conditions']:
        source = manifest['outputs'][kind]
        assert sha(source['path']) == source['sha256']
        sources[source['path']] = source['sha256']
        payload = torch.load(source['path'], map_location='cpu', weights_only=True)
        records = payload
        assert list(records) == names
        sums = {}
        component_sums = {c: torch.zeros(len(labels), dtype=torch.float64) for c in ['tower', 'qformer']}
        def add(key, value):
            sums[key] = sums.get(key, torch.zeros(len(labels), dtype=torch.float64)) + value
        for i, name in enumerate(names):
            g = torch.stack([v[name].flatten().double() for v in gradients])
            g = torch.cat([g, g.mean(0, keepdim=True)], dim=0)
            pg = project(g, records[name])
            denominator = g.abs() + plan['adam_epsilon']
            raw = -g/denominator
            raw_from_tangent = -pg/(pg.abs()+plan['adam_epsilon'])
            update_only = project(raw, records[name])
            project_both = project(raw_from_tangent, records[name])
            tangent_contribution = project(-pg/denominator, records[name])
            normal_contribution = project(-(g-pg)/denominator, records[name])
            vectors = {'gradient':g, 'projected_gradient':pg, 'raw_proposal':raw,
                'raw_proposal_from_tangent':raw_from_tangent, 'update_only':update_only,
                'project_both':project_both, 'tangent_contribution':tangent_contribution,
                'normal_contribution':normal_contribution}
            assert all(torch.isfinite(v).all() for v in vectors.values()), name
            for key, value in vectors.items(): add(key+'_norm_sq', value.square().sum(1))
            for key, value in [('update_only',update_only),('project_both',project_both),
                               ('tangent_contribution',tangent_contribution),('normal_contribution',normal_contribution)]:
                add('gradient_dot_'+key, (g*value).sum(1))
            add('update_only_dot_project_both', (update_only*project_both).sum(1))
            add('tangent_dot_normal', (tangent_contribution*normal_contribution).sum(1))
            add('decomposition_error_sq', (update_only-tangent_contribution-normal_contribution).square().sum(1))
            component_sums[name.split('.')[0]] += g.square().sum(1)
            if (i+1)%100 == 0: print(kind,'parameters',i+1,'/',len(names),flush=True)
        summaries = {}
        for j, label in enumerate(labels):
            s = {k:float(v[j]) for k,v in sums.items()}
            norm = lambda k: s[k+'_norm_sq']**.5
            raw_norm, update_norm = norm('raw_proposal'), norm('update_only')
            other_raw, other_update = norm('raw_proposal_from_tangent'), norm('project_both')
            assert min(raw_norm, update_norm, other_raw, other_update) > 0
            error = (s['decomposition_error_sq']/s['update_only_norm_sq'])**.5
            assert error < 1e-7, (kind,label,error)
            assert s['projected_gradient_norm_sq'] <= s['gradient_norm_sq']*(1+1e-6)
            summaries[label] = {
                'gradient_norm':norm('gradient'),
                'projected_gradient_energy_fraction':s['projected_gradient_norm_sq']/s['gradient_norm_sq'],
                'update_only_restoration_scale':raw_norm/update_norm,
                'project_both_restoration_scale':other_raw/other_update,
                'update_direction_cosine':s['update_only_dot_project_both']/(update_norm*other_update),
                'update_only_gradient_cosine':s['gradient_dot_update_only']/(norm('gradient')*update_norm),
                'project_both_gradient_cosine':s['gradient_dot_project_both']/(norm('gradient')*other_update),
                'update_only_restored_directional_derivative':plan['learning_rate']*raw_norm/update_norm*s['gradient_dot_update_only'],
                'project_both_restored_directional_derivative':plan['learning_rate']*other_raw/other_update*s['gradient_dot_project_both'],
                'normal_contribution_norm_over_update_norm':norm('normal_contribution')/update_norm,
                'tangent_contribution_norm_over_update_norm':norm('tangent_contribution')/update_norm,
                'tangent_normal_cosine':s['tangent_dot_normal']/(norm('tangent_contribution')*norm('normal_contribution')),
                'unrestored_tangent_directional_derivative':s['gradient_dot_tangent_contribution'],
                'unrestored_normal_directional_derivative':s['gradient_dot_normal_contribution'],
                'decomposition_relative_l2_error':error}
        norm_errors[kind] = {}
        for c, squared in component_sums.items():
            reference = torch.tensor([cache['records'][t]['component_gradient_norm'][c] for t in tasks],dtype=torch.float64)
            err = float(((squared[:len(tasks)].sqrt()-reference).abs()/reference).max())
            assert err < 1e-5,(kind,c,err)
            norm_errors[kind][c] = err
        conditions[kind] = summaries
        del payload, records
        print(kind,'complete','mean-direction cosine',summaries[labels[-1]]['update_direction_cosine'],flush=True)
    result = {'unix':time.time(),'plan':plan,'plan_sha256':sha(args.plan),'worker_sha256':sha(__file__),
        'input_sha256':sources,'torch_version':torch.__version__,'labels':labels,
        'conditions':conditions,'cached_norm_max_relative_errors':norm_errors,
        'seconds':time.time()-started,'new_leaf_forwards':0,'new_backpropagations':0,
        'new_optimizer_updates':0,'new_checkpoint_exports':0,
        'scope':plan['scope']}
    with Path(plan['output']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'output':plan['output'],'sha256':sha(plan['output']),'seconds':result['seconds']}),flush=True)


if __name__=='__main__': main()
