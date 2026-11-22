"""Functionally ablate a fixed repair update inside/outside an augmented pilot space."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_ri_span import project
from diagnose_sensor_projection import calibration_and_validation
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES, group_for_name


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == expected, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    source_dir = Path(plan['source'])
    source = json.loads((source_dir/'run.json').read_text())
    bank = json.loads((Path(source['plan']['source_fit'])/'run.json').read_text())
    basis_path = Path(plan['basis_run'])
    basis = json.loads(basis_path.read_text())
    assert basis['plan']['source_fit'] == str(Path(source['plan']['source_fit'])/'run.json')
    cache = json.loads(Path(basis['gradient_cache']).read_text())
    assert sha256(Path(basis['gradient_cache'])) == basis['gradient_cache_sha256']
    cached_initial = cache['plan']['initial_checkpoint']
    assert cache['plan']['input_sha256'][cached_initial] == source['plan']['initial_checkpoint_sha256']
    assert plan['basis_kind'] in ['gradient', 'random']
    extra_record = basis['outputs'][plan['basis_kind']]
    assert sha256(Path(extra_record['path'])) == extra_record['sha256']
    extra = torch.load(extra_record['path'], map_location='cpu', weights_only=True)
    tasks = source['datasets']
    assert tasks == bank['datasets']
    base = torch.load(bank['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {t: torch.load(bank['experts'][t], map_location='cpu', weights_only=True) for t in tasks}
    initial = torch.load(source['plan']['initial_checkpoint'], map_location='cpu', weights_only=True)
    full = torch.load(source_dir/'merged.ckpt', map_location='cpu', weights_only=True)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, source_dir/'merged.ckpt').to(device).eval().requires_grad_(False)
    active = [n for n, _ in leaf.named_parameters() if not n.startswith(UNUSED_PREFIXES)]
    states = {'initial': initial, 'full_update': full, 'parallel_only': dict(initial), 'orthogonal_only': dict(initial)}
    geometry, totals, blocks, coefficients = {}, {}, {}, []
    max_reconstruction = 0.
    for name in active:
        matrix = torch.stack([(experts[t][name]-base[name]).flatten() for t in tasks]).double()
        matrix = torch.cat([matrix, extra[name].reshape(16, -1).double()])
        delta = (full[name].double()-initial[name].double()).flatten().unsqueeze(0)
        parallel, orthogonal, coefficient, solver = project(matrix, delta)
        assert solver['normal_equation_relative_residual'] < 1e-6, name
        assert solver['rank'] == 32, name
        _, self_error, _, _ = project(matrix, matrix[:1])
        assert float(self_error.norm()/matrix[:1].norm().clamp_min(1e-30)) < 1e-6, name
        recomposed = initial[name].double()+(parallel+orthogonal).reshape(initial[name].shape)
        error = float((recomposed-full[name].double()).abs().max())
        max_reconstruction = max(max_reconstruction, error)
        assert torch.equal(recomposed.float(), full[name]), name
        measures = {'delta_energy': float(delta.square().sum()), 'parallel_energy': float(parallel.square().sum()),
                    'orthogonal_energy': float(orthogonal.square().sum()), 'cross_dot': float((parallel*orthogonal).sum())}
        assert abs(measures['delta_energy']-measures['parallel_energy']-measures['orthogonal_energy']) < max(1e-12, measures['delta_energy']*1e-7), name
        group = blocks.setdefault(group_for_name(name), {k:0. for k in measures})
        for key, value in measures.items():
            totals[key] = totals.get(key, 0.)+value
            group[key] += value
        geometry[name] = {**solver, **measures}
        coefficients.append(coefficient[:,0].numpy())
        for condition, displacement in [('parallel_only', parallel), ('orthogonal_only', orthogonal)]:
            states[condition][name] = (initial[name].double()+displacement.reshape(initial[name].shape)).float()
    totals['orthogonal_energy_fraction'] = totals['orthogonal_energy']/totals['delta_energy']
    paths = {'initial': source['plan']['initial_checkpoint'], 'full_update': str(source_dir/'merged.ckpt')}
    for condition in ['parallel_only', 'orthogonal_only']:
        state = states[condition]
        assert all(torch.equal(state[n], initial[n]) for n in initial if n not in active)
        assert all(torch.isfinite(state[n]).all() for n in active)
        path = args.outdir/f'{condition}.ckpt'
        torch.save(state, path)
        states[condition] = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(state[n], states[condition][n]) for n in state)
        paths[condition] = str(path)
    np.savez(args.outdir/'coefficients.npz', coefficients=np.stack(coefficients), tensor_names=np.array(active), original_expert_tasks=np.array(tasks),
             basis_row_labels=np.array(tasks+[plan['basis_kind']+'_basis_'+str(i) for i in range(16)]))
    (args.outdir/'geometry.json').write_text(json.dumps({'totals':totals, 'blocks':blocks, 'tensors':geometry, 'max_reconstruction_error':max_reconstruction}, indent=2)+'\n')
    del experts, base
    print('Geometry', totals, flush=True)
    indices = np.load(source['plan']['calibration'])
    evaluations = {c: {} for c in states}
    for task in tasks:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        assert array_hash(cx) == source['dataset_audits'][task]['eeg_hash']
        assert len(vy) == source['dataset_audits'][task]['validation_size']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), 128, False)
        logits = {}
        for condition, state in states.items():
            leaf.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            assert torch.isfinite(values).all()
            logits[condition] = values
            evaluations[condition][task] = {split:classification_metrics(values[ids],truth[ids]) for split,ids in [('calibration',slice(0,n)),('validation',slice(n,None))]}
        for condition, ref in [('initial',source['initial_validation']['validation']),('full_update',source['average_validation'])]:
            assert abs(evaluations[condition][task]['validation']['BAcc']-ref[task]['BAcc']) < 1e-12
        assert abs(evaluations['full_update'][task]['calibration']['BAcc']-source['average_calibration'][task]['BAcc']) < 1e-12
        torch.save({'truth':truth, 'n_calibration':n, 'logits':logits},args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations,indent=2)+'\n')
        print(task,{c:100*(evaluations[c][task]['validation']['BAcc']-evaluations['initial'][task]['validation']['BAcc']) for c in states if c!='initial'},flush=True)
    summaries, contrasts = {}, {}
    for split in ['calibration','validation']:
        summaries[split] = {c:float(np.mean([evaluations[c][t][split]['BAcc'] for t in tasks])) for c in states}
        contrasts[split] = {}
        for treatment,control in [(c,'initial') for c in states if c!='initial']+[('full_update','parallel_only'),('full_update','orthogonal_only')]:
            gains = {t:100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in tasks}
            contrasts[split][treatment+'_vs_'+control] = {'macro_gain_pp':float(np.mean(list(gains.values()))),'per_task_gain_pp':gains,'improved':sum(v>0 for v in gains.values()),'declined':sum(v<0 for v in gains.values())}
        summaries[split]['factorial_interaction_pp'] = 100*(summaries[split]['full_update']-summaries[split]['parallel_only']-summaries[split]['orthogonal_only']+summaries[split]['initial'])
    result = {'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
              'basis_kind':plan['basis_kind'],'basis_source':{'path':str(basis_path),'sha256':sha256(basis_path)},
              'geometry_totals':totals,'evaluations':evaluations,'summaries':summaries,'contrasts':contrasts,
              'exported_paths':paths,'checkpoint_sha256':{c:sha256(Path(p)) for c,p in paths.items()},
              'exact_validation_bacc_replays':True,'fixed_tensors_exact':True,'exact_fp32_update_recomposition':True,
              'seconds':time.time()-started,'test_data_read':False,'test_inference_performed':False,
              'scope':'Exploratory functional decomposition of the existing full-pool averaged repair into the original expert span plus a fixed gradient/random extension and its complement. Basis used old512 gradients; reference update used full-pool full-weight training. No new fitting, scaling or Joint assistance. This uses a completed full-weight repair and is a coordinate-dependent diagnostic, not a standalone candidate, capacity bound, independent confirmation or new algorithm.'}
    (args.outdir/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed',summaries['validation'],flush=True)


if __name__ == '__main__':
    main()
