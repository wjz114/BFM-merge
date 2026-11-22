"""Fixed-update diagnostic: operator rows versus equal-sized shuffled groups.

Projects an already completed CE repair, not a new trainable merging method.
The partition and all four interventions are fixed before seeing results.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES, group_for_name

R = Path('runs/paper_merge_v2')
PLAN = R/'inputs/row_expert_span_plan.json'
OUT = R/'seed0/row_expert_span'
CONDITIONS = ['row_parallel', 'row_orthogonal', 'shuffled_parallel', 'shuffled_orthogonal']


def write_new(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


def project_rows(matrix, target):
    """Independent FP64 row groups: matrix [groups, experts, coordinates]."""
    assert matrix.dtype == target.dtype == torch.float64
    gram = matrix @ matrix.transpose(-1, -2)
    rhs = (matrix @ target.unsqueeze(-1)).squeeze(-1)
    values, vectors = torch.linalg.eigh(gram)
    cutoff = values[:, -1:].clamp_min(0)*1e-10
    keep = values > cutoff
    inverse = torch.where(keep, values, torch.ones_like(values)).reciprocal()*keep
    coefficients = (vectors @ ((vectors.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)*inverse).unsqueeze(-1)).squeeze(-1)
    parallel = (coefficients.unsqueeze(1) @ matrix).squeeze(1)
    residual = target-parallel
    normal_error = ((gram @ coefficients.unsqueeze(-1)).squeeze(-1)-rhs).norm(dim=-1)/rhs.norm(dim=-1).clamp_min(1e-30)
    return parallel, residual, coefficients, keep.sum(-1), normal_error


def check_toy():
    g = torch.Generator().manual_seed(20261119)
    a = torch.randn(4, 3, 11, generator=g, dtype=torch.float64)
    a[1, 2] = a[1, 0]+a[1, 1]
    a[2].zero_()
    target = torch.randn(4, 11, generator=g, dtype=torch.float64)
    p, o, _, rank, error = project_rows(a, target)
    reference = torch.stack([x.T @ torch.linalg.lstsq(x.T, y, driver='gelsd', rcond=1e-5).solution for x, y in zip(a, target)])
    assert torch.allclose(p, reference, atol=1e-12, rtol=1e-12)
    assert torch.allclose(p+o, target, atol=1e-14, rtol=1e-14)
    assert torch.allclose((p*o).sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert rank.tolist() == [3, 2, 0, 3] and error.max() < 1e-12
    perm = torch.randperm(44, generator=g)
    ap = a.permute(1, 0, 2).reshape(3, -1)[:, perm].reshape(3, 4, 11).permute(1, 0, 2)
    yp = target.flatten()[perm].reshape(4, 11)
    pp, oo, _, _, _ = project_rows(ap, yp)
    recovered = torch.empty(44, dtype=torch.float64)
    recovered[perm] = (pp+oo).flatten()
    assert torch.allclose(recovered, target.flatten(), atol=1e-14, rtol=1e-14)
    return {'gelsd_max_difference': float((p-reference).abs().max()),
            'ranks': rank.tolist(), 'permutation_recomposition_passed': True}


def prepare():
    torch.set_num_threads(4)
    source_path = R/'seed0/full_control_coverage_full_balanced_ce/run.json'
    source = json.loads(source_path.read_text())
    bank_path = Path(source['plan']['source_fit'])/'run.json'
    bank = json.loads(bank_path.read_text())
    reference_path = R/'seed0/full_pool_span_diagnostic/run.json'
    reference = json.loads(reference_path.read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    model = load_leaf(config, Path(source['plan']['initial_checkpoint']))
    active = [n for n, _ in model.named_parameters() if not n.startswith(UNUSED_PREFIXES)]
    eligible = {name+'.weight': {'shape': list(module.weight.shape), 'rows': module.weight.shape[0],
                                'coordinates_per_row': module.weight[0].numel(), 'module_type': type(module).__name__}
                for name, module in model.named_modules()
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and name+'.weight' in active}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            for attribute in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                weight = getattr(module, attribute, None)
                key = name+'.'+attribute
                if weight is not None and key in active:
                    eligible[key] = {'shape': list(weight.shape), 'rows': weight.shape[0],
                                     'coordinates_per_row': weight[0].numel(),
                                     'module_type': 'MultiheadAttention', 'operator_role': attribute}
    checkpoints = {'initial': source['plan']['initial_checkpoint'],
                   'full_update': str(source_path.parent/'merged.ckpt'),
                   'tensor_parallel': reference['exported_paths']['parallel_only']}
    files = [source_path, bank_path, reference_path, reference_path.parent/'geometry.json',
             Path(source['plan']['calibration']), Path(bank['base_checkpoint']),
             *map(Path, bank['experts'].values()), *map(Path, checkpoints.values()),
             *[reference_path.parent/f'{t}_logits.pt' for t in bank['datasets']]]
    code = [Path(__file__), Path('adapt_experts_ri.py'), Path('aligned_support.py'),
            Path('cache_preprojection_representations.py'), Path('check_merge_export_margins.py'),
            Path('d_downstream_qformer_ce.py'), Path('diagnose_sensor_subspaces.py'),
            Path('load_config.py'), Path('load_datasets.py'), Path('merge_poc.py'), Path('LEAF.py'),
            Path('configs/LEAF_mpnet.yaml'), Path('configs/env.yaml')]
    total_rows = sum(512+source['dataset_audits'][t]['validation_size'] for t in bank['datasets'])
    plan = {'fixed_unix': time.time(), 'source': str(source_path), 'bank': str(bank_path),
            'reference': str(reference_path), 'calibration': source['plan']['calibration'],
            'checkpoints': checkpoints, 'datasets': bank['datasets'], 'active': active,
            'eligible': eligible, 'conditions': CONDITIONS, 'batch_size': 128,
            'source_sha256': {str(p): sha256(p) for p in code},
            'input_sha256': {str(p): sha256(p) for p in files}, 'toy_checks': check_toy(),
            'budget': {'optimizer_updates': 0, 'teacher_forward_examples': 0,
                       'native_forward_examples': total_rows*6},
            'partition': 'Rows of active Linear weights, fused/separate MultiheadAttention projection weights, and flattened output-channel rows of active Conv2d weights. Other active tensors use the historical tensorwise projection unchanged, including biases, norms, positions and learned queries. Shuffled control permutes all coordinates within each eligible tensor before forming groups with the same count and size; one shared permutation for all experts and target. Seed from SHA256(name), base integer20261119. Nominal coefficient counts match; numerical ranks are reported.',
            'target': 'Fixed historical full-pool1600-step CE mean minus its exact initial merger, calculated inFP64. Expert-minus-base differences useFP32 subtraction as in the original merger. No Joint weights or predictions enter construction.',
            'solver': 'Batched FP64 Gram eigendecomposition, eigenvalue relative cutoff1e-10,128groups per chunk. Check normal residual, Pythagorean identity and containment of the historical tensor-span projection. No ridge, scale or rank search.',
            'evaluation': 'All16 already exposed validation sets plus original512 training rows, all four Pdelta/(I-P)delta interventions and fresh initial/full comparators. Fixed scale1. Compare row against shuffled partitions and report all tasks, macro and task-deletion sensitivity. No test data. Native ordinary checkpoint export and unchanged inactive tensors.',
            'scope': 'Subsequent fixed-update diagnostic after the raw-feature KD controls. It uses the learned full-CE update and pays that training cost; it is not a new algorithm, attainable-capacity bound, efficient replacement for full CE, or independent confirmation. Generic finer-grained merging has prior art including NP Merge. Structural grouping must be distinguished from increased coefficient count.'}
    write_new(PLAN, plan)
    print(json.dumps({'plan': str(PLAN), 'sha256': sha256(PLAN), 'eligible_tensors': len(eligible),
                      'budget': plan['budget'], 'toy': plan['toy_checks']}, indent=2))


def verify_plan(plan):
    for p, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(p)) == digest, p


@torch.no_grad()
def construct():
    started = time.time()
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text()); verify_plan(plan)
    OUT.mkdir(exist_ok=False)
    bank = json.loads(Path(plan['bank']).read_text())
    base = torch.load(bank['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = [torch.load(bank['experts'][t], map_location='cpu', weights_only=True) for t in plan['datasets']]
    states = {k: torch.load(p, map_location='cpu', weights_only=True) for k, p in plan['checkpoints'].items()}
    initial, full, whole = [states[k] for k in ['initial', 'full_update', 'tensor_parallel']]
    old = json.loads((Path(plan['reference']).parent/'geometry.json').read_text())
    outputs = {c: {n: v.clone() for n, v in initial.items()} for c in CONDITIONS}
    records, coefficients = {}, {}
    for index, name in enumerate(plan['active']):
        target = (full[name].double()-initial[name].double()).flatten()
        records[name] = {'block': group_for_name(name), 'eligible': name in plan['eligible'],
                         'delta_energy': float(target.square().sum()),
                         'tensor_parallel_energy': old['tensors'][name]['parallel_energy'], 'partitions': {}}
        if name not in plan['eligible']:
            parallel = (whole[name].double()-initial[name].double()).flatten()
            for partition in ['row', 'shuffled']:
                outputs[partition+'_parallel'][name] = whole[name].clone()
                outputs[partition+'_orthogonal'][name] = (full[name].double()-parallel.reshape(full[name].shape)).float()
                records[name]['partitions'][partition] = {
                    'parallel_energy': old['tensors'][name]['parallel_energy'],
                    'residual_energy': old['tensors'][name]['orthogonal_energy'],
                    'coefficient_count': 16, 'rank_sum': old['tensors'][name]['rank'],
                    'copied_historical_projection': True}
            continue
        spec = plan['eligible'][name]; nr, width = spec['rows'], spec['coordinates_per_row']
        matrix = torch.stack([(s[name]-base[name]).flatten() for s in experts]).double()
        seed = (int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)+20261119) % (2**63-1)
        permutation = torch.randperm(target.numel(), generator=torch.Generator().manual_seed(seed))
        for partition in ['row', 'shuffled']:
            a = matrix if partition == 'row' else matrix[:, permutation]
            y = target if partition == 'row' else target[permutation]
            a = a.reshape(16, nr, width).permute(1, 0, 2)
            y = y.reshape(nr, width)
            ps, cs, ranks, errors = [], [], [], []
            for start in range(0, nr, 128):
                p, _, c, rank, error = project_rows(a[start:start+128], y[start:start+128])
                ps.append(p); cs.append(c); ranks.append(rank); errors.append(error)
            p = torch.cat(ps).flatten()
            if partition == 'shuffled':
                restored = torch.empty_like(p); restored[permutation] = p; p = restored
            residual = target-p
            energy, pe, oe = [float(v.square().sum()) for v in [target, p, residual]]
            cross = float((p*residual).sum())
            error = float(torch.cat(errors).max())
            assert error < 1e-6, (name, partition, error)
            assert abs(energy-pe-oe) < max(1e-12, energy*1e-7), (name, partition)
            assert pe+max(1e-12, energy*1e-7) >= records[name]['tensor_parallel_energy'], (name, partition, 'containment')
            assert torch.equal((initial[name].double()+(p+residual).reshape(initial[name].shape)).float(), full[name])
            for suffix, displacement in [('parallel', p), ('orthogonal', residual)]:
                outputs[partition+'_'+suffix][name] = (initial[name].double()+displacement.reshape(initial[name].shape)).float()
            rank = torch.cat(ranks)
            records[name]['partitions'][partition] = {
                'parallel_energy': pe, 'residual_energy': oe, 'cross_dot': cross,
                'coefficient_count': 16*nr, 'rank_sum': int(rank.sum()),
                'rank_histogram': torch.bincount(rank, minlength=17).tolist(),
                'max_normal_equation_relative_residual': error,
                'permutation_seed': seed if partition == 'shuffled' else None,
                'permutation_hash': array_hash(permutation) if partition == 'shuffled' else None}
            coefficients[name+':'+partition] = torch.cat(cs)
        if index % 30 == 0:
            print('Projected', index, name, flush=True)
    totals = {'delta_energy': sum(v['delta_energy'] for v in records.values()),
              'tensor_parallel_energy': sum(v['tensor_parallel_energy'] for v in records.values())}
    for partition in ['row', 'shuffled']:
        totals[partition] = {k: sum(v['partitions'][partition][k] for v in records.values())
                             for k in ['parallel_energy', 'residual_energy', 'coefficient_count', 'rank_sum']}
        totals[partition]['captured_energy_fraction'] = totals[partition]['parallel_energy']/totals['delta_energy']
    exported = {}
    for condition, state in outputs.items():
        assert list(state) == list(initial)
        assert all(torch.equal(v, initial[n]) for n, v in state.items() if n not in plan['active'])
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        path = OUT/f'{condition}.ckpt'; torch.save(state, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(v, restored[n]) for n, v in state.items())
        exported[condition] = {'path': str(path), 'sha256': sha256(path)}
    torch.save(coefficients, OUT/'coefficients.pt')
    result = {'plan_sha256': sha256(PLAN), 'source_sha256': sha256(Path(__file__)),
              'status': 'complete', 'totals': totals, 'tensors': records, 'exported': exported,
              'fixed_tensors_exact': True, 'native_state_keys_exact': True,
              'coefficients_sha256': sha256(OUT/'coefficients.pt'), 'seconds': time.time()-started,
              'optimizer_updates': 0, 'forward_examples': 0, 'scope': plan['scope']}
    write_new(OUT/'construction.json', result)
    print(json.dumps({'completed': True, 'totals': totals, 'seconds': result['seconds']}, indent=2), flush=True)


@torch.no_grad()
def evaluate(shard, gpu):
    started = time.time()
    plan = json.loads(PLAN.read_text()); verify_plan(plan)
    construction = json.loads((OUT/'construction.json').read_text())
    assert construction['plan_sha256'] == sha256(PLAN)
    outdir = OUT/f'evaluation_shard{shard}'; outdir.mkdir(exist_ok=False)
    device = configure(gpu, 0)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source = json.loads(Path(plan['source']).read_text())
    paths = {k: v for k, v in plan['checkpoints'].items() if k != 'tensor_parallel'}
    for key, spec in construction['exported'].items():
        assert sha256(Path(spec['path'])) == spec['sha256']; paths[key] = spec['path']
    states = {k: torch.load(p, map_location='cpu', weights_only=True) for k, p in paths.items()}
    model = load_leaf(config, Path(paths['initial'])).to(device).eval().requires_grad_(False)
    indices = np.load(plan['calibration'])
    records, forwards = {}, 0
    for task in plan['datasets'][shard::2]:
        reference = torch.load(Path(plan['reference']).parent/f'{task}_logits.pt', map_location='cpu', weights_only=True)
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            tr, ty = row_map(handle, task, 'calibration'); vr, vy = row_map(handle, task, 'validation')
            cr = [tr[int(i)] for i in indices[task]]
            x = torch.from_numpy(selected_eeg(handle, cr+vr))
            y = torch.from_numpy(np.concatenate([ty[indices[task]], vy]).astype(np.int64))
        n = len(cr)
        assert n == 512 == reference['n_calibration'] and torch.equal(y, reference['truth'])
        assert array_hash(x[:n]) == source['dataset_audits'][task]['eeg_hash']
        assert len(vy) == source['dataset_audits'][task]['validation_size']
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
        logits, evaluations, replay = {}, {}, {}
        for condition, state in states.items():
            model.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(x[i:i+128].to(device)).cpu() for i in range(0, len(x), 128)])
            assert torch.isfinite(values).all()
            assert all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items())
            forwards += len(x); logits[condition] = values
            evaluations[condition] = {split: classification_metrics(values[s], y[s])
                for split, s in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
            if condition in ['initial', 'full_update']:
                replay[condition] = compare(reference['logits'][condition], values, y)
        path = outdir/f'{task}_logits.pt'
        torch.save({'truth': y, 'n_calibration': n, 'logits': logits}, path)
        records[task] = {'evaluations': evaluations, 'native_reference_replay': replay,
                         'selected_eeg_hash': array_hash(x), 'truth_hash': array_hash(y),
                         'logits_path': str(path), 'logits_sha256': sha256(path)}
        (outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, {k: v['validation']['BAcc'] for k, v in evaluations.items()}, flush=True)
    result = {'status': 'complete', 'plan_sha256': sha256(PLAN), 'code_sha256': sha256(Path(__file__)),
              'construction_sha256': sha256(OUT/'construction.json'), 'records': records,
              'forward_examples': forwards, 'all_states_unchanged': True, 'test_data_read': False,
              'optimizer_updates': 0, 'seconds': time.time()-started, 'scope': plan['scope']}
    write_new(outdir/'run.json', result)
    print('Completed', result['seconds'], flush=True)


def report():
    plan = json.loads(PLAN.read_text())
    records, sources, forwards = {}, {str(PLAN): sha256(PLAN)}, 0
    for shard in [0, 1]:
        path = OUT/f'evaluation_shard{shard}/run.json'; data = json.loads(path.read_text())
        assert data['status'] == 'complete' and data['plan_sha256'] == sha256(PLAN)
        assert data['all_states_unchanged'] and not data['test_data_read']
        assert data['code_sha256'] == sha256(Path(__file__))
        assert data['construction_sha256'] == sha256(OUT/'construction.json')
        sources[str(path)] = sha256(path); records.update(data['records']); forwards += data['forward_examples']
    assert set(records) == set(plan['datasets']) and forwards == plan['budget']['native_forward_examples']
    models = ['initial', 'full_update', *CONDITIONS]
    macro = {s: {m: float(np.mean([records[t]['evaluations'][m][s]['BAcc'] for t in plan['datasets']])) for m in models}
             for s in ['calibration', 'validation']}
    contrasts = {}
    for split in ['calibration', 'validation']:
        contrasts[split] = {}
        for a, b in [(m, 'initial') for m in models[1:]]+[(f'row_{s}', f'shuffled_{s}') for s in ['parallel', 'orthogonal']]:
            delta = {t: 100*(records[t]['evaluations'][a][split]['BAcc']-records[t]['evaluations'][b][split]['BAcc']) for t in plan['datasets']}
            values = list(delta.values()); deletion = [(sum(values)-v)/15 for v in values]
            contrasts[split][a+'_vs_'+b] = {'macro_gain_pp': float(np.mean(values)), 'per_task_gain_pp': delta,
                'positive_tasks': sum(v > 0 for v in values), 'negative_tasks': sum(v < 0 for v in values),
                'task_deletion_range_pp': [min(deletion), max(deletion)]}
    construction = json.loads((OUT/'construction.json').read_text())
    sources[str(OUT/'construction.json')] = sha256(OUT/'construction.json')
    result = {'sources': sources, 'datasets': plan['datasets'], 'records': records, 'macro': macro,
              'contrasts': contrasts, 'geometry_totals': construction['totals'],
              'forward_examples': forwards, 'optimizer_updates': 0, 'scope': plan['scope']}
    out = R/'analysis/row_expert_span_summary.json'; write_new(out, result)
    print(json.dumps({'output': str(out), 'sha256': sha256(out), 'macro': macro,
                      'row_vs_shuffle': contrasts['validation']['row_parallel_vs_shuffled_parallel'],
                      'geometry': construction['totals']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'construct', 'evaluate', 'report'])
    parser.add_argument('--shard', type=int, choices=[0, 1]); parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'construct': construct()
    elif args.mode == 'evaluate':
        assert args.shard is not None and args.gpu is not None
        evaluate(args.shard, args.gpu)
    else: report()
