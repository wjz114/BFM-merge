"""Test calibration-derived sensor-nullspace canonicalization in a fixed merger.

No optimization, coefficient selection, rank search, or runtime model bank.
This is an exploratory structural intervention, not a novelty claim.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from load_datasets import PREDEFINED_SPLITS, SUBJECT_SPLITS, filter_classes, load_by_index, split_train_valid


TEMPORAL_BIAS = 'tower.tokenizer.enc_conv.0.bias'
SPATIAL_WEIGHT = 'tower.tokenizer.enc_conv.1.weight'
SPATIAL_BIAS = 'tower.tokenizer.enc_conv.1.bias'
CHANGED = [SPATIAL_WEIGHT, SPATIAL_BIAS]
CONDITIONS = ['canonical', 'random_removed_norm', 'isotropic_final_norm']


def projected_parameters(expert, base, projector, task_index):
    b = expert[SPATIAL_WEIGHT][..., 0].double()
    b0 = base[SPATIAL_WEIGHT][..., 0].double()
    delta = b-b0
    null = torch.as_tensor(projector, dtype=torch.float64, device=b.device)
    removed = delta@null
    retained = delta-removed
    nullity = int(round(float(null.trace())))
    rng = np.random.default_rng([0, task_index, 817])
    q, _ = np.linalg.qr(rng.standard_normal((65, 65)))
    random_basis = torch.as_tensor(q[:, :nullity], dtype=torch.float64, device=b.device)
    random_removed = delta@(random_basis@random_basis.T)
    random_scale = float(removed.norm()/random_removed.norm()) if float(random_removed.norm()) > 0 else 0.0
    shrink_scale = float(retained.norm()/delta.norm()) if float(delta.norm()) > 0 else 1.0
    assert shrink_scale <= 1+1e-10
    edits = {'canonical': -removed, 'random_removed_norm': -random_scale*random_removed,
             'isotropic_final_norm': (shrink_scale-1)*delta}
    values, geometry = {}, {}
    for condition, change in edits.items():
        weight = (b+change).unsqueeze(-1).float()
        # Use the actual rounded FP32 weight change when compensating the bias.
        actual_change = weight[..., 0].double()-b
        bias_change = -torch.einsum('ofc,f->o', actual_change, expert[TEMPORAL_BIAS].double())
        bias = (expert[SPATIAL_BIAS].double()+bias_change).float()
        values[condition] = {SPATIAL_WEIGHT: weight, SPATIAL_BIAS: bias}
        geometry[condition] = {
            'spatial_change_norm': float(actual_change.norm()),
            'spatial_delta_to_base_norm': float((weight[..., 0].double()-b0).norm()),
            'spatial_bias_change_norm': float((bias.double()-expert[SPATIAL_BIAS].double()).norm())}
    return values, {'nullity': nullity, 'original_spatial_delta_norm': float(delta.norm()),
                    'own_null_delta_energy_fraction': float(removed.square().sum()/delta.square().sum()) if float(delta.norm())>0 else 0.0,
                    'random_scale': random_scale, 'isotropic_scale': shrink_scale,
                    'orthogonal_decomposition_relative_error': float((delta.square().sum()-removed.square().sum()-retained.square().sum()).abs()/delta.square().sum()) if float(delta.norm())>0 else 0.0,
                    'conditions': geometry}


def calibration_and_validation(task, indices):
    # Match load_dataset's exact partition without reading its official test.
    if task in PREDEFINED_SPLITS:
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            tx, ty = h['trainX'][:], h['trainY'][:]
            vx, vy = h['validX'][:], h['validY'][:]
    else:
        assert task in SUBJECT_SPLITS
        x, y = filter_classes(task, *load_by_index(task, range(SUBJECT_SPLITS[task][0])))
        tx, ty, vx, vy = split_train_valid(x, y, .2)
        del x, y
    cx, cy = np.asarray(tx[indices], dtype=np.float32), np.asarray(ty[indices], dtype=np.int64)
    return cx, cy, np.asarray(vx, dtype=np.float32), np.asarray(vy, dtype=np.int64)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    source = Path(plan['source_fit'])
    average_dir = Path(plan['source_average'])
    fit = json.loads((source/'run.json').read_text())
    average = json.loads((average_dir/'run.json').read_text())
    subspace_dir = Path(plan['subspace_source'])
    subspaces = json.loads((subspace_dir/'run.json').read_text())
    assert list(subspaces['records']) == fit['datasets']
    assert sha256(average_dir/'merged.ckpt') == plan['initial_checkpoint_sha256']
    assert sha256(source/'calibration_indices.npz') == plan['calibration_sha256']
    assert fit['precision'] == '32-true' and fit['granularity'] == 'tensor'
    alpha = np.load(average_dir/'best_alpha.npy')
    calibration = np.load(source/'calibration_indices.npz')
    base = torch.load(fit['base_checkpoint'], map_location='cpu', weights_only=True)
    merged = torch.load(average_dir/'merged.ckpt', map_location='cpu', weights_only=True)
    accum = {c: {k: merged[k].double().clone() for k in CHANGED} for c in CONDITIONS}
    geometry, expert_hashes, own_values = {}, {}, {}
    for i, task in enumerate(fit['datasets']):
        expert = torch.load(fit['experts'][task], map_location='cpu', weights_only=True)
        expert_hashes[task] = sha256(Path(fit['experts'][task]))
        data = np.load(subspace_dir/f'{task}.npz')
        values, geometry[task] = projected_parameters(expert, base, data['null_projector'], i)
        own_values[task] = values['canonical']
        for condition in CONDITIONS:
            for key in CHANGED:
                accum[condition][key] += float(alpha[i, fit['groups'].index(key)])*(values[condition][key].double()-expert[key].double())
    (args.outdir/'geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')
    states = {'original': merged}
    checkpoints = {'original': str(average_dir/'merged.ckpt')}
    for condition in CONDITIONS:
        state = {key: (accum[condition][key].float() if key in CHANGED else value.clone())
                 for key, value in merged.items()}
        assert all(torch.equal(state[k], merged[k]) for k in merged if k not in CHANGED)
        # The unchanged causal attention mask deliberately contains -infinity.
        # Require finite edited tensors; every other tensor is exact above.
        assert all(torch.isfinite(state[k]).all() for k in CHANGED)
        folder = args.outdir/condition
        folder.mkdir()
        checkpoint = folder/'merged.ckpt'
        torch.save(state, checkpoint)
        states[condition] = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert all(torch.equal(states[condition][k], state[k]) for k in state)
        checkpoints[condition] = str(checkpoint)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, average_dir/'merged.ckpt').to(device).eval()
    layers = leaf.tower.tokenizer.enc_conv
    assert isinstance(layers[0], torch.nn.Conv2d) and isinstance(layers[1], torch.nn.Conv2d)
    assert layers[0].kernel_size == (1, 40) and layers[1].kernel_size == (65, 1)
    assert layers[0].groups == layers[1].groups == 1
    reference = json.loads((Path(plan['reference_control'])/'run.json').read_text())['dataset_audits']
    evaluations, own_checks = {name: {} for name in states}, {}
    for task in fit['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, calibration[task])
        assert array_hash(cx) == reference[task]['eeg_hash'] == subspaces['records'][task]['calibration_eeg_hash']
        teacher = torch.load(Path(plan['teacher_cache'])/f'{task}.pt', map_location='cpu', weights_only=True)
        assert torch.equal(torch.from_numpy(cy), teacher['truth'][:len(cy)])
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy, teacher
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        logits = {}

        def predict(state):
            leaf.load_state_dict(state, strict=True)
            return torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])

        for condition, state in states.items():
            logits[condition] = predict(state)
            evaluations[condition][task] = {
                split: classification_metrics(logits[condition][ids], truth[ids])
                for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
        assert abs(evaluations['original'][task]['validation']['BAcc']-average['folded_val'][task]['BAcc']) < 1e-12
        expert = torch.load(fit['experts'][task], map_location='cpu', weights_only=True)
        before = predict(expert)
        expert.update(own_values[task])
        after = predict(expert)
        own_checks[task] = {
            split: {'comparison': compare(before[ids], after[ids], truth[ids]),
                    'original': classification_metrics(before[ids], truth[ids]),
                    'canonical': classification_metrics(after[ids], truth[ids])}
            for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
        torch.save({'truth': truth, 'n_calibration': n, 'merged_logits': logits,
                    'original_expert_logits': before, 'canonical_expert_logits': after}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps({'evaluations': evaluations, 'own_checks': own_checks}, indent=2)+'\n')
        print(task, 'own max error', own_checks[task]['validation']['comparison']['max_abs_logit_error'],
              'merged deltas pp', {c: 100*(evaluations[c][task]['validation']['BAcc']-evaluations['original'][task]['validation']['BAcc']) for c in CONDITIONS}, flush=True)
    contrasts = {}
    for split in ['calibration', 'validation']:
        contrasts[split] = {}
        for treatment, control in [(c, 'original') for c in CONDITIONS]+[('canonical', c) for c in CONDITIONS[1:]]:
            change = {t: 100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in fit['datasets']}
            contrasts[split][treatment+'_vs_'+control] = {
                'macro_change_pp': float(np.mean(list(change.values()))), 'per_task_change_pp': change,
                'improved': sum(v>0 for v in change.values()), 'declined': sum(v<0 for v in change.values()),
                'leave_one_task_out_pp': {t: float(np.mean([v for other,v in change.items() if other!=t])) for t in change}}
    result = {'evaluations': evaluations, 'contrasts': contrasts, 'own_checks': own_checks,
              'geometry': geometry, 'exported_paths': checkpoints,
              'checkpoint_sha256': {c: sha256(Path(p)) for c,p in checkpoints.items()},
              'expert_hashes': expert_hashes, 'subspace_run_sha256': sha256(subspace_dir/'run.json'),
              'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'seconds': time.time()-started, 'test_data_read': False, 'test_inference_performed': False,
              'scope': 'Exploratory seed0 intervention. All16 full exposed validation tasks; fixed existing coefficients. Null projectors use original calibration only. Source-function preservation must be assessed numerically and does not imply merged-function preservation. No new learned parameters or runtime bank; no novelty or independent-confirmation claim.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', {c: v['macro_change_pp'] for c,v in contrasts['validation'].items()}, flush=True)


if __name__ == '__main__':
    main()
