"""Frozen conditional-affine response matching and correspondence control."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_sensor_projection import calibration_and_validation
from load_config import build_model_config, load_yaml

W, B = 'qformer.film.weight', 'qformer.film.bias'


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
    for key in ['geometry', 'arrays', 'initial_checkpoint', 'joint_checkpoint']:
        assert sha256(Path(plan[key])) == plan[key+'_sha256'], key
    args.outdir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    device = configure(args.gpu, 0)
    geometry = json.loads(Path(plan['geometry']).read_text())
    arrays = np.load(plan['arrays'])
    q, fit, initial = [torch.from_numpy(arrays[k]) for k in ['augmented_instructions', 'fitted_affine', 'original_merged_affine']]
    first = [geometry['tasks'].index(group[0]) for group in geometry['instruction_groups']]
    unique = q[:, first]
    corrections = (fit-initial)@unique
    rng = np.random.default_rng([0,823])
    permutation = rng.permutation(len(first))
    while np.any(permutation == np.arange(len(first))):
        permutation = rng.permutation(len(first))
    sham_delta = corrections[:, permutation]@torch.linalg.pinv(unique, rtol=1e-10)
    response_norm = ((fit-initial)@q).norm()
    sham_scale = float(response_norm/(sham_delta@q).norm())
    sham = initial+sham_scale*sham_delta
    assert abs(float(((sham-initial)@q).norm()-response_norm)) < 1e-10
    source = json.loads((Path(plan['source_fit'])/'run.json').read_text())
    assert source['datasets'] == geometry['tasks']
    merged = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    joint = torch.load(plan['joint_checkpoint'], map_location='cpu', weights_only=True)
    original_affine = torch.cat([merged[W].double(), merged[B].double().unsqueeze(1)], dim=1)
    assert torch.equal(original_affine, initial)
    affines = {'response_fit': fit, 'response_shuffled': sham,
               'joint_film_oracle': torch.cat([joint[W].double(), joint[B].double().unsqueeze(1)], dim=1)}
    states = {'original': merged}
    paths = {'original': plan['initial_checkpoint']}
    for name, affine in affines.items():
        state = dict(merged)
        state.update({W: affine[:, :-1].float(), B: affine[:, -1].float()})
        assert all(torch.equal(state[k], merged[k]) for k in merged if k not in [W,B])
        assert torch.isfinite(state[W]).all() and torch.isfinite(state[B]).all()
        folder = args.outdir/name
        folder.mkdir()
        path = folder/'merged.ckpt'
        torch.save(state, path)
        states[name] = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(states[name][k], state[k]) for k in state)
        paths[name] = str(path)
    assert list(states) == plan['conditions']
    affine_export_errors = {}
    for name, affine in affines.items():
        actual = torch.nn.functional.linear(q[:-1].T.float(), states[name][W], states[name][B]).double().T
        affine_export_errors[name] = float((actual-affine@q).abs().max())
        assert affine_export_errors[name] < 1e-5
    construction = {'unique_condition_groups': geometry['instruction_groups'], 'correction_permutation': permutation.tolist(),
                    'shuffled_response_matching_scale': sham_scale,
                    'correction_response_norm_all16': float(response_norm),
                    'shuffled_correction_response_norm_all16': float(((sham-initial)@q).norm()),
                    'fit_displacement_norm': float((fit-initial).norm()),
                    'shuffled_displacement_norm': float((sham-initial).norm()),
                    'same_condition_irreducible_residual': geometry['joint_affine_fit']['duplicate_instruction_irreducible_residual_norm'],
                    'fp32_affine_export_max_errors': affine_export_errors}
    (args.outdir/'construction.json').write_text(json.dumps(construction, indent=2)+'\n')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    indices = np.load(Path(plan['source_fit'])/'calibration_indices.npz')
    reference = json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    data_reference = json.loads(Path(plan['data_reference']).read_text())['dataset_audits']
    evaluations = {name: {} for name in states}
    for task in source['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        i = source['datasets'].index(task)
        assert torch.equal(wrapper.instruction.cpu().double(), q[:-1, i])
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        assert array_hash(cx) == data_reference[task]['eeg_hash']
        assert len(vy) == data_reference[task]['validation_size']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), 128, False)
        logits = {}
        for name, state in states.items():
            leaf.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx,_ in loader])
            logits[name] = values
            evaluations[name][task] = {split: classification_metrics(values[ids], truth[ids])
                                      for split,ids in [('calibration', slice(0,n)), ('validation', slice(n,None))]}
        assert abs(evaluations['original'][task]['validation']['BAcc']-reference['folded_val'][task]['BAcc']) < 1e-12
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {c: round(100*(evaluations[c][task]['validation']['BAcc']-evaluations['original'][task]['validation']['BAcc']),4)
                     for c in states if c!='original'}, flush=True)
    contrasts = {}
    for split in ['calibration', 'validation']:
        contrasts[split] = {}
        for treatment, control in [(c,'original') for c in affines]+[('response_fit','response_shuffled')]:
            deltas = {t: 100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in source['datasets']}
            contrasts[split][treatment+'_vs_'+control] = {'macro_gain_pp': float(np.mean(list(deltas.values()))),
                                                        'per_task_gain_pp': deltas,
                                                        'improved': sum(v>0 for v in deltas.values()),
                                                        'declined': sum(v<0 for v in deltas.values())}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'construction': construction, 'evaluations': evaluations, 'contrasts': contrasts,
              'exported_paths': paths, 'checkpoint_sha256': {c:sha256(Path(p)) for c,p in paths.items()},
              'seconds': time.time()-start, 'test_data_read': False, 'test_inference_performed': False,
              'scope': 'All16 full exposed validation, seed0. Only existing FiLM affine parameters change. Response fitting uses original expert weights and fixed text, with the pre-existing labeled merger as initialization. Joint condition is oracle-only. No new architecture or runtime bank; generic affine fitting is prior art, novelty unestablished.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', {k:v['macro_gain_pp'] for k,v in contrasts['validation'].items()}, flush=True)


if __name__ == '__main__':
    main()
