"""Full native evaluation of fixed prototype-contrast and feature ridge heads."""
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
    fit = json.loads(Path(plan['fit_run']).read_text())
    cache = json.loads(Path(plan['feature_cache']).read_text())
    initial_run = json.loads(Path(plan['initial_run']).read_text())
    assert not fit['validation_used_in_fit'] and fit['trunk_tensors_unchanged']
    tasks = fit['datasets']
    assert tasks == cache['datasets'] == initial_run['datasets']
    paths = {'initial': plan['initial_checkpoint'], **{k: v['path'] for k, v in fit['outputs'].items()}}
    states = {c: torch.load(p, map_location='cpu', weights_only=True) for c, p in paths.items()}
    head = ['qformer.output_embed.2.weight', 'qformer.output_embed.2.bias']
    for condition, state in states.items():
        assert all(torch.equal(state[k], states['initial'][k]) for k in state if k not in head)
        assert all(torch.isfinite(state[k]).all() for k in head)
        if condition != 'initial':
            assert sha256(Path(paths[condition])) == fit['outputs'][condition]['sha256']
            assert fit['outputs'][condition]['solver']['actual_relative_residual'] <= 1.01e-8
    indices = np.load(plan['calibration'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(paths['initial'])).to(device).eval().requires_grad_(False)
    evaluations, audits = {c: {} for c in paths}, {}
    for task in tasks:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        hashes = {k: array_hash(v) for k, v in [('calibration_eeg', cx), ('calibration_labels', cy),
                                               ('validation_eeg', vx), ('validation_labels', vy)]}
        assert hashes == cache['records'][task]['data_hashes']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        audits[task] = {**hashes, 'calibration_size': n, 'validation_size': len(vy)}
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        logits = {}
        for condition, state in states.items():
            leaf.load_state_dict(state, strict=True)
            logits[condition] = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            evaluations[condition][task] = {
                split: classification_metrics(logits[condition][ids], truth[ids])
                for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
        assert evaluations['initial'][task]['validation']['BAcc'] == initial_run['folded_val'][task]['BAcc']
        assert all(evaluations['initial'][task][s]['BAcc'] == cache['records'][task]['initial_metrics'][s]['BAcc']
                   for s in ['calibration', 'validation'])
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {c: 100*(evaluations[c][task]['validation']['BAcc']-evaluations['initial'][task]['validation']['BAcc'])
                     for c in paths if c != 'initial'}, flush=True)
    summaries, contrasts = {}, {}
    for split in ['calibration', 'validation']:
        summaries[split] = {c: float(np.mean([evaluations[c][t][split]['BAcc'] for t in tasks])) for c in paths}
        contrasts[split] = {}
        pairs = [(c, 'initial') for c in paths if c != 'initial']
        pairs += [('prototype_contrast', 'full_feature_ridge'), ('prototype_contrast', 'rotated_contrast_control')]
        for treatment, control in pairs:
            gains = {t: 100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in tasks}
            contrasts[split][treatment+'_vs_'+control] = {
                'macro_gain_pp': float(np.mean(list(gains.values()))), 'per_task_gain_pp': gains,
                'improved': sum(v > 0 for v in gains.values()), 'declined': sum(v < 0 for v in gains.values()),
                'leave_one_task_out_pp': {t: float(np.mean([v for k, v in gains.items() if k != t])) for t in tasks}}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': tasks, 'evaluations': evaluations, 'summaries': summaries, 'contrasts': contrasts,
              'dataset_audits': audits, 'initial_bacc_replays_exact': True, 'trunk_tensors_unchanged': True,
              'exported_paths': paths, 'checkpoint_sha256': {c: sha256(Path(p)) for c, p in paths.items()},
              'seconds': time.time()-started, 'test_data_read': False, 'test_inference_performed': False,
              'scope': 'All16 exposed full validation with native FP32 forwards from ordinary checkpoints for every condition. Only shared final affine head changes. One fixed ridge and prescribed targets; no selected checkpoint or task-specific runtime parameters. Single-bank hypothesis test; no novelty or independent-confirmation claim.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', summaries['validation'], flush=True)


if __name__ == '__main__':
    main()
