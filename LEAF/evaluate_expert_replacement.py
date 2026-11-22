"""Replace one expert update in a fixed trained tensor merger; no refitting."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha256(Path(__file__)) == plan['code_sha256']
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    fit_dir, average_dir = Path(plan['source']), Path(plan['average'])
    fit = json.loads((fit_dir/'run.json').read_text())
    average = json.loads((average_dir/'run.json').read_text())
    assert sha256(average_dir/'run.json') == plan['average_run_sha256']
    assert fit['granularity'] == 'tensor' and fit['precision'] == '32-true'
    alpha = np.load(average_dir/'best_alpha.npy')
    index = fit['datasets'].index(plan['expert'])
    assert alpha.shape == (16, len(fit['groups']))
    calibration = np.load(fit_dir/'calibration_indices.npz')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, average_dir/'merged.ckpt').to(device).eval()
    active = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    assert active == set(fit['groups'])
    merged = torch.load(average_dir/'merged.ckpt', map_location='cpu', weights_only=True)
    original_expert = torch.load(fit['experts'][plan['expert']], map_location='cpu', weights_only=True)
    paths = {'original': average_dir/'merged.ckpt'}
    for condition, checkpoint in plan['replacements'].items():
        assert sha256(Path(checkpoint)) == plan['replacement_hashes'][condition]
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        replacement = {name: value.clone() for name, value in merged.items()}
        for column, name in enumerate(fit['groups']):
            replacement[name] += float(alpha[index, column])*(state[name]-original_expert[name])
        assert all(torch.equal(replacement[name], merged[name]) for name in merged if name not in active)
        assert all(torch.isfinite(replacement[name]).all() for name in active)
        folder = args.outdir/condition
        folder.mkdir()
        paths[condition] = folder/'merged.ckpt'
        torch.save(replacement, paths[condition])
    hashes = {name: sha256(path) for name, path in paths.items()}
    evaluations = {name: {} for name in paths}
    for task in fit['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        _, tx, ty, vx, vy, sx, sy = load_dataset(task)
        x = np.concatenate([tx[calibration[task]], vx])
        y = np.concatenate([ty[calibration[task]], vy])
        n = len(calibration[task])
        del tx, ty, vx, vy, sx, sy
        truth = torch.from_numpy(y.astype('int64'))
        loader = make_loader(x, y, config, len(labels), 128, False)
        logits = {}
        for condition, path in paths.items():
            leaf.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            logits[condition] = values
            evaluations[condition][task] = {split: classification_metrics(values[ids], truth[ids])
                                           for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
        assert abs(evaluations['original'][task]['validation']['BAcc']-average['folded_val'][task]['BAcc']) < 1e-12
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {name: round(100*(rows[task]['validation']['BAcc']-evaluations['original'][task]['validation']['BAcc']), 4)
                     for name, rows in evaluations.items()}, flush=True)
    contrasts = {}
    for split in ['calibration', 'validation']:
        contrasts[split] = {}
        for treatment, control in [('ri', 'original'), ('composed', 'original'), ('direct', 'original'), ('composed', 'direct')]:
            values = {task: 100*(evaluations[treatment][task][split]['BAcc']-evaluations[control][task][split]['BAcc'])
                      for task in fit['datasets']}
            contrasts[split][treatment+'_vs_'+control] = {'macro_change_pp': float(np.mean(list(values.values()))),
                                                        'per_task_change_pp': values,
                                                        'improved': sum(v>0 for v in values.values()),
                                                        'declined': sum(v<0 for v in values.values())}
    result = {'evaluations': evaluations, 'contrasts': contrasts, 'checkpoint_hashes': hashes,
              'exported_paths': {k: str(v) for k, v in paths.items()}, 'seconds': time.time()-started,
              'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'test_inference_performed': False,
              'interpretation': 'One expert is replaced with all original coefficients fixed. This measures a local intervention in the actual16-expert merger, not re-optimized coefficients or adaptation of the full bank. Full development-exposed source validation.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
