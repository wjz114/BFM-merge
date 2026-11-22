"""Full six-MI validation of the fixed Cho context-distillation pilot."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure, divergence
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_cross_domain_transfer import TASKS
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
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    calibration = np.load(Path(plan['source'])/'calibration_indices.npz')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    active = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    base = {name: value.detach().clone() for name, value in leaf.state_dict().items()}
    states = {bank: torch.load(path, map_location=device, weights_only=True)
              for bank, path in plan['checkpoints'].items()}
    for state in states.values():
        assert all(torch.equal(state[name], base[name]) for name in base if name not in active)
    hashes = {bank: sha256(Path(path)) for bank, path in plan['checkpoints'].items()}
    own, records = {}, []
    for task in TASKS:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        _, tx, ty, vx, vy, sx, sy = load_dataset(task)
        x = np.concatenate([tx[calibration[task]], vx])
        y = np.concatenate([ty[calibration[task]], vy])
        n = len(calibration[task])
        del tx, ty, vx, vy, sx, sy
        truth = torch.from_numpy(y.astype('int64'))
        loader = make_loader(x, y, config, len(labels), 128, False)

        def predict(state):
            leaf.load_state_dict(state, strict=True)
            return torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])

        original = np.load(Path(plan['original_transfer'])/f'{task}_logits.npz')
        assert np.array_equal(original['truth'], y) and int(original['n_calibration']) == n
        target = torch.load(source['experts'][task], map_location=device, weights_only=True)
        assert all(torch.equal(target[name], base[name]) for name in base if name not in active)
        hashes['target/'+task] = sha256(Path(source['experts'][task]))
        baselines = {'base': predict(base), 'target_expert': predict(target)}
        for context, key in [('base', 'base'), ('target_expert', task)]:
            assert float((baselines[context]-torch.from_numpy(original[key])).abs().max()) <= 1e-5
        saved = {}
        if task == plan['expert']:
            for bank, state in states.items():
                logits = predict(state)
                saved[bank] = logits
                own[bank] = {}
                for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
                    row = classification_metrics(logits[ids], truth[ids])
                    reference = classification_metrics(baselines['target_expert'][ids], truth[ids])
                    row['change_vs_original_pp'] = 100*(row['BAcc']-reference['BAcc'])
                    row['kl_from_original'] = float(divergence(logits[ids], baselines['target_expert'][ids]).mean())
                    own[bank][split] = row
        else:
            for bank, state in states.items():
                for context, anchor in [('base', base), ('target_expert', target)]:
                    for scale in plan['scales']:
                        composed = {name: (anchor[name]+scale*(state[name]-base[name]) if name in active else base[name])
                                    for name in base}
                        logits = predict(composed)
                        key = f'{bank}__{context}__{scale}'
                        saved[key] = logits
                        row = {'target': task, 'bank': bank, 'context': context, 'scale': scale, 'splits': {}}
                        for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
                            metrics = classification_metrics(logits[ids], truth[ids])
                            baseline = classification_metrics(baselines[context][ids], truth[ids])
                            metrics['change_vs_context_pp'] = 100*(metrics['BAcc']-baseline['BAcc'])
                            metrics['kl_from_context'] = float(divergence(logits[ids], baselines[context][ids]).mean())
                            row['splits'][split] = metrics
                        records.append(row)
                        print(task, bank, context, scale, row['splits']['validation']['change_vs_context_pp'], flush=True)
        torch.save({'truth': truth, 'n_calibration': n, 'baselines': baselines, 'predictions': saved}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps({'own': own, 'records': records}, indent=2)+'\n')
    result = {'own': own, 'records': records, 'checkpoint_hashes': hashes,
              'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'seconds': time.time()-started, 'test_inference_performed': False,
              'scope': 'Exploratory Cho pilot, all six pre-existing MI domains, original calibration plus full exposed validation. No checkpoint or cell selection. Pairwise compositions do not establish final16-task merge performance.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
