"""Read-only cross-domain transfer matrix for six shared-label MI experts.

This measures whether off-domain expert changes help or hurt relative to the
base. It does not establish the effect of suppressing those changes in a merge.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset


TASKS = ['MI_OpenBMI', 'MI_ShanghaiU', 'MI_HighGamma', 'MI_Cho2017',
         'MI_Shin2017A', 'MI_PhysioNet']


def transfer_changes(logits, baseline, truth):
    prediction, base_prediction = logits.argmax(1), baseline.argmax(1)
    good, base_good = prediction.eq(truth), base_prediction.eq(truth)
    return {
        'changed_predictions': int(prediction.ne(base_prediction).sum()),
        'base_wrong_expert_correct': int((~base_good & good).sum()),
        'base_correct_expert_wrong': int((base_good & ~good).sum()),
        'balanced_help_rate': float(torch.stack([
            (~base_good & good)[truth == c].float().mean()
            for c in truth.unique()]).mean()),
        'balanced_harm_rate': float(torch.stack([
            (base_good & ~good)[truth == c].float().mean()
            for c in truth.unique()]).mean()),
        'mean_total_variation_from_base': float(
            (logits.softmax(1)-baseline.softmax(1)).abs().sum(1).mean()/2),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--experts-dir', type=Path, help='Optional adapted bank; source still defines original calibration IDs')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    source = json.loads((args.source/'run.json').read_text())
    calibration = np.load(args.source/'calibration_indices.npz')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    paths = {'base': source['base_checkpoint'],
             **{task: str(args.experts_dir/task/'best.ckpt') if args.experts_dir else source['experts'][task] for task in TASKS},
             'prior_candidate': str(args.candidate/'merged.ckpt')}
    states = {name: torch.load(path, map_location='cpu', weights_only=True)
              for name, path in paths.items()}
    result = {
        'diagnostic_only': True, 'test_inference_performed': False,
        'tasks': TASKS, 'model_paths': paths,
        'adapted_expert_bank': str(args.experts_dir) if args.experts_dir else None,
        'checkpoint_hashes': {name: sha256(Path(path)) for name, path in paths.items()},
        'source_calibration_sha256': sha256(args.source/'calibration_indices.npz'),
        'code_sha256': sha256(Path(__file__)), 'domains': {},
        'interpretation': 'Cross-domain benefits and harms relative to base are descriptive. '
                          'They do not prove that RI or another adaptation destroys useful transfer, '
                          'or that retaining these changes improves a merged model.',
    }
    readouts = {}
    for task in TASKS:
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert len(labels) == 2
        readouts[task] = (instruction.cpu(), prototypes.cpu())
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(task)
        x = np.concatenate([train_x[calibration[task]], val_x])
        y = np.concatenate([train_y[calibration[task]], val_y])
        n_cal = len(calibration[task])
        del train_x, train_y, val_x, val_y, test_x, test_y
        loader = make_loader(x, y, config, len(labels), 128, False)
        truth = torch.from_numpy(y.astype(np.int64))
        logits = {}
        for name, state in states.items():
            leaf.load_state_dict(state, strict=True)
            logits[name] = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
        np.savez_compressed(args.outdir/f'{task}_logits.npz', truth=truth.numpy(),
                            calibration_indices=calibration[task], n_calibration=n_cal,
                            **{name: values.numpy() for name, values in logits.items()})
        domain = {'labels': labels, 'instruction': text,
                  'calibration_size': n_cal, 'validation_size': len(truth)-n_cal,
                  'splits': {}}
        for split, selection in [('calibration', slice(0, n_cal)),
                                 ('validation', slice(n_cal, None))]:
            rows = {}
            for name, values in logits.items():
                rows[name] = classification_metrics(values[selection], truth[selection])
                rows[name].update(transfer_changes(values[selection], logits['base'][selection],
                                                   truth[selection]))
            for name, row in rows.items():
                row['bacc_gain_vs_base'] = row['BAcc']-rows['base']['BAcc']
                assert abs(row['bacc_gain_vs_base']-
                           (row['balanced_help_rate']-row['balanced_harm_rate'])) < 1e-6
            domain['splits'][split] = rows
        result['domains'][task] = domain
        (args.outdir/'progress.json').write_text(json.dumps(result, indent=2)+'\n')
        print(task, {name: round(row['bacc_gain_vs_base']*100, 3)
                     for name, row in domain['splits']['validation'].items()}, flush=True)
    result['readout_checks'] = {}
    first_instruction, first_prototypes = readouts[TASKS[0]]
    for task, (instruction, prototypes) in readouts.items():
        result['readout_checks'][task] = {
            'instruction_exactly_equal_first': torch.equal(instruction, first_instruction),
            'prototype_min_permutation_max_abs': min(
                float((prototypes-first_prototypes).abs().max()),
                float((prototypes.flip(0)-first_prototypes).abs().max())),
        }
    result['seconds'] = time.time()-started
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
