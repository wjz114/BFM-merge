"""Inspect a selected merge's validation prediction changes after export.

This read-only check reconstructs the frozen coefficients and compares the
functional forward, plain reloaded forward, and existing validation wrapper.
It never fits parameters or reads test scores.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, prototype_logits, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES, group_for_name


def compare(before, after, labels):
    error = (after - before).abs().amax(dim=1)
    top = before.topk(2, dim=1).values
    margin = top[:, 0] - top[:, 1]
    old, new = before.argmax(1), after.argmax(1)
    changed = (old != new).nonzero().flatten().tolist()
    return {
        'max_abs_logit_error': error.max().item(),
        'changed_predictions': len(changed),
        'changed_rows': [
            {'validation_row': i, 'label': labels[i].item(),
             'before': old[i].item(), 'after': new[i].item(),
             'before_margin': margin[i].item(), 'max_abs_logit_error': error[i].item(),
             'margin_within_twice_error': bool(margin[i] <= 2 * error[i])}
            for i in changed],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    run = json.loads((args.run / 'run.json').read_text())
    assert run['method'] == 'block' and run['task_vector_preprocessing'] is None
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(run['base_checkpoint'])).to(device).eval()
    base = {name: value.detach() for name, value in leaf.named_parameters()}
    buffers = dict(leaf.named_buffers())
    selected = [name for name in base if not name.startswith(UNUSED_PREFIXES)]
    groups = run['groups']
    group_ids = {name: groups.index(name if run['granularity'] == 'tensor' else group_for_name(name))
                 for name in selected}
    deltas = {name: [] for name in selected}
    for dataset in run['datasets']:
        state = torch.load(run['experts'][dataset], map_location=device, weights_only=True)
        for name in selected:
            deltas[name].append(state[name].float() - base[name].float())
    del state
    alpha = torch.tensor(run['best_alpha'], dtype=torch.float32, device=device)
    params = dict(base)
    for name in selected:
        params[name] = base[name] + torch.tensordot(alpha[:, group_ids[name]],
                                                  torch.stack(deltas[name]), dims=1)
    del deltas
    reloaded = load_leaf(config, args.run / 'merged.ckpt').to(device).eval()
    state_error = max((params[name] - value).abs().max().item()
                      for name, value in reloaded.named_parameters())
    assert state_error <= 1e-6, state_error
    result = {}
    for dataset in args.datasets.split(','):
        labels, text, instruction, prototypes = prepare_text(config, dataset)
        assert labels == run['labels'][dataset] and text == run['instruction'][dataset]
        instruction, prototypes = instruction.to(device), prototypes.to(device)
        wrapper = PrototypeClassifier(reloaded, instruction, prototypes).to(device).eval()
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(dataset)
        del train_x, train_y, test_x, test_y
        loader = make_loader(val_x, val_y, config, len(labels), run['batch_size'], False)
        outputs = {'functional': [], 'reloaded_same_text': [], 'reloaded_wrapper': []}
        ys = []
        for x, y in loader:
            x = x.to(device)
            condition = instruction[None].expand(len(x), -1)
            z, _ = functional_call(leaf, (params, buffers), (x, condition))
            outputs['functional'].append(prototype_logits(z, prototypes).cpu())
            z, _ = reloaded(x, condition)
            outputs['reloaded_same_text'].append(prototype_logits(z, prototypes).cpu())
            outputs['reloaded_wrapper'].append(wrapper(x).cpu())
            ys.append(y)
        outputs = {name: torch.cat(values) for name, values in outputs.items()}
        y = torch.cat(ys)
        metrics = {name: classification_metrics(values, y) for name, values in outputs.items()}
        result[dataset] = {
            'samples': len(y), 'metrics': metrics,
            'reproduces_selected_bacc': metrics['functional']['BAcc'] == run['best_val'][dataset]['BAcc'],
            'reproduces_folded_bacc': metrics['reloaded_wrapper']['BAcc'] == run['folded_val'][dataset]['BAcc'],
            'wrapper_instruction_renormalization_max_abs': (wrapper.instruction - instruction).abs().max().item(),
            'wrapper_prototype_renormalization_max_abs': (wrapper.prototypes - prototypes).abs().max().item(),
            'export_same_text': compare(outputs['functional'], outputs['reloaded_same_text'], y),
            'validation_wrapper': compare(outputs['functional'], outputs['reloaded_wrapper'], y),
        }
        print(dataset, json.dumps(result[dataset]), flush=True)
    report = {
        'run': str(args.run.resolve()), 'run_sha256': sha256(args.run / 'run.json'),
        'checkpoint_sha256': sha256(args.run / 'merged.ckpt'),
        'code_sha256': sha256(Path(__file__)), 'state_reconstruction_max_abs': state_error,
        'precision': 'FP32 forward/readout/composition, autocast and TF32 disabled',
        'test_scores_used': False, 'datasets': result, 'seconds': time.time() - start,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    assert all(v['reproduces_selected_bacc'] and v['reproduces_folded_bacc']
               and v['export_same_text']['max_abs_logit_error'] <= 1e-5
               and v['validation_wrapper']['max_abs_logit_error'] <= 1e-5
               and all(row['margin_within_twice_error'] for row in v['validation_wrapper']['changed_rows'])
               for v in result.values()), report


if __name__ == '__main__':
    main()
