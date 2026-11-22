"""Validation-only component swaps from plan section 8.2; no fitting or selection.

Evaluate B0/AM and BM/A0 using completed merge runs. The previously measured
B0/A0 and BM/AM controls are included with their source paths. Buffers and
inactive tensors in these merges were already checked against the same base.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, evaluate, load_leaf, make_loader,
    prepare_text, selection_key, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/aligned_merge_v1/seed0')
    parser.add_argument('--methods', default='block_ce,cpsc_block_ce')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'analysis/component_swaps.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    runs = {name: json.loads((args.root / name / 'run.json').read_text())
            for name in args.methods.split(',')}
    first = next(iter(runs.values()))
    names = first['datasets']
    if any(run['datasets'] != names for run in runs.values()):
        raise ValueError('Component controls require the same task manifest')
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    base_path = Path(first['base_checkpoint'])
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    leaf = load_leaf(config, base_path).to(device).eval().requires_grad_(False)
    loaders, readouts = {}, {}
    for ds in names:
        labels, _, instruction, prototypes = prepare_text(config, ds)
        _, x, y, vx, vy, tx, ty = load_dataset(ds)
        loaders[ds] = make_loader(vx, vy, config, len(labels), args.bs, False)
        readouts[ds] = instruction.to(device), prototypes.to(device)
        del x, y, vx, vy, tx, ty
    base_history = args.root / 'repair_base/history.json'
    base_measurement = json.loads(base_history.read_text())[0]
    if base_measurement['step'] != 0:
        raise ValueError('Base reference must be the unrepaired step-zero checkpoint')
    result = {
        'protocol': 'validation, training instruction, FP32, eval mode, no fitting or checkpoint selection',
        'base_checkpoint': str(base_path), 'base_sha256': sha256(base_path),
        'base_control_source': str(base_history), 'base_control': base_measurement['val'],
        'datasets': names, 'batch_size': args.bs, 'runs': {},
    }
    for name, run in runs.items():
        if run['base_checkpoint_sha256'] != result['base_sha256']:
            raise ValueError(f'{name}: different base checkpoint')
        path = args.root / name / 'merged.ckpt'
        merged = torch.load(path, map_location='cpu', weights_only=True)
        conditions = {'B0_AM': True, 'BM_A0': False}
        measured = {}
        for condition, use_merged_interface in conditions.items():
            state = {key: (merged[key] if key.startswith('qformer.') == use_merged_interface else tensor)
                     for key, tensor in base.items()}
            leaf.load_state_dict(state, strict=True)
            metrics = {ds: evaluate(PrototypeClassifier(leaf, *readouts[ds]).to(device), loaders[ds], device)
                       for ds in names}
            bacc, loss, _ = selection_key(metrics)
            measured[condition] = {'macro_BAcc': bacc, 'macro_Loss': loss, 'per_dataset': metrics}
            print(f'{name} {condition}: validation macro BAcc={bacc:.6f}', flush=True)
        b0a0 = base_measurement['val_macro_BAcc']
        bm_am = run['best_val_macro_BAcc']
        result['runs'][name] = {
            'checkpoint_sha256': sha256(path), 'merged_control_source': str(args.root / name / 'run.json'),
            'BM_AM': {'macro_BAcc': bm_am, 'per_dataset': run['best_val']},
            **measured,
            'interaction_pp': 100 * (bm_am - measured['BM_A0']['macro_BAcc'] - measured['B0_AM']['macro_BAcc'] + b0a0),
        }
    result.update(seconds=time.time() - started, code_sha256=sha256(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {output}. Non-additivity alone does not establish semantic drift.')


if __name__ == '__main__':
    main()
