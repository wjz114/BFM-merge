"""Fixed EEG adaptations of RegMean and RegMean++ on existing calibration IDs.

Both merge every attention/MLP linear map in all 28 transformer layers.
RegMean uses original expert inputs; RegMean++ uses the previously merged
layer's inputs. Remaining active parameters are averaged; buffers and unused
pretraining heads stay at the common base. No labels enter the solve.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, evaluate, load_leaf, make_loader, prepare_text, selection_key, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES, stratified_indices
from regmean_support import advance_layer, capture_layer_inputs, layer_statistics, merge_layer
from utils import seed_all


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['regmean', 'regmeanpp'], required=True)
    parser.add_argument('--base', type=Path, default=ROOT / 'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt')
    parser.add_argument('--experts-dir', type=Path, default=ROOT / 'runs/poc16/seed0/experts')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'runs/poc16/seed0/merge_block/calibration_indices.npz')
    parser.add_argument('--datasets', help='Default: every task in the calibration archive')
    parser.add_argument('--calibration-limit', type=int, default=512, help='Smokes only may shorten each fixed index array')
    parser.add_argument('--validation-limit', type=int, help='Engineering smokes only; full runs omit this')
    parser.add_argument('--alpha', type=float, default=.95, help='Official CLIP default off-diagonal factor; fixed before EEG evaluation')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha < 1 or not 1 <= args.calibration_limit <= 512 or args.bs < 1:
        parser.error('Require alpha in [0,1), calibration-limit in [1,512], and positive batch size')
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, args.base).eval().requires_grad_(False)
    base = {name: value.cpu().clone() for name, value in leaf.state_dict().items()}
    active = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    fixed = set(base) - active
    base_hash = sha256(args.base)
    indices = np.load(args.calibration)
    names = args.datasets.split(',') if args.datasets else indices.files
    states, sources = {}, {}
    for ds in names:
        path = args.experts_dir / ds / 'best.ckpt'
        meta = json.loads(path.with_name('run.json').read_text())
        if meta['datasets'] != [ds] or meta['source_checkpoint_sha256'] != base_hash:
            raise ValueError(f'{ds}: wrong task or common initialization')
        state = torch.load(path, map_location='cpu', weights_only=True)
        leaf.load_state_dict(state, strict=True)
        if any(not torch.equal(state[name], base[name]) for name in fixed):
            raise ValueError(f'{ds}: changed buffers or inactive parameters')
        states[ds] = state
        sources[ds] = {'checkpoint': str(path.resolve()), 'sha256': sha256(path)}
    uniform = {name: torch.stack([states[ds][name] for ds in names]).mean(0)
               if name in active else base[name] for name in base}
    leaf.load_state_dict(uniform, strict=True)
    leaf.to(device)
    train, validation, readouts, used_indices, data_info = {}, {}, {}, {}, {}
    for ds in names:
        labels, _, instruction, prototypes = prepare_text(config, ds)
        readouts[ds] = instruction.to(device), prototypes.to(device)
        _, x, y, vx, vy, tx, ty = load_dataset(ds)
        del tx, ty
        if args.validation_limit is not None:
            subset = stratified_indices(vy, args.validation_limit, np.random.default_rng(args.seed))
            vx, vy = vx[subset], vy[subset]
        original_indices = indices[ds]
        index = original_indices[stratified_indices(y[original_indices], args.calibration_limit,
                                                    np.random.default_rng(args.seed))]
        if index.min() < 0 or index.max() >= len(x) or len(np.unique(index)) != len(index):
            raise ValueError(f'{ds}: invalid calibration IDs')
        tensors = make_loader(x[index], y[index], config, len(labels), args.bs, False).dataset.tensors
        train[ds] = tensors[0]
        validation[ds] = make_loader(vx, vy, config, len(labels), 128, False)
        used_indices[ds] = index
        data_info[ds] = {'examples': len(index), 'eeg_sha256': array_hash(tensors[0]),
                         'instruction_sha256': array_hash(instruction), 'prototypes_sha256': array_hash(prototypes)}
        print(f'{ds}: calibration={len(index)}; validation={len(vy)}', flush=True)
        del x, y, vx, vy, tensors
    np.savez(args.outdir / 'calibration_indices.npz', **used_indices)
    setup_seconds = time.time() - started
    history = []

    def validate(step):
        metrics = {ds: evaluate(PrototypeClassifier(leaf, *readouts[ds]).to(device), validation[ds], device)
                   for ds in names}
        bacc, loss, _ = selection_key(metrics)
        history.append({'step': step, 'val_macro_BAcc': bacc, 'val_macro_Loss': loss, 'val': metrics})
        (args.outdir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'step {step}: validation BAcc={bacc:.8f}, CE={loss:.8f}', flush=True)
        return metrics, bacc, loss

    validate(0)
    solve_started = time.time()
    first_names = ['tower.mask_transformer.layers.0', 'tower.next_transformer.layers.0', 'qformer.layers.0']

    def initial_inputs(target_names, original_experts):
        inputs = {name: {} for name in target_names}
        probe = copy.deepcopy(leaf) if original_experts else leaf
        for ds in names:
            if original_experts:
                probe.load_state_dict(states[ds], strict=True)
            for name in target_names:
                inputs[name][ds] = []
            with capture_layer_inputs(probe, target_names) as captured:
                for start in range(0, len(train[ds]), args.bs):
                    captured.clear()
                    x = train[ds][start:start + args.bs].to(device)
                    probe(x, readouts[ds][0][None].expand(len(x), -1))
                    for name in target_names:
                        inputs[name][ds].append(captured[name])
        return inputs

    original = args.method == 'regmean'
    inputs = initial_inputs(first_names if original else first_names[:2], original)
    layer_audits = []
    for prefix in ['tower.mask_transformer', 'tower.next_transformer', 'qformer']:
        if prefix == 'qformer' and not original:
            inputs.update(initial_inputs([first_names[2]], False))
        batches = inputs.pop(prefix + '.layers.0')
        layers = dict(leaf.named_modules())[prefix].layers
        for index, layer in enumerate(layers):
            full_name = f'{prefix}.layers.{index}'
            expert_layer = copy.deepcopy(layer)
            layer_states, statistics, observations, next_batches = [], [], {}, {}
            for ds in names:
                state = {name[len(full_name) + 1:]: value for name, value in states[ds].items()
                         if name.startswith(full_name + '.')}
                expert_layer.load_state_dict(state, strict=True)
                grams, observations[ds] = layer_statistics(expert_layer, batches[ds], device)
                layer_states.append(state)
                statistics.append(grams)
                if original and index < len(layers) - 1:
                    next_batches[ds] = advance_layer(expert_layer, batches[ds], device)
            merged, solves = merge_layer(layer_states, statistics, args.alpha)
            layer.load_state_dict(merged, strict=True)
            observation_error = max(value['attention_reconstruction_max_abs'] for value in observations.values())
            if observation_error > 1e-4:
                raise RuntimeError(f'{full_name}: fused attention observer mismatch {observation_error}')
            if not original and index < len(layers) - 1:
                next_batches = {ds: advance_layer(layer, batches[ds], device) for ds in names}
            batches = next_batches
            audit = {'layer': full_name, 'elapsed_s': time.time() - solve_started,
                     'observations': observations, 'solves': solves}
            layer_audits.append(audit)
            with (args.outdir / 'layers.jsonl').open('a') as handle:
                handle.write(json.dumps(audit) + '\n')
            print(f'merged {full_name}: maps={len(solves)}, attention error={observation_error:.3g}', flush=True)
            del expert_layer, statistics, layer_states
    solve_seconds = time.time() - solve_started
    metrics, bacc, loss = validate(len(layer_audits))
    final_state = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
    if any(not torch.equal(final_state[name], base[name]) for name in fixed):
        raise RuntimeError('Merge changed a fixed buffer or unused parameter')
    checkpoint = args.outdir / 'merged.ckpt'
    torch.save(final_state, checkpoint)
    reloaded = load_leaf(config, checkpoint).to(device).eval().requires_grad_(False)
    export_errors = {}
    for ds in names:
        x = train[ds][:8].to(device)
        instruction = readouts[ds][0][None].expand(len(x), -1)
        export_errors[ds] = (leaf(x, instruction)[0] - reloaded(x, instruction)[0]).abs().max().item()
    if max(export_errors.values()) > 1e-6:
        raise RuntimeError(f'Checkpoint reload mismatch: {export_errors}')
    record = {
        'method': args.method + '_adapted', 'datasets': names, 'base_checkpoint_sha256': base_hash,
        'experts': sources, 'calibration': str(args.calibration.resolve()), 'data': data_info,
        'alpha': args.alpha, 'batch_size': args.bs, 'seed': args.seed,
        'validation_limit': args.validation_limit,
        'reference_commit': 'a9fdd97629707062a5872653dd9b34f7e25c7243',
        'definition': 'Every attention/MLP linear map in 28 layers; other active parameters averaged; buffers/inactive at base',
        'statistics': 'FP32 no-TF32 forwards and Gram products, FP64 accumulation and solves; token mean per task, equal task sum',
        'cross_layer_inputs': 'original expert' if original else 'previous merged layer',
        'fitting_labels_used': False, 'parameter_retraining': False, 'optimizer_updates': 0,
        'selection': 'One fixed final closed-form checkpoint; no validation checkpoint search',
        'best_step': len(layer_audits), 'best_val_macro_BAcc': bacc, 'best_val_macro_Loss': loss,
        'best_val': metrics, 'folded_val_macro_BAcc': bacc, 'folded_val_macro_Loss': loss,
        'export_embedding_max_abs': export_errors, 'fixed_state_verified': True,
        'code_sha256': {name: sha256(ROOT / name) for name in ['merge_regmean.py', 'regmean_support.py', 'LEAF.py', 'merge_poc.py']},
        'seconds': {'data_and_setup': setup_seconds, 'layerwise_merging': solve_seconds, 'total': time.time() - started},
        'command': ' '.join(sys.argv)}
    (args.outdir / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Complete: {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
