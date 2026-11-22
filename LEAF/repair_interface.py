"""Matched CE repair on fixed calibration pools (plan §4.3 and bounded extensions).

Default: shared Q-Former only. Projection modes restrict the existing final
layer; full mode trains both backbone and interface. No task-specific parameters.
Eval mode fixes BN statistics and dropout for every mode.
The default budget is 400 total updates, validation every 40, AdamW 1e-4/1e-3.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from d_downstream_qformer_ce import (
    ROOT, BatchStream, PrototypeClassifier, classification_metrics, evaluate, load_leaf, make_loader,
    prepare_text, prototype_logits, selection_key, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from utils import seed_all
from projection_repair import ProjectionDelta, plain_state, projection_features, prototype_union_basis
from aligned_support import array_hash
from merge_poc import UNUSED_PREFIXES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--datasets', help='Default: calibration file order')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--eval-every', type=int, default=40)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--mode', choices=['interface', 'full', 'output', 'contrast', 'random'], default='interface')
    args = p.parse_args()
    started = time.time()
    indices = np.load(args.calibration)
    names = args.datasets.split(',') if args.datasets else indices.files
    if args.steps % len(names) or args.steps < 0 or min(args.eval_every, args.bs) <= 0:
        p.error('Use nonnegative whole dataset rounds and positive evaluation/batch sizes')
    args.outdir.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    projection_mode = args.mode in ('output', 'contrast', 'random')
    if projection_mode:
        torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, args.checkpoint).to(device).eval()
    leaf.requires_grad_(False)
    if args.mode == 'full':
        for name, parameter in leaf.named_parameters():
            parameter.requires_grad_(not name.startswith(UNUSED_PREFIXES))
    elif not projection_mode:
        leaf.qformer.requires_grad_(True)
    allowed_updates = {n for n, t in leaf.named_parameters() if t.requires_grad}
    if projection_mode:
        allowed_updates.update(['qformer.output_embed.2.weight', 'qformer.output_embed.2.bias'])
    fixed_state = {n: t.detach().cpu().clone() for n, t in leaf.state_dict().items()
                   if n not in allowed_updates}
    readouts, train, val, raw_probes, feature_info = {}, {}, {}, {}, {}
    cache_seconds = 0.0
    for ds in names:
        labels, text, instruction, prototypes = prepare_text(config, ds)
        readouts[ds] = instruction.to(device), prototypes.to(device)
        _, x, y, vx, vy, tx, ty = load_dataset(ds)
        del tx, ty
        index = indices[ds]
        if (index.min() < 0 or index.max() >= len(x) or len(np.unique(index)) != len(index)
                or len(index) > 512):
            raise ValueError(f'{ds}: invalid calibration indices')
        train[ds] = make_loader(x[index], y[index], config, len(labels), args.bs, True)
        val[ds] = make_loader(vx, vy, config, len(labels), args.bs, False)
        if projection_mode:
            cache_started = time.time()
            raw_probes[ds] = torch.from_numpy(np.asarray(vx[:8], dtype=np.float32)).clone()
            ordered = make_loader(x[index], y[index], config, len(labels), args.bs, False)
            features, targets = projection_features(leaf, ordered, readouts[ds][0], device)
            validation_features, validation_targets = projection_features(leaf, val[ds], readouts[ds][0], device)
            train[ds] = DataLoader(TensorDataset(features, targets), batch_size=args.bs, shuffle=True)
            val[ds] = DataLoader(TensorDataset(validation_features, validation_targets), batch_size=args.bs)
            feature_info[ds] = {'train_samples': len(features), 'validation_samples': len(validation_features),
                                'instruction_hash': array_hash(readouts[ds][0]),
                                'prototypes_hash': array_hash(readouts[ds][1]),
                                'storage': 'in-memory only; freshly generated, no disk reuse'}
            cache_seconds += time.time() - cache_started
            del ordered, features, targets, validation_features, validation_targets
        del x, y, vx, vy
    basis_info = None
    if projection_mode:
        semantic_basis, singular_values = prototype_union_basis([readouts[ds][1] for ds in names])
        basis = semantic_basis
        if args.mode == 'random':
            rng = np.random.default_rng(args.seed)
            q, _ = np.linalg.qr(rng.standard_normal(semantic_basis.shape))
            basis = torch.from_numpy(q).float()
        elif args.mode == 'output':
            basis = None
        leaf.qformer.output_embed[-1] = ProjectionDelta(leaf.qformer.output_embed[-1], basis)
        basis_info = {'semantic_rank': semantic_basis.shape[1],
                      'update_rank': config.text_emb_model_dim if basis is None else basis.shape[1],
                      'singular_values': singular_values,
                      'definition': 'union of within-task centered prototype spans; no between-task centroids'}
        if basis is not None:
            np.save(args.outdir / 'projection_basis.npy', basis.numpy())
    np.savez(args.outdir / 'calibration_indices.npz', **{ds: indices[ds] for ds in names})
    history, best = [], {'key': (-float('inf'), -float('inf'))}

    @torch.no_grad()
    def evaluate_projection(linear):
        metrics = {}
        for ds in names:
            logits, labels = [], []
            for features, y in val[ds]:
                z = F.normalize(linear(features.to(device)), dim=-1)
                logits.append(prototype_logits(z, readouts[ds][1]).cpu())
                labels.append(y)
            metrics[ds] = classification_metrics(torch.cat(logits), torch.cat(labels))
        return metrics

    def record(step):
        metrics = (evaluate_projection(leaf.qformer.output_embed[-1]) if projection_mode else
                   {ds: evaluate(PrototypeClassifier(leaf, *readouts[ds]).to(device), val[ds], device)
                    for ds in names})
        bacc, ce, key = selection_key(metrics)
        history.append({'step': step, 'elapsed_s': time.time() - started,
                        'val_macro_BAcc': bacc, 'val_macro_Loss': ce, 'val': metrics})
        if key > best['key']:
            best.update(key=key, step=step, bacc=bacc, loss=ce, val=metrics)
            torch.save(plain_state(leaf), args.outdir / 'merged.ckpt')
            if projection_mode:
                torch.save(leaf.qformer.output_embed[-1].state_dict(), args.outdir / 'projection_updates.pt')
        (args.outdir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'step {step} val macro BAcc={bacc:.6f} CE={ce:.6f}', flush=True)

    record(0)
    trainable = [t for t in leaf.parameters() if t.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    streams = {ds: BatchStream(train[ds]) for ds in names}
    order_rng = np.random.default_rng(args.seed)
    component_gradient_max = {'tower': 0.0, 'qformer': 0.0}
    step = 0
    for _ in range(args.steps // len(names)):
        for k in order_rng.permutation(len(names)):
            ds = names[k]
            x, y = streams[ds].next()
            x, y = x.to(device), y.to(device)
            instruction, prototypes = readouts[ds]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and not projection_mode):
                if projection_mode:
                    z = F.normalize(leaf.qformer.output_embed[-1](x), dim=-1)
                else:
                    if args.mode == 'full':
                        tokens = leaf.tower(x)
                    else:
                        with torch.no_grad():
                            tokens = leaf.tower(x)
                    z = F.normalize(leaf.qformer(tokens, instruction[None].expand(len(x), -1)), dim=-1)
                loss = F.cross_entropy(prototype_logits(z, prototypes), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [t.grad for t in trainable if t.grad is not None]
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f'Non-finite repair loss/gradient at step {step + 1}')
            grad_norm = sum(g.square().sum() for g in gradients).sqrt().item()
            component_gradients = {}
            for component in component_gradient_max:
                squares = [t.grad.square().sum() for n, t in leaf.named_parameters()
                           if n.startswith(component + '.') and t.grad is not None]
                component_gradients[component] = torch.stack(squares).sum().sqrt().item() if squares else 0.0
                component_gradient_max[component] = max(component_gradient_max[component], component_gradients[component])
            optimizer.step()
            step += 1
            with (args.outdir / 'dynamics.jsonl').open('a') as handle:
                handle.write(json.dumps({'step': step, 'dataset': ds, 'ce': loss.item(),
                                         'grad_norm': grad_norm,
                                         'component_grad_norm': component_gradients}) + '\n')
            if step % args.eval_every == 0 or step == args.steps:
                record(step)
    for n, t in plain_state(leaf).items():
        if n in fixed_state and not torch.equal(t.cpu(), fixed_state[n]):
            raise AssertionError(f'Repair changed frozen tensor {n}')
    if args.mode == 'full' and args.steps and any(value <= 0 for value in component_gradient_max.values()):
        raise AssertionError('Full-model repair requires nonzero backbone and interface gradients')
    check = load_leaf(config, args.outdir / 'merged.ckpt').to(device).eval()
    reloaded = (evaluate_projection(check.qformer.output_embed[-1]) if projection_mode else
                {ds: evaluate(PrototypeClassifier(check, *readouts[ds]).to(device), val[ds], device) for ds in names})
    if abs(selection_key(reloaded)[0] - best['bacc']) > 1e-10:
        raise AssertionError('Repair reload changed validation predictions')
    probe_errors = {}
    if projection_mode:
        with torch.no_grad():
            for ds in names:
                x = raw_probes[ds].to(device)
                direct, _ = check(x, readouts[ds][0][None].expand(len(x), -1))
                cached = F.normalize(check.qformer.output_embed[-1](val[ds].dataset.tensors[0][:len(x)].to(device)), dim=-1)
                error = (prototype_logits(direct, readouts[ds][1]) - prototype_logits(cached, readouts[ds][1])).abs().max().item()
                probe_errors[ds] = error
                if error > 1e-5:
                    raise AssertionError(f'{ds}: full-model vs cached-projection logit error {error}')
    metadata = {
        'method': f'{args.mode}_ce_repair', 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256(args.checkpoint), 'datasets': names,
        'seed': args.seed, 'steps': args.steps, 'eval_every': args.eval_every, 'batch_size': args.bs,
        'lr': args.lr, 'weight_decay': args.weight_decay, 'optimizer': 'AdamW',
        'training_mode': 'eval: dropout disabled, BN fixed',
        'precision': 'FP32 frozen features, projection, CE and validation' if projection_mode else 'bf16 forward, fp32 CE',
        'trainable_parameters': sum(t.numel() for t in trainable), 'basis': basis_info,
        'trainable_component_parameters': {
            component: sum(t.numel() for n, t in leaf.named_parameters()
                           if t.requires_grad and n.startswith(component + '.'))
            for component in ('tower', 'qformer')},
        'feature_cache': feature_info, 'cached_vs_full_logit_max_abs': probe_errors,
        'weight_decay_applies_to': ('zero-initialized update coordinates' if projection_mode else
                                   'active backbone and shared interface parameters' if args.mode == 'full' else 'shared interface parameters'),
        'calibration': str(args.calibration.resolve()), 'calibration_sha256': sha256(args.calibration),
        'frozen_backbone_unchanged': True if args.mode != 'full' else None,
        'frozen_tensors_unchanged': True, 'backbone_trainable': args.mode == 'full',
        'component_gradient_max': component_gradient_max, 'best_step': best['step'],
        'best_val_macro_BAcc': best['bacc'], 'best_val_macro_Loss': best['loss'], 'best_val': best['val'],
        'folded_val_macro_BAcc': selection_key(reloaded)[0],
        'seconds': {'total': time.time() - started, 'feature_cache_included_in_setup': cache_seconds},
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        'command': ' '.join(sys.argv), 'code_sha256': sha256(Path(__file__)),
        'projection_helper_sha256': sha256(ROOT / 'projection_repair.py') if projection_mode else None,
        'active_parameter_definition_sha256': sha256(ROOT / 'merge_poc.py') if args.mode == 'full' else None,
    }
    (args.outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {args.outdir}; best step={best["step"]}, BAcc={best["bacc"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
