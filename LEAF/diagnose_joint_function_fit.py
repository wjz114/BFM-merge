"""Fit a Joint teacher inside the original tensor span, with matched CE continuation.

Oracle diagnosis only: Joint-assisted checkpoints are never candidate methods.
Both branches share initialization, samples, minibatches, updates and optimizer.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from d_downstream_qformer_ce import (
    ROOT, BatchStream, PrototypeClassifier, classification_metrics, load_leaf,
    make_loader, prepare_text, prototype_logits, selection_key, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES


def centered(logits):
    return logits - logits.mean(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    plan = json.loads(args.plan.read_text())
    source_path = Path(plan['initial_run'])
    source = json.loads((source_path / 'run.json').read_text())
    assert source['granularity'] == 'tensor'
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    base = {name: value.detach().clone() for name, value in leaf.named_parameters()}
    buffers = dict(leaf.named_buffers())
    active = [name for name in base if not name.startswith(UNUSED_PREFIXES)]
    assert active == source['groups']
    fixed = [name for name in leaf.state_dict() if name not in active]
    bank = {name: [] for name in active}
    source_hashes = {'base': sha256(Path(source['base_checkpoint'])),
                     'initial_run': sha256(source_path / 'run.json'),
                     'initial_alpha': sha256(source_path / 'best_alpha.npy'),
                     'joint': sha256(Path(plan['joint_checkpoint']))}
    for task, path in source['experts'].items():
        state = torch.load(path, map_location=device, weights_only=True)
        assert all(torch.equal(state[name], leaf.state_dict()[name]) for name in fixed)
        for name in active:
            bank[name].append(state[name].float() - base[name])
        source_hashes[task] = sha256(Path(path))
    del state
    bank = {name: torch.stack(values) for name, values in bank.items()}
    alpha0 = torch.from_numpy(np.load(source_path / 'best_alpha.npy')).to(device)
    alpha = {mode: alpha0.clone().requires_grad_(True) for mode in plan['branches']}
    optimizers = {mode: torch.optim.Adam([value], lr=plan['lr'], weight_decay=0)
                  for mode, value in alpha.items()}

    def compose(value):
        params = dict(base)
        for index, name in enumerate(active):
            params[name] = base[name] + torch.tensordot(value[:, index], bank[name], dims=1)
        return params

    initial_state = torch.load(source_path / 'merged.ckpt', map_location=device, weights_only=True)
    with torch.no_grad():
        initial_error = max((value - initial_state[name]).abs().max().item()
                            for name, value in compose(alpha0).items())
    assert initial_error <= 1e-6, initial_error
    del initial_state
    teacher = load_leaf(config, Path(plan['joint_checkpoint'])).to(device).eval().requires_grad_(False)
    tasks = source['datasets'][:2] if args.smoke else source['datasets']
    indices = np.load(source_path / 'calibration_indices.npz')
    readouts, calibration, validation, weights, target_energy, teachers = {}, {}, {}, {}, {}, {}
    bs = 8 if args.smoke else plan['batch_size']
    for task in tasks:
        labels, _, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task]
        readouts[task] = instruction.to(device), prototypes.to(device)
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(task)
        cal = make_loader(train_x[indices[task]], train_y[indices[task]], config, len(labels), bs, False)
        val = make_loader(val_x, val_y, config, len(labels), bs, False)
        del train_x, train_y, val_x, val_y, test_x, test_y
        if args.smoke:
            # Engineering-only subset; metrics still have all label classes.
            for loader in [cal, val]:
                x, y = loader.dataset.tensors
                keep = torch.cat([torch.where(y == c)[0][:8] for c in range(len(labels))])
                loader.dataset.tensors = x[keep], y[keep]
        targets = {}
        with torch.no_grad():
            wrapper = PrototypeClassifier(teacher, *readouts[task]).to(device).eval()
            for split, loader in [('calibration', cal), ('validation', val)]:
                targets[split] = torch.cat([wrapper(x.to(device)).cpu() for x, _ in loader])
        teachers[task] = targets
        x, y = cal.dataset.tensors
        counts = torch.bincount(y, minlength=len(labels))
        weights[task] = (len(y) / (len(labels) * counts.float())).to(device)
        target_energy[task] = centered(targets['calibration']).square().mean().clamp_min(1e-12).to(device)
        generator = torch.Generator().manual_seed(plan['seed'] * 100 + tasks.index(task))
        calibration[task] = DataLoader(TensorDataset(x, y, targets['calibration']), batch_size=bs,
                                       shuffle=True, generator=generator)
        validation[task] = val
        print('Teacher cached:', task, 'cal', len(y), 'val', len(val.dataset), flush=True)
    del teacher

    def logits(params, task, x):
        instruction, prototypes = readouts[task]
        z, _ = functional_call(leaf, (params, buffers), (x, instruction[None].expand(len(x), -1)))
        return prototype_logits(z, prototypes)

    history, snapshots = [], {mode: [] for mode in alpha}

    @torch.no_grad()
    def measure(step, supplied=None, label=None):
        rows = {}
        for mode, value in (alpha if supplied is None else supplied).items():
            params = compose(value)
            split_results = {}
            for split, mapping in [('calibration', calibration), ('validation', validation)]:
                per_task = {}
                for task, loader in mapping.items():
                    # Sequential evaluation does not advance the training sampler.
                    x, y = loader.dataset.tensors[:2]
                    predictions = torch.cat([logits(params, task, x[i:i+bs].to(device)).cpu()
                                             for i in range(0, len(x), bs)])
                    target = teachers[task][split]
                    result = classification_metrics(predictions, y)
                    result['teacher_disagreement'] = float((predictions.argmax(-1) != target.argmax(-1)).float().mean())
                    result['normalized_centered_mse'] = float((centered(predictions) - centered(target)).square().mean() / target_energy[task].cpu())
                    per_task[task] = result
                bacc, ce, _ = selection_key(per_task)
                split_results[split] = {'macro_bacc': bacc, 'macro_ce': ce, 'per_task': per_task,
                    'macro_teacher_disagreement': float(np.mean([r['teacher_disagreement'] for r in per_task.values()])),
                    'macro_normalized_centered_mse': float(np.mean([r['normalized_centered_mse'] for r in per_task.values()]))}
            rows[mode] = split_results
        history.append({'step': step, 'label': label, 'results': rows, 'elapsed_s': time.time()-started})
        (args.outdir / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print('Evaluation:', step, label, {m:r['validation']['macro_bacc'] for m,r in rows.items()}, flush=True)

    measure(0)
    streams = {task: BatchStream(loader) for task, loader in calibration.items()}
    order = np.random.default_rng(plan['seed'])
    steps = 4 if args.smoke else plan['steps']
    interval = steps if args.smoke else plan['eval_every']
    assert steps % len(tasks) == 0
    step = 0
    for _ in range(steps // len(tasks)):
        for task_index in order.permutation(len(tasks)):
            task = tasks[task_index]
            x, y, target = [value.to(device) for value in streams[task].next()]
            dynamics = {'step': step+1, 'task': task, 'branches': {}}
            for mode, value in alpha.items():
                pred = logits(compose(value), task, x)
                fit = (centered(pred) - centered(target)).square().mean() / target_energy[task]
                ce = (F.cross_entropy(pred, y, reduction='none') * weights[task][y]).mean()
                loss = fit if mode == 'joint_function' else ce
                optimizers[mode].zero_grad(set_to_none=True)
                loss.backward()
                assert value.grad is not None and torch.isfinite(value.grad).all() and torch.isfinite(loss)
                dynamics['branches'][mode] = {'loss': loss.item(), 'balanced_ce': ce.item(),
                    'normalized_centered_mse': fit.item(), 'gradient_norm': value.grad.norm().item()}
                optimizers[mode].step()
                assert torch.isfinite(value).all()
            with (args.outdir / 'dynamics.jsonl').open('a') as handle:
                handle.write(json.dumps(dynamics)+'\n')
            step += 1
            if step % interval == 0:
                measure(step)
            if step in plan['averaging_steps'] or args.smoke and step == steps:
                for mode, value in alpha.items():
                    snapshots[mode].append(value.detach().cpu().numpy().copy())
    average = {mode: torch.from_numpy(np.stack(values).astype(np.float64).mean(0).astype(np.float32)).to(device)
               for mode, values in snapshots.items()}
    measure(steps, average, 'fixed_average')
    exports = {}
    with torch.no_grad():
        for suffix, values in [('final', alpha), ('average', average)]:
            for mode, value in values.items():
                name = f'{mode}_{suffix}'
                params = compose(value)
                state = {key: tensor.cpu().clone() for key, tensor in leaf.state_dict().items()}
                state.update({key: tensor.cpu().clone() for key, tensor in params.items()})
                assert all(torch.equal(state[key], leaf.state_dict()[key].cpu()) for key in fixed)
                torch.save(state, args.outdir / f'{name}.ckpt')
                np.save(args.outdir / f'{name}_alpha.npy', value.detach().cpu().numpy())
                check = load_leaf(config, args.outdir / f'{name}.ckpt').to(device).eval()
                errors, flips = {}, {}
                for task, loader in validation.items():
                    wrapper = PrototypeClassifier(check, *readouts[task]).to(device).eval()
                    error, changed = 0., 0
                    for x, _ in loader:
                        x = x.to(device)
                        before, after = logits(params, task, x), wrapper(x)
                        error = max(error, (before-after).abs().max().item())
                        changed += int((before.argmax(-1) != after.argmax(-1)).sum())
                    errors[task], flips[task] = error, changed
                assert max(errors.values()) <= 1e-5, errors
                exports[name] = {'sha256': sha256(args.outdir / f'{name}.ckpt'),
                                 'max_logit_error': errors, 'prediction_changes': flips}
                del check
    result = {'diagnostic_only': True, 'joint_assisted': True, 'test_inference_performed': False,
              'engineering_smoke': args.smoke, 'tasks': tasks, 'source_hashes': source_hashes,
              'initial_composition_error': initial_error, 'actual_updates_per_branch': step,
              'exports': exports, 'seconds': time.time()-started, 'plan': plan,
              'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'teacher_metrics': {task:{split:classification_metrics(target, (calibration if split=='calibration' else validation)[task].dataset.tensors[1])
                                        for split,target in splits.items()} for task,splits in teachers.items()}}
    (args.outdir / 'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
