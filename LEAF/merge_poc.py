"""Merge independently fine-tuned LEAF experts with task-vector coefficients.

For expert k, delta_k = theta_k - theta_0. For parameter group g,

    theta_M[g] = theta_0[g] + sum_k alpha[k, g] * delta_k[g]

Methods:
    uniform  fixed alpha_k = 1/K (no optimization)
    global   one learnable coefficient per expert
    block    one learnable coefficient per expert and structural block (33 groups)
    softmax_global  theta_M = sum_k softmax(beta)_k theta_k, one logit per expert
    softmax_block   theta_M[g] = sum_k softmax_k(beta[:, g]) theta_k[g], per block (33 groups)

The softmax methods average expert parameters directly with convex weights
(logits start at 0, i.e. alpha = 1/K); with weights summing to 1 this equals
task-vector merging with the same weights, so they isolate the convex constraint.

Learnable coefficients start at 1/K (so global/block start at uniform), are
unconstrained reals (no softmax, gate, mask or regularizer), and are optimized
with Adam on a fixed, stratified calibration subset (at most N per dataset) of
each source train split. theta_0 and every delta are constants; the merged model
is executed with torch.func.functional_call. Buffers always come from theta_0.
The best alpha by source validation macro-BAcc (ties: smaller macro CE) is
folded into an ordinary LEAF state dict (merged.ckpt).

Example, from LEAF/:
    python merge_poc.py --base checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt \
        --experts-dir runs/poc/experts --datasets MI_OpenBMI,MI_Cho2017 \
        --method block --outdir runs/poc/merge_block
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from d_downstream_qformer_ce import (
    METRIC_KEYS, ROOT, BatchStream, PrototypeClassifier, classification_metrics,
    evaluate, load_leaf, make_loader, parse_datasets, prepare_text,
    prototype_logits, resolve_file, selection_key, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from utils import seed_all


# Not used by the classification forward; kept at theta_0.
UNUSED_PREFIXES = ('tower.mask_token', 'tower.mask_outProj.', 'tower.next_outProj.')


def group_for_name(name):
    """Structural merge group of one LEAF parameter (see the POC plan, Section 5)."""
    if name.startswith('tower.tokenizer.') or name == 'tower.pos_embedding':
        return 'tower.input'
    for branch in ('mask', 'next'):
        if name.startswith((f'tower.{branch}_inProj.', f'tower.{branch}_transformer.norm.')):
            return f'tower.{branch}.outer'
        match = re.match(rf'tower\.{branch}_transformer\.layers\.(\d+)\.', name)
        if match:
            return f'tower.{branch}.layer{int(match.group(1)):02d}'
    if name in ('qformer.base_queries', 'qformer.pos_embedding') or name.startswith('qformer.film.'):
        return 'qformer.condition'
    match = re.match(r'qformer\.layers\.(\d+)\.', name)
    if match:
        return f'qformer.layer{int(match.group(1)):02d}'
    if name.startswith(('qformer.final_norm.', 'qformer.output_embed.')):
        return 'qformer.output'
    raise KeyError(f'No merge group for parameter {name}')


def stratified_indices(y, limit, rng):
    """At most `limit` indices, class proportions preserved (largest remainder)."""
    y = np.asarray(y)
    if len(y) <= limit:
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    quota = counts * limit / len(y)
    take = np.floor(quota).astype(int)
    for c in np.argsort(-(quota - take), kind='stable')[:limit - take.sum()]:
        take[c] += 1
    chosen = [rng.choice(np.flatnonzero(y == c), size=n, replace=False)
              for c, n in zip(classes, take)]
    return np.sort(np.concatenate(chosen))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--base', type=Path, required=True, help='theta_0: the checkpoint every expert started from')
    p.add_argument('--experts-dir', type=Path, required=True, help='Contains <dataset>/best.ckpt and run.json')
    p.add_argument('--datasets', required=True, help='Comma-separated expert datasets (merge order)')
    p.add_argument('--method', choices=['uniform', 'global', 'block', 'softmax_global', 'softmax_block'],
                   required=True)
    p.add_argument('--calibration-per-dataset', type=int, default=512)
    p.add_argument('--steps', type=int, default=500, help='Total alpha updates (ignored for uniform)')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], default='bf16-mixed')
    p.add_argument('--embedding-dir', type=Path, default=ROOT / 'text_embeddings')
    p.add_argument('--outdir', type=Path, required=True, help='New output directory')
    args = p.parse_args()
    args.datasets = parse_datasets(args.datasets, p)
    if len(args.datasets) < 2:
        p.error('Merging needs at least two experts')
    if min(args.calibration_per_dataset, args.bs, args.eval_every) <= 0 or args.steps < 0:
        p.error('--calibration-per-dataset, --bs and --eval-every must be positive; --steps nonnegative')
    if args.method != 'uniform' and args.steps % len(args.datasets):
        p.error('--steps must be a multiple of the number of datasets (equal-frequency rounds)')
    if args.gpu != '-1' and not args.gpu.isdigit():
        p.error('--gpu must be one nonnegative GPU index, or -1 for CPU')
    return args


def save_alpha_table(path_csv, path_png, alpha, groups, experts, title):
    lines = ['group,' + ','.join(experts)]
    lines += [f'{g},' + ','.join(f'{alpha[k, i]:.6f}' for k in range(len(experts)))
              for i, g in enumerate(groups)]
    path_csv.write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    table = alpha.T  # rows: groups, columns: experts
    center = 1.0 / len(experts)  # uniform initialization; color shows deviation from it
    span = max(float(np.abs(table - center).max()), 1e-6)
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
    norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
    height = max(2.0, 0.28 * len(groups) + 1.2)
    fig, ax = plt.subplots(figsize=(1.3 * len(experts) + 2.6, height))
    image = ax.imshow(table, cmap=cmap, norm=norm, aspect='auto')
    for i in range(table.shape[0]):
        for k in range(table.shape[1]):
            ax.text(k, i, f'{table[i, k]:.2f}', ha='center', va='center', fontsize=7, color='#222222')
    ax.set_xticks(range(len(experts)), experts, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(groups)), groups, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(f'merge coefficient (center = 1/K = {center:.2f})', fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def main():
    started = time.time()
    args = parse_args()
    names, K = args.datasets, len(args.datasets)
    base_path = resolve_file(args.base, ROOT / 'checkpoints')
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f'Output directory already exists: {outdir}; choose a new --outdir')
    config = build_model_config(load_yaml(resolve_file(args.config, ROOT)))
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    use_bf16 = device.type == 'cuda' and args.precision == 'bf16-mixed'
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    # ---- theta_0 and experts (strict loading, same initialization) ----
    leaf = load_leaf(config, base_path).to(device).eval()
    base_state = {k: v.detach().clone() for k, v in leaf.state_dict().items()}
    param_names = [n for n, _ in leaf.named_parameters()]
    buffer_names = [n for n, _ in leaf.named_buffers()]
    selected = [n for n in param_names if not n.startswith(UNUSED_PREFIXES)]
    unused = [n for n in param_names if n.startswith(UNUSED_PREFIXES)]
    group_names = list(dict.fromkeys(group_for_name(n) for n in selected))
    expected_groups = 3 + 2 * config.num_layers + 1 + config.num_qformer_layers + 1
    if len(group_names) != expected_groups:
        raise ValueError(f'Expected {expected_groups} merge groups, found {len(group_names)}: {group_names}')
    base_sha = sha256(base_path)
    softmax = args.method.startswith('softmax_')
    expert_paths, deltas = [], {n: [] for n in selected}  # softmax: expert params; else task vectors
    for ds in names:
        run_dir = (args.experts_dir / ds).resolve()
        run = json.loads((run_dir / 'run.json').read_text())
        if run['datasets'] != [ds] or run['source_checkpoint_sha256'] != base_sha:
            raise ValueError(f'{run_dir}: expected a {ds} expert trained from {base_path}')
        state = torch.load(run_dir / 'best.ckpt', map_location=device, weights_only=True)
        if set(state) != set(base_state) or any(state[k].shape != base_state[k].shape for k in state):
            raise ValueError(f'{run_dir}/best.ckpt does not match the LEAF state dict')
        for name in buffer_names + unused:
            # Fixed BN statistics and unused heads must equal theta_0 in every expert.
            if not torch.equal(state[name], base_state[name]):
                raise ValueError(f'{ds} expert changed {name}; experts must keep theta_0 buffers/unused params')
        for name in selected:
            deltas[name].append(state[name].float() if softmax else state[name].float() - base_state[name].float())
        expert_paths.append(str(run_dir / 'best.ckpt'))
        del state
    deltas = {n: torch.stack(v) for n, v in deltas.items()}  # (K, *shape), constants
    base_params = {n: base_state[n].float() for n in param_names}
    base_buffers = {n: base_state[n] for n in buffer_names}

    if args.method in ('block', 'softmax_block'):
        groups = group_names
        group_of = {n: groups.index(group_for_name(n)) for n in selected}
    else:
        groups = ['all']
        group_of = {n: 0 for n in selected}
    G = len(groups)
    learnable = args.method != 'uniform'
    if softmax:
        beta = torch.zeros((K, G), device=device, requires_grad=True)  # softmax(0) = 1/K
        coefficients = lambda: torch.softmax(beta, dim=0)  # normalized over experts, per group
        trainable = beta
    else:
        alpha = torch.full((K, G), 1.0 / K, device=device, requires_grad=learnable)
        coefficients = lambda: alpha
        trainable = alpha
    offset = (lambda n: base_params[n]) if softmax else (lambda n: 0.0)
    delta_norm = {ds: math.sqrt(sum(float((deltas[n][k] - offset(n)).pow(2).sum()) for n in selected))
                  for k, ds in enumerate(names)}
    print(f'{args.method}: K={K} experts, G={G} coefficient groups, '
          f'{K * G} coefficients ({"learnable" if learnable else "fixed"}); '
          f'{len(selected)} merged tensors, {len(unused)} unused tensors kept at theta_0', flush=True)
    print('task-vector L2 norms: ' + ', '.join(f'{ds}={v:.3f}' for ds, v in delta_norm.items()), flush=True)

    def compose(coefficients):
        # fp32 composition outside autocast; only the coefficients carry gradients.
        params = {n: base_params[n] for n in unused}
        for n in selected:
            mix = torch.tensordot(coefficients[:, group_of[n]], deltas[n], dims=1)
            params[n] = mix if softmax else base_params[n] + mix  # sum_k a_k theta_k | theta_0 + sum_k a_k d_k
        return params

    # ---- fixed text readouts and data ----
    readouts, texts = {}, {}
    for ds in names:
        labels, text, instruction, prototypes = prepare_text(config, ds, None, args.embedding_dir)
        readouts[ds] = (instruction.to(device), prototypes.to(device))
        texts[ds] = (labels, text)

    def merged_logits(params, ds, x):
        instruction, prototypes = readouts[ds]
        embedding, _ = functional_call(leaf, (params, base_buffers),
                                       (x, instruction.unsqueeze(0).expand(len(x), -1)))
        return prototype_logits(embedding, prototypes)

    cal_loaders, val_loaders, calibration = {}, {}, {}
    for k, ds in enumerate(names):
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(ds)
        del test_x, test_y  # never used for merging
        # Deterministic in (seed, dataset): identical across uniform/global/block.
        rng = np.random.default_rng([args.seed, k])
        index = stratified_indices(train_y, args.calibration_per_dataset, rng)
        calibration[ds] = index
        n_classes = len(texts[ds][0])
        cal_loaders[ds] = make_loader(train_x[index], train_y[index], config, n_classes, args.bs, True)
        val_loaders[ds] = make_loader(val_x, val_y, config, n_classes, args.bs, False)
        counts = np.bincount(train_y[index], minlength=n_classes).tolist()
        print(f'{ds}: calibration={len(index)} of train={len(train_y)} per-class={counts}; '
              f'val={len(val_y)}', flush=True)
        del train_x, train_y
    data_seconds = time.time() - started

    outdir.mkdir(parents=True, exist_ok=False)
    np.savez(outdir / 'calibration_indices.npz', **{ds: idx for ds, idx in calibration.items()})

    @torch.no_grad()
    def validate():
        params = compose(coefficients().detach())
        result = {}
        for ds in names:
            logits, labels = [], []
            for x, y in val_loaders[ds]:
                logits.append(merged_logits(params, ds, x.to(device)).cpu())
                labels.append(y)
            result[ds] = classification_metrics(torch.cat(logits), torch.cat(labels))
        return result

    history, alpha_history, logit_history = [], [], []
    best = {'key': (-math.inf, -math.inf)}

    def record(step, train_loss, elapsed):
        validation = validate()
        bacc, loss, key = selection_key(validation)
        history.append({'step': step, 'elapsed_s': elapsed, 'val_macro_BAcc': bacc,
                        'val_macro_Loss': loss, 'train_loss': train_loss, 'val': validation})
        alpha_history.append(coefficients().detach().cpu().numpy().copy())
        logit_history.append(trainable.detach().cpu().numpy().copy())
        if key > best['key']:
            best.update(key=key, step=step, alpha=alpha_history[-1].copy(), logits=logit_history[-1].copy(),
                        val=validation,
                        bacc=bacc, loss=loss)
        per_ds = ' '.join(f'{ds}={validation[ds]["BAcc"]:.4f}' for ds in names)
        print(f'step {step} val macro BAcc={bacc:.4f} loss={loss:.4f} | {per_ds}', flush=True)

    optimize_started = time.time()
    record(0, None, 0.0)  # step 0 = uniform initialization
    steps = args.steps if learnable else 0
    step_dataset = np.zeros(steps, dtype=np.int16)
    step_loss = np.zeros(steps, dtype=np.float32)
    if learnable:
        optimizer = torch.optim.Adam([trainable], lr=args.lr, weight_decay=0)
        streams = {ds: BatchStream(cal_loaders[ds]) for ds in names}
        order_rng = np.random.default_rng(args.seed)
        interval = {ds: [0.0, 0] for ds in names}
        leaf.eval()  # dropout off; gradients still flow to the coefficients
        step = 0
        for _ in range(steps // K):
            for index in order_rng.permutation(K):
                ds = names[index]
                x, y = streams[ds].next()
                x, y = x.to(device), y.to(device)
                params = compose(coefficients())
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    loss = F.cross_entropy(merged_logits(params, ds, x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite merge loss at step {step + 1} ({ds})')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                value = loss.item()
                step_dataset[step], step_loss[step] = index, value
                interval[ds][0] += value * len(y)
                interval[ds][1] += len(y)
                step += 1
                if step % args.eval_every == 0 or step == steps:
                    train_loss = {d: s / n for d, (s, n) in interval.items() if n}
                    interval = {d: [0.0, 0] for d in names}
                    record(step, train_loss, time.time() - optimize_started)
        if step != steps:
            raise RuntimeError(f'Expected {steps} alpha updates, ran {step}')
    optimize_seconds = time.time() - optimize_started

    # ---- fold the best alpha into an ordinary LEAF state dict ----
    best_alpha = torch.as_tensor(best['alpha'], device=device)
    with torch.no_grad():
        folded = compose(best_alpha)
    merged_state = {k: v.detach().clone() for k, v in base_state.items()}
    for n in selected:
        merged_state[n] = folded[n].to(base_state[n].dtype)
    merged_state = {k: v.cpu() for k, v in merged_state.items()}
    torch.save(merged_state, outdir / 'merged.ckpt.tmp')
    (outdir / 'merged.ckpt.tmp').replace(outdir / 'merged.ckpt')

    # The deployable checkpoint must reproduce the selected validation result.
    check = load_leaf(config, outdir / 'merged.ckpt').to(device)
    folded_val = {}
    for ds in names:
        instruction, prototypes = readouts[ds]
        folded_val[ds] = evaluate(PrototypeClassifier(check, instruction, prototypes).to(device),
                                  val_loaders[ds], device)
    folded_bacc, folded_loss, _ = selection_key(folded_val)
    print(f'folded merged.ckpt val macro BAcc={folded_bacc:.4f} loss={folded_loss:.4f} '
          f'(selected step {best["step"]}: {best["bacc"]:.4f} / {best["loss"]:.4f})', flush=True)

    alpha_np = best['alpha']
    np.save(outdir / 'best_alpha.npy', alpha_np)
    if softmax:
        np.save(outdir / 'best_logits.npy', best['logits'])
    np.savez(outdir / 'metrics.npz', step_dataset=step_dataset, step_loss=step_loss,
             val_step=np.array([h['step'] for h in history]),
             val_macro_BAcc=np.array([h['val_macro_BAcc'] for h in history]),
             val_macro_Loss=np.array([h['val_macro_Loss'] for h in history]),
             alpha_history=np.stack(alpha_history), logit_history=np.stack(logit_history),
             **{f'val{k}__{ds}': np.array([h['val'][ds][k] for h in history])
                for ds in names for k in METRIC_KEYS},
             bestStep=best['step'])
    (outdir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    save_alpha_table(outdir / 'alpha.csv', outdir / 'alpha_heatmap.png', alpha_np, groups, names,
                     f'{args.method} merge: best alpha (step {best["step"]}), groups x experts')
    metadata = {
        'method': args.method, 'base_checkpoint': str(base_path), 'base_checkpoint_sha256': base_sha,
        'experts': dict(zip(names, expert_paths)), 'datasets': names,
        'labels': {ds: texts[ds][0] for ds in names}, 'instruction': {ds: texts[ds][1] for ds in names},
        'K': K, 'groups': groups, 'num_coefficients': K * G,
        'params_per_group': {g: sum(base_params[n].numel() for n in selected if group_of[n] == i)
                             for i, g in enumerate(groups)},
        'merged_tensors': len(selected), 'unused_tensors_kept_at_base': unused,
        'task_vector_l2': delta_norm,
        'alpha_init': 1.0 / K,
        'alpha_parametrization': ('softmax over experts per group of logits initialized to 0' if softmax
                                  else 'unconstrained real coefficients'),
        'composition': ('theta_M = sum_k alpha_k theta_k (direct weighted average)' if softmax
                        else 'theta_M = theta_0 + sum_k alpha_k (theta_k - theta_0)'),
        'best_logits': best['logits'].tolist() if softmax else None,
        'optimizer': 'Adam' if learnable else None, 'lr': args.lr if learnable else None,
        'weight_decay': 0.0, 'steps_requested': args.steps, 'alpha_updates_actual': steps,
        'eval_every': args.eval_every, 'batch_size': args.bs, 'seed': args.seed,
        'precision': 'bf16-mixed' if use_bf16 else '32-true',
        'calibration_per_dataset': args.calibration_per_dataset,
        'calibration_sizes': {ds: int(len(i)) for ds, i in calibration.items()},
        'calibration_source': 'stratified subset of each load_dataset() train split; indices in calibration_indices.npz',
        'selection': 'maximum source validation macro BAcc; ties resolved by minimum macro validation loss',
        'best_step': best['step'], 'best_alpha': alpha_np.tolist(),
        'best_val_macro_BAcc': best['bacc'], 'best_val_macro_Loss': best['loss'], 'best_val': best['val'],
        'folded_val_macro_BAcc': folded_bacc, 'folded_val_macro_Loss': folded_loss, 'folded_val': folded_val,
        'seconds': {'data_and_setup': data_seconds, 'optimization_incl_validation': optimize_seconds,
                    'total': time.time() - started},
        'command': ' '.join(sys.argv),
    }
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved merged.ckpt, best alpha, calibration indices and curves to {outdir}')


if __name__ == '__main__':
    main()
