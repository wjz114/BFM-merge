"""Fine-tune the complete LEAF model using frozen text-prototype classification.

No classifier is added. Every model weight, including
the Q-Former, must load strictly from an instruction-tuned LEAF checkpoint.
Logits are cosine similarities (equivalently negative cosine distances,
up to a class-independent constant).

Single-dataset (expert) or multi-dataset (joint) training. ``--ds`` accepts one
name or a comma-separated list. With ``--steps-per-dataset R`` training runs R
rounds; each round draws one batch from every dataset in a freshly shuffled
order, and every batch is one optimizer step (R x D steps in total). Only the
fixed instruction and label prototypes change between datasets; the trainable
LEAF parameters are shared. Without ``--steps-per-dataset`` a single dataset is
trained for ``--epochs`` epochs, as in the original script.

Example, from LEAF/:
    python d_downstream_qformer_ce.py --ds MI_OpenBMI,MI_Cho2017 \
        --ckpt leaf-v1.0-instruct-mpnet-base.ckpt --steps-per-dataset 1000 --gpu 0

The best checkpoint is selected by dataset-macro validation balanced accuracy,
with macro validation loss breaking ties. Training only touches train and
validation splits; test evaluation lives in eval_poc.py.

SyncBatchNorm running statistics stay fixed at the initialization checkpoint's
values (the BN modules are kept in eval mode while the rest of the model
trains); their affine parameters still train.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing
and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score,
)

from LEAF import LEAF
from load_config import (
    build_model_config, build_train_config, load_instructions, load_targets,
    load_yaml,
)
from load_datasets import load_dataset
from utils import cosine_annealing_lr, make_lr_lambda, seed_all


ROOT = Path(__file__).resolve().parent
METRIC_KEYS = ['Loss', 'BAcc', 'Auroc', 'wF1', 'Kappa']


def resolve_file(path, fallback_root):
    for candidate in (path, fallback_root / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'File not found: {path} (also checked {fallback_root / path})')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized(vector, name):
    vector = torch.as_tensor(vector, dtype=torch.float32).detach().clone()
    norms = vector.norm(dim=-1, keepdim=True)
    if not torch.isfinite(vector).all() or (norms <= 1e-12).any():
        raise ValueError(f'{name} must contain finite, nonzero embeddings')
    return vector / norms


def prepare_text(config, dataset, instruction_text=None, embedding_dir=None, labels=None):
    """Keep prototype row i aligned with numeric dataset label i; never sort labels.

    ``labels`` is only for datasets outside configs/tasks.yaml (external
    evaluation); registered datasets always use their configured class order.
    """
    labels = load_targets()[dataset] if labels is None else list(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f'{dataset}: expected at least two distinct target labels')
    directory = Path(embedding_dir) if embedding_dir is not None else ROOT / 'text_embeddings'
    name = config.text_emb_model_name
    texts = (directory / f'{name}.txt').read_text().splitlines()
    vectors = np.load(directory / f'{name}.npy', allow_pickle=False)
    if vectors.shape != (len(texts), config.text_emb_model_dim) or len(set(texts)) != len(texts):
        raise ValueError(f'Invalid text embedding cache for {name}')
    cache = dict(zip(texts, vectors))
    instructions = load_instructions()
    if instruction_text is None:
        family = next((family for prefix, family in (
            ('MI_', 'Motor'), ('EMO_', 'Emotion'), ('CS_', 'Speech'),
            ('SSVEP_', 'SSVEP'),
        ) if dataset.startswith(prefix)), dataset)
        instruction_text = instructions[family][0]
    missing = [text for text in [*labels, instruction_text] if text not in cache]
    if missing:
        raise ValueError(f'Missing cached text embeddings: {missing}. Run init_text_embeddings.py {name}.')
    prototypes = normalized(np.stack([cache[label] for label in labels]), 'prototypes')
    instruction = normalized(cache[instruction_text], 'instruction')
    # Condition on the task and candidate classes, never the trial's true label.
    target_mean = normalized(prototypes.mean(dim=0), 'mean prototype')
    instruction = normalized(instruction + target_mean, 'blended instruction')
    return labels, instruction_text, instruction, prototypes


def prototype_logits(embedding, prototypes):
    """Cosine logits; explicit fp32 normalization keeps the readout stable under bf16."""
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        return F.normalize(embedding.float(), dim=-1) @ prototypes.T


class PrototypeClassifier(nn.Module):
    """Parameter-free cosine readout around an existing, fully restored LEAF."""

    def __init__(self, leaf, instruction, prototypes):
        super().__init__()
        dim = leaf.config.text_emb_model_dim
        if instruction.shape != (dim,) or prototypes.ndim != 2 or prototypes.shape[1] != dim:
            raise ValueError('Instruction/prototype dimensions do not match the LEAF configuration')
        self.leaf = leaf
        self.register_buffer('instruction', normalized(instruction, 'instruction'))
        self.register_buffer('prototypes', normalized(prototypes, 'prototypes'))

    def forward(self, x):
        embedding, _ = self.leaf(x, self.instruction.unsqueeze(0).expand(x.size(0), -1))
        return prototype_logits(embedding, self.prototypes)


def load_leaf(config, checkpoint):
    leaf = LEAF(config)
    # A Tower-only or partially matching checkpoint must fail, rather than leave
    # any randomly initialized Q-Former weights in the finetuning model.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    leaf.load_state_dict(state, strict=True)
    return leaf


def load_classifier(config, checkpoint, instruction, prototypes):
    return PrototypeClassifier(load_leaf(config, checkpoint), instruction, prototypes)


def train_mode_fixed_bn(leaf):
    """Normal training mode (dropout on) with BN running statistics frozen."""
    leaf.train()
    for module in leaf.modules():
        if isinstance(module, nn.SyncBatchNorm):
            module.eval()


def make_loader(x, y, config, n_classes, batch_size, training):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if x.ndim != 3 or x.shape[1] != config.n_channels or len(x) == 0:
        raise ValueError(f'Expected nonempty (N, {config.n_channels}, T) EEG, got {x.shape}')
    if not config.window_len <= x.shape[2] <= config.max_seq_len:
        raise ValueError(f'EEG length {x.shape[2]} is outside the configured input limits')
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Labels must be a one-dimensional integer array matching the EEG trials')
    if not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError('Every split must contain labels 0..K-1 in configs/tasks.yaml order')
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=False)


def classification_metrics(logits, labels):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite evaluation logits')
    probability = logits.softmax(dim=1).numpy()
    truth, prediction = labels.numpy(), logits.argmax(dim=1).numpy()
    auc = (roc_auc_score(truth, probability[:, 1]) if probability.shape[1] == 2
           else roc_auc_score(truth, probability, multi_class='ovr', average='macro'))
    return {
        'Loss': F.cross_entropy(logits, labels).item(),
        'BAcc': float(balanced_accuracy_score(truth, prediction)),
        'Auroc': float(auc),
        'wF1': float(f1_score(truth, prediction, average='weighted')),
        'Kappa': float(cohen_kappa_score(truth, prediction)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return classification_metrics(torch.cat(logits), torch.cat(labels))


def selection_key(per_dataset):
    """Dataset-macro BAcc, ties broken by smaller macro CE."""
    bacc = float(np.mean([m['BAcc'] for m in per_dataset.values()]))
    loss = float(np.mean([m['Loss'] for m in per_dataset.values()]))
    return bacc, loss, (bacc, -loss)


class BatchStream:
    """Endless batches from one loader; the iterator is rebuilt when exhausted."""

    def __init__(self, loader):
        self.loader, self.iterator = loader, iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def parse_datasets(value, parser):
    names = [name.strip() for name in value.split(',') if name.strip()]
    known = load_targets()
    unknown = [name for name in names if name not in known]
    if not names or unknown or len(set(names)) != len(names):
        parser.error(f'--ds must be distinct names from configs/tasks.yaml; got {value!r}')
    return names


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--ckpt', type=Path, required=True, help='Full instruction-tuned LEAF state dict; also searched in LEAF/checkpoints/')
    p.add_argument('--ds', default='MI_BCIC_IV2a', help='One dataset or a comma-separated list')
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--epochs', type=int, help='Override downstream.max_epochs (single dataset, epoch mode)')
    p.add_argument('--steps-per-dataset', type=int,
                   help='Step mode: number of rounds; each round is one optimizer step per dataset')
    p.add_argument('--eval-every-rounds', type=int, default=100, help='Step mode validation cadence')
    p.add_argument('--warmup-frac', type=float, default=0.05, help='Step mode warmup fraction of total steps')
    p.add_argument('--seed', type=int, default=42, help='Finetuning seed; checkpoint is selected explicitly by --ckpt')
    p.add_argument('--dim', help='Architecture override; must match the checkpoint')
    p.add_argument('--emb', help='Text embedding model override; must match the checkpoint')
    p.add_argument('--instrct', help='Override instruction with an exact cached string (single dataset only)')
    p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], help='Override downstream.precision; CPU always uses fp32')
    p.add_argument('--embedding-dir', type=Path, default=ROOT / 'text_embeddings')
    p.add_argument('--outdir', type=Path, help='New output directory; existing results are never overwritten')
    args = p.parse_args()
    args.datasets = parse_datasets(args.ds, p)
    if args.bs <= 0 or (args.epochs is not None and args.epochs <= 0):
        p.error('--bs and --epochs must be positive')
    if args.steps_per_dataset is not None:
        if args.steps_per_dataset <= 0 or args.eval_every_rounds <= 0:
            p.error('--steps-per-dataset and --eval-every-rounds must be positive')
        if args.epochs is not None:
            p.error('--epochs and --steps-per-dataset are mutually exclusive')
        if not 0 <= args.warmup_frac < 1:
            p.error('--warmup-frac must be in [0, 1)')
    elif len(args.datasets) > 1:
        p.error('Multi-dataset training requires --steps-per-dataset')
    if args.instrct is not None and len(args.datasets) > 1:
        p.error('--instrct is only supported for a single dataset')
    if args.gpu != '-1' and not args.gpu.isdigit():
        p.error('--gpu must be one nonnegative GPU index, or -1 for CPU')
    return args


def main():
    started = time.time()
    args = parse_args()
    names = args.datasets
    checkpoint = resolve_file(args.ckpt, ROOT / 'checkpoints')
    cfg = load_yaml(resolve_file(args.config, ROOT))
    config = build_model_config(cfg, args.dim, emb=args.emb)
    train = build_train_config(cfg, 'downstream')
    precision = args.precision or train.precision
    if precision not in ('32-true', 'bf16-mixed'):
        raise ValueError(f'Unsupported precision: {precision}')
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if precision == 'bf16-mixed' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support bf16; use --precision 32-true')
    use_bf16 = device.type == 'cuda' and precision == 'bf16-mixed'
    run_name = '+'.join(names)
    outdir = (args.outdir or checkpoint.parent / 'qformer_finetuning' / checkpoint.stem
              / run_name / f'seed{args.seed}').resolve()
    if outdir.exists():
        raise FileExistsError(f'Output directory already exists: {outdir}; choose a new --outdir')

    seed_all(args.seed)
    # Avoid convolution autotuning after repeated CUDA illegal-address failures.
    # Keep this local: seed_all is also used by unrelated training entry points.
    torch.backends.cudnn.benchmark = False
    leaf = load_leaf(config, checkpoint).to(device)
    texts, classifiers = {}, {}
    for ds in names:
        labels, text, instruction, prototypes = prepare_text(
            config, ds, args.instrct, args.embedding_dir,
        )
        texts[ds] = (labels, text)
        # All readouts share the same LEAF instance; only instruction/prototypes differ.
        classifiers[ds] = PrototypeClassifier(leaf, instruction, prototypes).to(device)

    train_loaders, val_loaders = {}, {}
    for ds in names:
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(ds)
        del test_x, test_y  # the test split is evaluated only by eval_poc.py
        n_classes = len(texts[ds][0])
        train_loaders[ds] = make_loader(train_x, train_y, config, n_classes, args.bs, True)
        val_loaders[ds] = make_loader(val_x, val_y, config, n_classes, args.bs, False)
        print(f'{ds}: train={len(train_y)} val={len(val_y)} classes={texts[ds][0]} '
              f'instruction={texts[ds][1]!r}', flush=True)
    data_seconds = time.time() - started

    if args.steps_per_dataset is not None:
        rounds = args.steps_per_dataset
        eval_rounds = args.eval_every_rounds
        total_steps = rounds * len(names)
        warmup_steps = int(round(args.warmup_frac * total_steps))
        mode = 'steps'
    else:
        epochs = args.epochs if args.epochs is not None else train.max_epochs
        if epochs <= 0 or not 0 <= train.warmup_epochs <= epochs:
            raise ValueError('Require 0 <= warmup_epochs <= epochs and epochs > 0')
        steps_per_epoch = len(train_loaders[names[0]])
        rounds = epochs * steps_per_epoch
        eval_rounds = steps_per_epoch
        total_steps = rounds
        warmup_steps = train.warmup_epochs * steps_per_epoch
        mode = 'epochs'
    eval_points = sorted({r for r in range(eval_rounds, rounds + 1, eval_rounds)} | {rounds})

    fast = list(leaf.tower.tokenizer.parameters())
    fast_ids = {id(param) for param in fast}
    slow = [param for param in leaf.parameters() if id(param) not in fast_ids]
    lr_fast = train.lr_base * train.lr_scale_fast_params
    lr_slow = train.lr_base * train.lr_scale_slow_params
    optimizer = torch.optim.AdamW([
        {'params': fast, 'lr': lr_fast},
        {'params': slow, 'lr': lr_slow},
    ], weight_decay=train.weight_decay)
    schedule = cosine_annealing_lr(1.0, 0.1, total_steps, 1, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(schedule))

    outdir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': sha256(checkpoint),
        'datasets': names,
        'labels': {ds: texts[ds][0] for ds in names},
        'instruction': {ds: texts[ds][1] for ds in names}, 'conditioning': 'task+targets',
        'seed': args.seed, 'mode': mode, 'rounds': rounds, 'total_steps': total_steps,
        'warmup_steps': warmup_steps, 'eval_every_rounds': eval_rounds,
        'lr_tokenizer': lr_fast, 'lr_rest': lr_slow, 'weight_decay': train.weight_decay,
        'lr_schedule': 'linear warmup then cosine to 0.1x peak (per optimizer step)',
        'precision': 'bf16-mixed' if use_bf16 else '32-true',
        'batch_size': args.bs, 'device': str(device),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'batchnorm': 'running statistics frozen at source checkpoint (SyncBatchNorm in eval mode); affine trained',
        'schedule': 'each round: one batch per dataset in shuffled order; one optimizer step per batch',
        'split_sizes': {ds: {'train': len(train_loaders[ds].dataset),
                             'validation': len(val_loaders[ds].dataset)} for ds in names},
        'model_config': config.to_dict(), 'training_config': train.to_dict(),
        'selection': 'maximum dataset-macro validation BAcc; ties resolved by minimum macro validation loss',
        'command': ' '.join(sys.argv),
    }
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    readout = {'datasets': np.asarray(names)}
    for ds in names:
        readout[f'{ds}__labels'] = np.asarray(texts[ds][0])
        readout[f'{ds}__instruction'] = classifiers[ds].instruction.cpu().numpy()
        readout[f'{ds}__prototypes'] = classifiers[ds].prototypes.cpu().numpy()
    np.savez(outdir / 'text_readout.npz', **readout)

    history = []  # one entry per validation point; round 0 is the untouched source model
    step_dataset = np.zeros(total_steps, dtype=np.int16)
    step_loss = np.zeros(total_steps, dtype=np.float32)
    step_lr = np.zeros(total_steps, dtype=np.float32)

    def validate(round_id, step, train_loss, elapsed):
        validation = {ds: evaluate(classifiers[ds], val_loaders[ds], device) for ds in names}
        bacc, loss, key = selection_key(validation)
        entry = {'round': round_id, 'step': step, 'elapsed_s': elapsed,
                 'val_macro_BAcc': bacc, 'val_macro_Loss': loss,
                 'train_loss': train_loss, 'val': validation}
        history.append(entry)
        train_mode_fixed_bn(leaf)
        per_ds = ' '.join(f'{ds}={validation[ds]["BAcc"]:.4f}' for ds in names)
        print(f'round {round_id}/{rounds} step {step}/{total_steps} '
              f'val macro BAcc={bacc:.4f} loss={loss:.4f} | {per_ds}', flush=True)
        return key

    print(f'Loaded all LEAF weights from {checkpoint}; no new trainable parameters.')
    print(f'{mode} mode: {rounds} rounds x {len(names)} datasets = {total_steps} optimizer steps, '
          f'warmup={warmup_steps}, validation every {eval_rounds} rounds', flush=True)
    train_started = time.time()
    validate(0, 0, None, 0.0)  # logged for the training curve; not a selection candidate
    streams = {ds: BatchStream(train_loaders[ds]) for ds in names}
    order_rng = np.random.default_rng(args.seed)
    best_key, best_round, best_step = (-math.inf, -math.inf), None, None
    interval = {ds: [0.0, 0] for ds in names}
    step = 0
    train_mode_fixed_bn(leaf)
    for round_id in range(1, rounds + 1):
        for index in order_rng.permutation(len(names)):
            ds = names[index]
            x, y = streams[ds].next()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = F.cross_entropy(classifiers[ds](x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {step + 1} ({ds})')
            loss.backward()
            step_lr[step] = optimizer.param_groups[1]['lr']
            optimizer.step()
            scheduler.step()
            value = loss.item()
            step_dataset[step], step_loss[step] = index, value
            interval[ds][0] += value * len(y)
            interval[ds][1] += len(y)
            step += 1
        if round_id in eval_points:
            train_loss = {ds: s / n for ds, (s, n) in interval.items() if n}
            interval = {ds: [0.0, 0] for ds in names}
            key = validate(round_id, step, train_loss, time.time() - train_started)
            if key > best_key:
                best_key, best_round, best_step = key, round_id, step
                temporary = outdir / 'best.ckpt.tmp'
                torch.save(leaf.state_dict(), temporary)
                temporary.replace(outdir / 'best.ckpt')
            np.savez(outdir / 'metrics.npz', step_dataset=step_dataset[:step],
                     step_loss=step_loss[:step], step_lr=step_lr[:step],
                     val_round=np.array([h['round'] for h in history]),
                     val_macro_BAcc=np.array([h['val_macro_BAcc'] for h in history]),
                     val_macro_Loss=np.array([h['val_macro_Loss'] for h in history]),
                     **{f'val{k}__{ds}': np.array([h['val'][ds][k] for h in history])
                        for ds in names for k in METRIC_KEYS},
                     bestRound=best_round)
            (outdir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')

    if step != total_steps:
        raise RuntimeError(f'Expected {total_steps} optimizer steps, ran {step}')
    best = next(h for h in history if h['round'] == best_round)
    metadata.update(
        best_round=best_round, best_step=best_step,
        best_val_macro_BAcc=best['val_macro_BAcc'], best_val_macro_Loss=best['val_macro_Loss'],
        best_val={ds: best['val'][ds] for ds in names},
        source_val={ds: history[0]['val'][ds] for ds in names},
        actual_optimizer_steps=step,
        steps_per_dataset_actual={ds: int((step_dataset == i).sum()) for i, ds in enumerate(names)},
        seconds={'data_loading': data_seconds, 'training_incl_validation': time.time() - train_started,
                 'total': time.time() - started},
    )
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Best round={best_round} (step {best_step}); val macro BAcc={best["val_macro_BAcc"]:.4f} '
          f'(source {history[0]["val_macro_BAcc"]:.4f})')
    print(f'Saved checkpoint, fixed text readout, and metrics to {outdir}')


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        # Exit status 86 lets callers identify a CUDA out-of-memory failure.
        # Other failures (including host-memory failures) must not trigger it.
        if 'cuda' not in str(exc).lower():
            raise
        traceback.print_exc()
        print('CUDA OOM: this attempt failed; exiting with status 86.', file=sys.stderr)
        sys.exit(86)
