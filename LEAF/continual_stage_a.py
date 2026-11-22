"""Stage A: two independent frozen specialists, nested 32/128/512 train budgets."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from continual_leaf import expert_features, load_student, save_student
from continual_readout import cold_start, pool_branches, scores, targets
from d_downstream_qformer_ce import ROOT, load_leaf, prepare_text, prototype_logits, sha256
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from load_datasets import PREDEFINED_SPLITS, SUBJECT_SPLITS

TASKS = ('MI_BCIC_IV2a', 'EMO_FACED')
PILOT_TASKS = (*TASKS, 'MI_OpenBMI', 'EMO_SEED_3_seg4')
BUDGETS = (32, 128, 512)


def key(row, salt):
    return hashlib.sha256(f'{salt}|{row}'.encode()).digest()


def stratified_order(ids, labels, salt):
    # Round robin across hash-ordered class lists gives nested, balanced prefixes.
    buckets = [sorted(np.flatnonzero(labels == c), key=lambda i: key(ids[i], salt))
               for c in np.unique(labels)]
    return [b[j] for j in range(max(map(len, buckets))) for b in buckets if j < len(b)]


def support_indices(ids, labels):
    groups = [sorted(np.flatnonzero(labels == c), key=lambda i: key(ids[i], 'split'))
              for c in np.unique(labels)]
    return [i for g in groups for i in g[:1 if len(g) == 1 else
                                         min(len(g)-1, max(1, round(.75*len(g))))]]


def stage_a_rows(f, task):
    # Mirror LEAF's concatenation + every-fifth validation split using metadata alone.
    if task in PREDEFINED_SPLITS:
        y, vy = f['trainY'][:], f['validY'][:]
        rows = [('trainX', i) for i in range(len(y))]
        order = stratified_order(rows, y, f'seed0|{task}')
        rows.extend(('validX', i) for i in range(len(vy)))
        return rows, np.concatenate([y, vy]), order[:max(BUDGETS)], np.arange(len(y), len(rows))
    rows, labels = [], []
    for subject in list(f)[:SUBJECT_SPLITS[task][0]]:
        y = f[subject]['Y'][:]
        rows.extend((subject, i) for i in range(len(y)))
        labels.extend(y)
    labels = np.asarray(labels)
    val = np.arange(0, len(rows), 5)
    train = np.setdiff1d(np.arange(len(rows)), val)
    order = stratified_order([rows[i] for i in train], labels[train], f'seed0|{task}')
    return rows, labels, train[order[:max(BUDGETS)]], val


def read_eeg(f, key, index):
    source = f[key]
    return (source if isinstance(source, h5py.Dataset) else source['X'])[index]


@torch.no_grad()
def extract(task, outdir, device, batch_size, bins=1):
    cfg = build_model_config(load_yaml())
    labels, prompt, inst, P = prepare_text(cfg, task)
    expert = ROOT / 'runs/poc16/seed0/experts' / task
    run = json.loads((expert / 'run.json').read_text())
    assert labels == run['labels'][task] and prompt == run['instruction'][task]
    leaf = load_leaf(cfg, expert / 'best.ckpt').eval().requires_grad_(False).to(device)
    inst, P = inst.to(device), P.to(device)
    data = {'task': task, 'labels': labels, 'teacher_prompt': prompt,
            'teacher_ckpt_sha256': sha256(expert / 'best.ckpt'), 'teacher_P': P.cpu(),
            'device': str(device), 'batch_size': batch_size, 'bins': bins}
    with h5py.File(DATA_FOLDER / f'{task}.h5', locking=False) as f:
        rows, y, train, val = stage_a_rows(f, task)
        for split, indices in [('train', train), ('validation', val)]:
            start = time.monotonic()
            ids = [rows[i] for i in indices]
            h, logits = [], []
            for j in range(0, len(ids), batch_size):
                x = torch.from_numpy(np.stack([read_eeg(f, s, i) for s,i in ids[j:j+batch_size]])).to(device)
                embedding, tokens = leaf(x, inst.expand(len(x), -1))
                logits.append(prototype_logits(embedding, P).cpu())
                h.append(pool_branches(tokens, bins).cpu())
            data[split] = {'ids': ids, 'y': torch.tensor(y[indices], dtype=torch.long),
                           'h': torch.cat(h), 'logits': torch.cat(logits),
                           'seconds': time.monotonic() - start}
            print(task, split, len(ids), round(data[split]['seconds'], 2), flush=True)
    torch.save(data, outdir / f'{task}_features.pt')


def teacher_summary(data):
    train, val = data['train'], data['validation']
    assert len(set(train['ids'])) == max(BUDGETS) and set(train['ids']).isdisjoint(val['ids'])
    return dict(task=data['task'], seed=0, device=data['device'], calibration=len(train['ids']),
                validation=len(val['ids']), teacher_ckpt_sha256=data['teacher_ckpt_sha256'],
                teacher_bacc=float(balanced_accuracy_score(val['y'], val['logits'].argmax(1))),
                calibration_encoder_seconds=train['seconds'], validation_encoder_seconds=val['seconds'],
                class_counts={n: train['y'][:n].bincount().tolist() for n in BUDGETS})


def fit(task, outdir, prototype_dir):
    data = torch.load(outdir / f'{task}_features.pt', weights_only=True)
    source = torch.load(prototype_dir / f'{task}.pt', weights_only=True)
    assert source['labels'] == data['labels']
    P = source['prototypes'].float()
    torch.testing.assert_close(P.norm(dim=1), torch.ones(len(P)), atol=1e-5, rtol=0)
    train, val = data['train'], data['validation']
    teacher = val['logits'].argmax(1)
    result = []
    cfg = load_yaml()['model']
    expert = ROOT / 'runs/poc16/seed0/experts' / task / 'best.ckpt'
    assert sha256(expert) == data['teacher_ckpt_sha256']
    bins = data.get('bins', 1)
    encoder = expert_features(build_model_config({'model': cfg}), expert, bins)
    with h5py.File(DATA_FOLDER / f'{task}.h5', locking=False) as f:
        probes = torch.from_numpy(np.stack([read_eeg(f, s, i) for s,i in val['ids'][:3]]))
    for n in BUDGETS:
        start = time.monotonic()
        y = train['y'][:n]
        support = support_indices(train['ids'][:n], y.numpy())
        q, rms = targets(train['logits'][:n], y, support)
        W, trace = cold_start(train['h'][:n], P, q, y)
        fit_seconds = time.monotonic() - start
        prediction = scores(val['h'], W, P).argmax(1)
        result.append(dict(task=task, bins=bins, budget=n, support_size=len(support), validation_size=len(teacher),
                           teacher_bacc=float(balanced_accuracy_score(val['y'], teacher)),
                           student_bacc=float(balanced_accuracy_score(val['y'], prediction)),
                           agreement=(teacher == prediction).float().mean().item(),
                           cg_residual=trace, rms=rms.item(), fit_seconds=fit_seconds))
        path = outdir / f'{task}_student_{n}.pt'
        save_student(path, encoder, W, {task: source}, cfg)
        with torch.no_grad():
            restored = load_student(path)(probes, task)
        expected = scores(val['h'][:len(probes)], W, P)
        torch.testing.assert_close(restored, expected, atol=2e-5, rtol=1e-4)
        result[-1].update(export_max_abs_error=(restored-expected).abs().max().item(),
                          prototype_source=source.get('source', str(prototype_dir / f'{task}.pt')), student_bytes=path.stat().st_size,
                          train_encoder_examples=n, eval_encoder_examples=len(teacher),
                          teacher_ckpt_sha256=data['teacher_ckpt_sha256'])
    (outdir / f'{task}_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(task, [(r['budget'], r['student_bacc']) for r in result], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['extract', 'fit', 'teachers'])
    parser.add_argument('--outdir', type=Path, default=ROOT/'runs/continual_v12/stage_a')
    parser.add_argument('--prototype-dir', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=PILOT_TASKS, default=TASKS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bins', type=int, choices=[1, 2], default=1, help='Time bins per branch during extraction')
    args = parser.parse_args()
    if args.bins != 1 and args.outdir.resolve() == ROOT/'runs/continual_v12/stage_a':
        parser.error('use a separate --outdir for segmented pooling')
    args.outdir.mkdir(exist_ok=True, parents=True)
    if args.action == 'teachers':
        rows = [teacher_summary(torch.load(args.outdir / f'{t}_features.pt', weights_only=True))
                for t in args.tasks]
        (args.outdir / 'teacher_validation.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(rows)
        parser.exit()
    for task in args.tasks:
        if args.action == 'extract':
            extract(task, args.outdir, torch.device(args.device), args.batch_size, args.bins)
        elif args.prototype_dir is None:
            parser.error('fit requires --prototype-dir')
        else:
            fit(task, args.outdir, args.prototype_dir)
