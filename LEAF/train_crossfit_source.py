"""Use the existing source trainer with an audited train-only subset loader.

The original training file is not edited. The loader substitution is local to
this process and never loads the official test partition.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import d_downstream_qformer_ce as source_trainer
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER


def train_subset_loader(plan, task, fold, audit_path):
    with np.load(plan['folds']) as archive:
        ids = archive[f'{task}__train{fold}'].copy()
        assignment = archive[f'{task}__fold'].copy()
    metadata = json.loads(Path(plan['fold_audit']).read_text())['audits'][task]
    assert array_hash(ids) == metadata['training_index_hashes'][fold]
    assert array_hash(assignment) == metadata['assignment_hash']
    assert np.array_equal(ids, np.flatnonzero(assignment == fold))

    def loader(requested, val_ratio=.2):
        assert requested == task and val_ratio == .2
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as h:
            rows, y = row_map(h, task, 'calibration')
            valid_rows, vy = row_map(h, task, 'validation')
            y, vy = np.asarray(y, dtype=np.int64), np.asarray(vy, dtype=np.int64)
            assert array_hash(y) == metadata['full_train_labels_hash']
            assert array_hash(vy) == metadata['validation_labels_hash']
            assert not (set(rows) & set(valid_rows))
            x = selected_eeg(h, [rows[int(i)] for i in ids])
            vx = selected_eeg(h, valid_rows)
        truth = y[ids]
        assert array_hash(truth) == metadata['training_label_hashes'][fold]
        record = {'task': task, 'fold': fold, 'training_rows': len(truth),
            'training_index_hash': array_hash(ids), 'training_eeg_hash': array_hash(x),
            'training_labels_hash': array_hash(truth), 'validation_rows': len(vy),
            'validation_eeg_hash': array_hash(vx), 'validation_labels_hash': array_hash(vy),
            'excluded_other_fold_rows': int((assignment != fold).sum()),
            'official_test_read': False, 'full_train_size': len(y)}
        with audit_path.open('x') as out:
            json.dump(record, out, indent=2)
            out.write('\n')
        trial_time = 10 if task in {'EMO_FACED', 'ADHD_AliMotie'} else 4
        return trial_time, x, truth, vx, vy, None, None

    return loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert args.task in plan['tasks']
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert source_trainer.sha256(Path(path)) == digest, path
    label = f'{args.task}_fold{args.fold}'
    outdir = Path(plan['artifact_root']) / label
    link = Path(plan['project_source_root']) / label
    assert not outdir.exists() and not link.exists() and not link.is_symlink()
    outdir.parent.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outdir, target_is_directory=True)
    audit = Path(plan['audit_root']) / f'{label}_data.json'
    audit.parent.mkdir(parents=True, exist_ok=True)
    assert not audit.exists()
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source_trainer.load_dataset = train_subset_loader(plan, args.task, args.fold, audit)
    batch_sizes = []
    original_next = source_trainer.BatchStream.next

    def counted_next(stream):
        batch = original_next(stream)
        batch_sizes.append(len(batch[1]))
        return batch

    source_trainer.BatchStream.next = counted_next
    sys.argv = ['d_downstream_qformer_ce.py', '--ckpt', plan['base_checkpoint'],
        '--config', plan['config'], '--ds', args.task, '--steps-per-dataset', str(plan['source_steps']),
        '--eval-every-rounds', str(plan['eval_every_rounds']), '--bs', str(plan['batch_size']),
        '--seed', str(plan['source_seed']), '--gpu', str(args.gpu), '--precision', 'bf16-mixed',
        '--outdir', str(outdir)]
    started = time.monotonic()
    source_trainer.main()
    run = json.loads((outdir / 'run.json').read_text())
    assert run['actual_optimizer_steps'] == plan['source_steps']
    n, b, steps = run['split_sizes'][args.task]['train'], run['batch_size'], run['actual_optimizer_steps']
    passes, remaining = divmod(steps, (n + b - 1) // b)
    assert len(batch_sizes) == steps and sum(batch_sizes) == passes * n + remaining * b
    np.save(outdir / 'training_batch_sizes.npy', np.asarray(batch_sizes, dtype=np.int64))
    result = {'status': 'completed', 'task': args.task, 'fold': args.fold,
        'plan': str(args.plan), 'plan_sha256': source_trainer.sha256(args.plan),
        'run_sha256': source_trainer.sha256(outdir / 'run.json'),
        'checkpoint_sha256': source_trainer.sha256(outdir / 'best.ckpt'),
        'data_audit_sha256': source_trainer.sha256(audit),
        'optimizer_steps': steps, 'reconstructed_training_example_exposures': passes * n + remaining * b,
        'observed_training_example_exposures': sum(batch_sizes),
        'batch_size_trace_sha256': source_trainer.sha256(outdir / 'training_batch_sizes.npy'),
        'nominal_steps_times_maximum_batch_size': steps * b,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'elapsed_seconds': time.monotonic() - started, 'official_test_read': False,
        'pid': os.getpid(), 'outdir': str(outdir), 'project_link': str(link)}
    with (outdir / 'subset_completion.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
