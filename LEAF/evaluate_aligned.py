"""Run the unmodified official L2 evaluator on explicitly frozen, completed runs.

No model fitting or checkpoint selection happens here. A receipt records the
checkpoint hash and validation selection before any test inference. Existing
complete evaluations are reused; incomplete outputs require inspection.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def completed_result(path, datasets):
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['status'] != 'complete' or meta['completed_datasets'] != datasets:
        raise ValueError(f'Official evaluation incomplete: {path}')
    with path.open() as f:
        rows = {r['dataset']: r for r in csv.DictReader(f)}
    return {ds: float(rows[ds]['level2_bAcc']) for ds in datasets}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'runs/aligned_merge_v1/seed0')
    p.add_argument('--methods', required=True, help='Comma-separated completed run directory names')
    p.add_argument('--gpu', default='1')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    methods = args.methods.split(',')
    prepared = []
    for method in methods:
        source = root / method
        run = json.loads((source / 'run.json').read_text())
        checkpoint = source / 'merged.ckpt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        path = root / 'official_l2' / f'{method}.csv'
        command = [sys.executable, str(ROOT / 'c_direct_inference.py'), '--ckpt', str(checkpoint),
                   '--level', '2', '--seed', '42', '--bs', '256', '--ds', ','.join(run['datasets']),
                   '--gpu', args.gpu, '--output', str(path)]
        prepared.append((method, run, checkpoint, path, command))
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    out = root / 'official_l2'
    out.mkdir(exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    for method, run, checkpoint, path, command in prepared:
        if path.exists() or path.with_suffix('.json').exists():
            completed_result(path, run['datasets'])
            print(f'Reusing complete official evaluation: {method}', flush=True)
            continue
        receipt = {
            'method': method, 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
            'selected_step': run['best_step'], 'validation_macro_bacc': run['best_val_macro_BAcc'],
            'selection_source': str(root / method / 'run.json'),
            'selection_source_sha256': file_hash(root / method / 'run.json'),
            'selection_protocol': run.get('selection', 'validation macro BAcc, then validation CE; test not used for selection'),
            'evaluator_sha256': file_hash(ROOT / 'c_direct_inference.py'),
            'datasets': run['datasets'], 'command': command, 'started_unix': time.time(),
            'status': 'running',
        }
        receipt_path = out / f'{method}_receipt.json'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        with (out / f'{method}.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'Official evaluation {method}: pid={proc.pid}', flush=True)
            code = proc.wait()
        receipt.update(exit_code=code, elapsed_s=time.time() - receipt['started_unix'],
                       status='complete' if code == 0 else 'failed')
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        if code:
            raise RuntimeError(f'Official evaluator failed for {method}; inspect {out / (method + ".log")}')
        completed_result(path, run['datasets'])
    # Include every already-completed evaluation in the summary, not just this invocation.
    summary, per_task = [], []
    for path in sorted(out.glob('*.csv')):
        if path.name in ['summary.csv', 'per_dataset.csv']:
            continue
        meta_path = path.with_suffix('.json')
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('status') != 'complete':
            continue
        values = completed_result(path, meta['datasets'])
        summary.append({'method': path.stem, 'source_macro_bacc': sum(values.values()) / len(values),
                        'datasets': len(values), 'protocol': 'official L2, seed 42, batch 256'})
        per_task.extend({'method': path.stem, 'dataset': ds, 'bacc': value} for ds, value in values.items())
    for filename, rows in [('summary.csv', summary), ('per_dataset.csv', per_task)]:
        with (out / filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
