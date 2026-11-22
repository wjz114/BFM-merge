"""Run the fixed missing averaging controls with durable child/exit records."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--jobs', default='block_ce_avg,tensor_ce_avg')
    args = parser.parse_args()
    root = ROOT / 'runs/paper_merge_v2' / f'seed{args.seed}'
    source = ROOT / 'runs/aligned_merge_v1' / f'seed{args.seed}'
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    for name in args.jobs.split(','):
        assert name in ['block_ce_avg', 'tensor_ce_avg']
        out = root / name
        if (out / 'run.json').exists():
            print('Already complete:', out, flush=True)
            continue
        if out.exists():
            raise FileExistsError(f'Incomplete output requires inspection: {out}')
        command = [sys.executable, str(ROOT / 'average_merge_trajectory.py'),
                   '--source', str(source / name.removesuffix('_avg')),
                   '--outdir', str(out), '--gpu', str(args.gpu)]
        started = time.time()
        with (root / 'logs' / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            event = {'job': name, 'seed': args.seed, 'gpu': args.gpu, 'pid': process.pid,
                     'started_unix': started, 'command': command}
            with (root / 'queue_events.jsonl').open('a') as handle:
                handle.write(json.dumps({**event, 'status': 'started'}) + '\n')
            print(json.dumps(event), flush=True)
            code = process.wait()
        with (root / 'queue_events.jsonl').open('a') as handle:
            handle.write(json.dumps({**event, 'status': 'complete' if code == 0 else 'failed',
                                    'exit_code': code, 'elapsed_s': time.time() - started}) + '\n')
        if code or not (out / 'run.json').exists():
            raise RuntimeError(f'{name} failed; inspect its preserved log/output')


if __name__ == '__main__':
    main()
