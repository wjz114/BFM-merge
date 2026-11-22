"""Run named experiments with exact commands and durable exit records.

Example: python run_aligned_stage.py --gpu 1 --jobs group_scale,repair_base,repair_block,repair_cpsc
Use --dry-run to inspect commands without running. Existing completed runs are
reported, existing incomplete outputs stop the queue (never silently restart).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0')
    p.add_argument('--jobs', required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], default=0,
                   help='Expert, existing calibration, and optimizer seed; seeds 1/2 support the tensor replication')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.seed and set(args.jobs.split(',')) - {'block_ce', 'tensor_ce', 'tensor_scale_ce',
                                                'block_balanced_ce', 'tensor_balanced_ce',
                                                'block_balanced_avg', 'tensor_balanced_avg'}:
        p.error('Seeds1/2 support the prepared Block/tensor CE, class-balanced CE, and fixed averaging jobs')
    RUN = ROOT / f'runs/aligned_merge_v1/seed{args.seed}'
    POC = ROOT / f'runs/poc16/seed{args.seed}'
    CAL = POC / 'merge_block/calibration_indices.npz'
    names = json.loads((POC / 'merge_block/run.json').read_text())['datasets']
    common = ['--base', BASE, '--datasets', ','.join(names), '--calibration', CAL,
              '--steps', '1600', '--eval-every', '160', '--bs', '128', '--seed', str(args.seed), '--gpu', args.gpu]
    commands = []
    for job in args.jobs.split(','):
        if job in ('block_balanced_avg', 'tensor_balanced_avg'):
            cmd = [sys.executable, 'average_merge_trajectory.py',
                   '--source', RUN / job.replace('_avg', '_ce'), '--gpu', args.gpu]
        elif job in ('regmean_adapted', 'regmeanpp_adapted'):
            cmd = [sys.executable, 'merge_regmean.py', '--method', job.removesuffix('_adapted'),
                   '--base', BASE, '--experts-dir', POC / 'experts', '--calibration', CAL,
                   '--datasets', ','.join(names), '--alpha', '0.95', '--bs', '32', '--seed', '0', '--gpu', args.gpu]
        elif job in ('eta01', 'eta10'):
            eta = '0.1' if job == 'eta01' else '10'
            expert_dir = RUN / f'experts_cpsc_{job}'
            commands.append((f'fold_{job}', expert_dir / 'alignment_diagnostics.json', [sys.executable, 'cpsc_fold.py',
                '--experts-dir', POC / 'experts', '--calibration', CAL, '--eta', eta, '--gpu', args.gpu,
                '--outdir', expert_dir]))
            cmd = [sys.executable, 'merge_poc.py', *common, '--experts-dir', expert_dir, '--method', 'block']
        elif job.startswith('proj_'):
            _, initialization, mode = job.split('_')
            ckpt = {'base': BASE, 'block': RUN / 'block_ce/merged.ckpt',
                    'cpsc': RUN / 'cpsc_block_ce/merged.ckpt'}[initialization]
            cmd = [sys.executable, 'repair_interface.py', '--checkpoint', ckpt, '--calibration', CAL,
                   '--gpu', args.gpu, '--seed', '0', '--mode', mode]
        elif job.startswith('repair_'):
            ckpt = {'repair_base': BASE, 'repair_block': RUN / 'merge_block_mseed0/merged.ckpt',
                    'repair_cpsc': RUN / 'merge_cpsc_block/merged.ckpt',
                    'repair_block_fp32': RUN / 'block_ce/merged.ckpt',
                    'repair_cpsc_fp32': RUN / 'cpsc_block_ce/merged.ckpt',
                    'repair_base_full': BASE, 'repair_block_full': RUN / 'block_ce/merged.ckpt',
                    'repair_cpsc_full': RUN / 'cpsc_block_ce/merged.ckpt'}[job]
            cmd = [sys.executable, 'repair_interface.py', '--checkpoint', ckpt, '--calibration', CAL,
                   '--gpu', args.gpu, '--seed', '0']
            if job.endswith('_full'):
                cmd += ['--mode', 'full']
        else:
            methods = {'group_scale': ('group_scale', 'ce'), 'block_semkd': ('block', 'semkd'),
                       'block_ce': ('block', 'ce'), 'cpsc_block_ce': ('block', 'ce'),
                       'cpsc_block_semkd': ('block', 'semkd'), 'block_logitkd': ('block', 'logitkd'),
                       'expert_merging_adapted': ('block', 'expert_adapted'),
                       'block_hidden_ce': ('block', 'hidden_ce'),
                       'ties_fixed_adapted': ('uniform', 'ce'),
                       'ties_block_ce_adapted': ('block', 'ce'),
                       'tensor_ce': ('block', 'ce'), 'tensor_scale_ce': ('group_scale', 'ce'),
                       'block_balanced_ce': ('block', 'ce'), 'tensor_balanced_ce': ('block', 'ce'),
                       'cpsc_uniform': ('uniform', 'ce')}
            method, loss = methods[job]
            experts = RUN / 'experts_cpsc_eta1' if job.startswith('cpsc') else POC / 'experts'
            cmd = [sys.executable, 'merge_poc.py', *common, '--experts-dir', experts,
                   '--method', method, '--loss', loss]
            if job.startswith('ties_'):
                cmd += ['--ties-density', '0.2']
            if job.startswith('tensor_'):
                cmd += ['--granularity', 'tensor']
            if job.endswith('balanced_ce'):
                cmd += ['--class-balanced-ce']
            if loss != 'ce':
                cmd += ['--teacher-cache', RUN.parent / 'teacher_cache']
            if loss in ('expert_adapted', 'hidden_ce'):
                cmd += ['--kd-weight', '1', '--hidden-weight', '1', '--anchor-weight', '0.1']
        commands.append((job, RUN / job / 'run.json', [*cmd, '--outdir', RUN / job]))
    commands = [(name, done, [str(v) for v in cmd]) for name, done, cmd in commands]
    if args.dry_run:
        for name, done, cmd in commands:
            print(shlex.join(cmd))
        return
    (RUN / 'logs').mkdir(exist_ok=True, parents=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    ledger = RUN / 'queue_events.jsonl'
    for name, done, cmd in commands:
        if done.exists():
            print(f'already complete: {name}', flush=True)
            continue
        if done.parent.exists():
            raise FileExistsError(f'Incomplete output exists; inspect before resuming: {done.parent}')
        start = time.time()
        event = {'job': name, 'command': cmd, 'started_unix': start, 'gpu': args.gpu, 'seed': args.seed}
        with ledger.open('a') as f:
            f.write(json.dumps({**event, 'status': 'started'}) + '\n')
        print(f'start {name}: {shlex.join(cmd)}', flush=True)
        with (RUN / 'logs' / f'{name}.log').open('w') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'pid={process.pid}', flush=True)
            code = process.wait()
        with ledger.open('a') as f:
            f.write(json.dumps({**event, 'status': 'complete' if code == 0 else 'failed',
                                'exit_code': code, 'elapsed_s': time.time() - start}) + '\n')
        if code or not done.exists():
            raise RuntimeError(f'{name} failed; inspect {RUN / "logs" / (name + ".log")}')
        print(f'complete {name}', flush=True)


if __name__ == '__main__':
    main()
