"""Wait for an idle GPU, run the fixed full-CE retry once, then report."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_observation(index):
    def query(kind, fields):
        output = subprocess.check_output(
            ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True)
        return [[v.strip() for v in row] for row in csv.reader(io.StringIO(output)) if row]
    gpu = next(row for row in query('gpu', 'index,uuid,memory.free') if int(row[0]) == index)
    processes = [row for row in query('compute-apps', 'gpu_uuid,pid,used_memory') if row[0] == gpu[1]]
    return {'unix': time.time(), 'gpu': index, 'uuid': gpu[1], 'free_mib': int(gpu[2]),
            'compute_processes': [{'pid': int(row[1]), 'memory_mib': row[2]} for row in processes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha(name) == digest, name
    outdir = Path(plan['outdir'])
    outdir.mkdir(exist_ok=False)
    started = time.time()
    state = {'status': 'running', 'stage': 'waiting_for_idle_gpu', 'plan_sha256': sha(args.plan), 'stages': []}

    def save():
        state['seconds'] = time.time() - started
        (outdir/'run.json').write_text(json.dumps(state, indent=2)+'\n')

    def run_stage(name, command):
        state['stage'] = name
        with (outdir/f'{name}.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            event = {'name': name, 'pid': process.pid, 'command': command, 'started_unix': time.time()}
            state['stages'].append(event)
            save()
            event['exit_code'] = process.wait()
            event['seconds'] = time.time()-event['started_unix']
            save()
        if event['exit_code']:
            raise RuntimeError(f"{name} exited {event['exit_code']}; no automatic retry")

    try:
        clear_since = None
        while True:
            observation = gpu_observation(plan['gpu'])
            ready = not observation['compute_processes'] and observation['free_mib'] >= plan['minimum_free_mib']
            clear_since = (observation['unix'] if clear_since is None else clear_since) if ready else None
            state.update(resource_observation=observation, clear_since_unix=clear_since)
            save()
            if clear_since is not None and observation['unix']-clear_since >= plan['idle_seconds']:
                break
            time.sleep(plan['poll_seconds'])
        # Recheck pins after waiting. This observation does not reserve the GPU.
        for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
            assert sha(name) == digest, name
        run_stage('training_queue', [plan['python'], '-u', 'run_frozen_research_queue.py', '--plan', plan['queue_plan']])
        queue = json.loads(Path(plan['queue_receipt']).read_text())
        assert queue['status'] == 'complete' and all(row['exit_code'] == 0 for row in queue['jobs'])
        run_stage('report', [plan['python'], plan['reporter']])
        state.update(status='complete', stage='complete', report=plan['report'])
        save()
    except Exception as error:
        state.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
