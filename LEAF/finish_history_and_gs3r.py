"""Finish the frozen replication report, then run the prepared GS3R references."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('runs/streaming_history_replication_v1')
REFERENCE = Path('runs/gs3r_reference_v1')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    return {'state': fields[0], 'start_ticks': fields[19]}


def run_reference(order, gpu, plan_path):
    command = [sys.executable, 'run_gs3r_reference.py', '--plan', plan_path,
               '--outdir', str(REFERENCE / order), '--order', order, '--gpu', str(gpu)]
    assert not (REFERENCE / order).exists()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (REFERENCE / f'{order}.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        launch = {'unix': time.time(), 'pid': process.pid, 'command': command}
        write(REFERENCE / f'{order}_launch.json', launch)
        print(json.dumps(launch), flush=True)
        code = process.wait()
    result = {'order': order, 'returncode': code, 'ended_unix': time.time()}
    if code == 0:
        path = REFERENCE / order / 'run.json'
        record = json.loads(path.read_text())
        assert record['status'] == 'completed' and len(record['stages']) == 16
        result.update(macro_BAcc=record['stages'][-1]['macro_BAcc'], source=str(path),
                      source_sha256=sha256(path))
    print(json.dumps(result), flush=True)
    return result


def main():
    plan_path = ROOT / 'followup_plan.json'
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['sha256'].items():
        assert sha256(path) == digest, path
    state_path = ROOT / 'followup_state.json'
    assert not state_path.exists()
    state = {'status': 'waiting_for_replication', 'unix': time.time(),
             'controller_pid': os.getpid(), 'plan': str(plan_path),
             'plan_sha256': sha256(plan_path), 'replication_controller': plan['replication_controller']}
    write(state_path, state)
    print(json.dumps(state), flush=True)
    try:
        expected = plan['replication_controller']
        while True:
            current = process_identity(expected['pid'])
            if current is None or current['state'] == 'Z' or current['start_ticks'] != expected['start_ticks']:
                break
            time.sleep(30)
        results = json.loads((ROOT / 'queue_results.json').read_text())
        assert sum(map(len, results)) == 12
        assert all(r['returncode'] == 0 for bank in results for r in bank)
        for bank in (1, 2):
            for order in ('forward', 'reverse'):
                for condition in ('incoming_rows', 'history_sketch', 'random_sketch'):
                    record = json.loads((ROOT / f'seed{bank}' / f'{condition}_{order}' / 'run.json').read_text())
                    assert record['status'] == 'completed' and len(record['stages']) == 16
        state.update(status='reporting_replication', unix=time.time())
        write(state_path, state)
        assert sha256(plan['reporter']) == plan['sha256'][plan['reporter']]
        with (ROOT / 'full_report.log').open('x') as log:
            subprocess.run([sys.executable, plan['reporter']], stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status='running_gs3r_references', unix=time.time(),
                     replication_report=str(ROOT / 'comparison.json'))
        write(state_path, state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_reference, order, gpu, plan['reference_plan'])
                       for order, gpu in [('forward', 0), ('reverse', 1)]]
            references = [f.result() for f in futures]
        write(REFERENCE / 'queue_results.json', references)
        assert all(r['returncode'] == 0 for r in references)
        state.update(status='completed', unix=time.time(), references=references,
                     overall_research_goal_achieved=False)
        write(state_path, state)
    except Exception as error:
        state.update(status='failed', unix=time.time(), error=repr(error))
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
