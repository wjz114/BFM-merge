"""Run frozen FP32 coefficient fits and fixed averages for additional expert banks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha(name) == expected, name
    receipt = Path(plan['receipt_directory'])
    receipt.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records = []
    for job in plan['jobs']:
        output, log = Path(job['outdir']), Path(job['log'])
        assert not output.exists() and not log.exists()
        log.parent.mkdir(parents=True, exist_ok=True)
        event = {'name': job['name'], 'started_unix': time.time(), 'command': job['command']}
        with log.open('w') as handle:
            process = subprocess.Popen(job['command'], stdout=handle, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            event['pid'] = process.pid
            (receipt/'active.json').write_text(json.dumps(event, indent=2)+'\n')
            print(json.dumps(event), flush=True)
            code = process.wait()
        event.update(exit_code=code, elapsed_seconds=time.time()-event['started_unix'])
        records.append(event)
        result = {'plan_sha256': sha(args.plan), 'jobs': records,
                  'status': 'failed' if code else 'running', 'seconds': time.time()-started}
        (receipt/'run.json').write_text(json.dumps(result, indent=2)+'\n')
        if code:
            raise RuntimeError(f"{job['name']} failed with exit {code}; output and log retained")
        assert (output/'run.json').exists()
    result['status'] = 'complete'
    result['seconds'] = time.time()-started
    (receipt/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed frozen initialization replication', plan['seed'], flush=True)


if __name__ == '__main__':
    main()
