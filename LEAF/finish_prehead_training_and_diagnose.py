"""Wait for both frozen queues, report all arms, then run the fixed fitting audit."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path('runs/paper_merge_v2')
    launch_path = root/'inputs/prehead_training_launch.json'
    launch = json.loads(launch_path.read_text())
    diagnostic_plan = root/'inputs/prehead_fitting_diagnostic_plan.json'
    receipt = root/'seed0/prehead_training_finish'
    receipt.mkdir(exist_ok=False)
    frozen = {str(p): sha(p) for p in [Path(__file__), Path('report_prehead_distillation.py'),
              Path('report_grouped_ce_controls.py'), Path('report_duration_ce_controls_v2.py'),
              Path('diagnose_prehead_fitting.py'), diagnostic_plan, launch_path]}
    state = {'started_unix': time.time(), 'source_input_sha256': frozen,
             'status': 'waiting_for_training', 'stages': []}
    def record():
        state['updated_unix'] = time.time()
        (receipt/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    def execute(name, commands):
        for path, digest in frozen.items():
            assert sha(path) == digest, path
        state['status'] = name; state['active'] = []; record()
        processes = []
        for index, command in enumerate(commands):
            log = receipt/f'{name}_{index}.log'
            with log.open('x') as handle:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            start = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            event = {'command': command, 'pid': child.pid, 'start_ticks': start, 'log': str(log)}
            processes.append((child, event)); state['active'].append(event); record()
        exits = []
        for child, event in processes:
            code = child.wait()
            event.update(exit_code=code, log_sha256=sha(event['log']))
            exits.append(code); record()
        state['stages'].append({'name': name, 'jobs': state.pop('active')}); record()
        assert all(code == 0 for code in exits), f'{name} failed; outputs retained, no retries'
    record()
    try:
        for queue in launch['controllers']:
            while True:
                proc = Path(f'/proc/{queue["pid"]}')
                if not proc.exists():
                    break
                stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                assert stat[19] == queue['start_ticks'], 'queue PID identity changed'
                if stat[0] == 'Z':
                    break
                argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
                assert argv == queue['argv']
                time.sleep(5)
            result = json.loads((Path(queue['receipt_directory'])/'run.json').read_text())
            assert result['status'] == 'complete' and all(j['exit_code'] == 0 for j in result['jobs'])
        python = '/home/jingzhou/miniconda3/bin/python'
        execute('primary_report', [[python, '-u', 'report_prehead_distillation.py']])
        execute('fitting_diagnostic', [[python, '-u', 'diagnose_prehead_fitting.py', 'run',
                '--plan', str(diagnostic_plan), '--shard', str(i), '--gpu', str(i)] for i in (0, 1)])
        execute('fitting_report', [[python, '-u', 'diagnose_prehead_fitting.py', 'report', '--plan', str(diagnostic_plan)]])
        state['status'] = 'complete'; record()
        print('Completed primary report and training-only fitting diagnostic', flush=True)
    except Exception as error:
        state['status'] = 'failed'; state['error'] = repr(error); record()
        raise


if __name__ == '__main__':
    main()
