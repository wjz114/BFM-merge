"""Finish frozen engineering queues, verify, train all three arms, then report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from d_downstream_qformer_ce import sha256


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def identity(pid, expected):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
    except FileNotFoundError:
        return {'matches': False, 'state': 'absent'}
    return {'matches': fields[21] == str(expected) and fields[2] != 'Z',
            'state': fields[2], 'startticks': fields[21]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for p,h in {**plan['source_sha256'], **plan['input_sha256']}.items(): assert sha256(Path(p)) == h,p
    output = Path(plan['outdir']); output.mkdir(exist_ok=False)
    started = time.time()
    state = {'status':'running','stage':'waiting_engineering','plan_sha256':sha256(args.plan),'stages':[]}

    def save():
        state['seconds'] = time.time()-started
        (output/'run.json').write_text(json.dumps(state,indent=2)+'\n')

    def wait_queues(queues):
        while True:
            observations, complete = [], True
            for queue in queues:
                receipt = Path(queue['receipt'])/'run.json'
                result = read_optional(receipt)
                current = identity(queue['pid'],queue['startticks'])
                row = {'pid':queue['pid'],'identity':current,'receipt_status':None if result is None else result['status']}
                observations.append(row)
                if result is not None and result['status'] == 'complete':
                    assert all(j['exit_code'] == 0 for j in result['jobs'])
                    continue
                if result is not None and result['status'] == 'failed':
                    raise RuntimeError(f'Queue failed: {receipt}')
                complete = False
                if not current['matches']:
                    # Re-read a possible final receipt write; never restart a missing queue.
                    time.sleep(2)
                    result = read_optional(receipt)
                    if result is None or result['status'] != 'complete':
                        raise RuntimeError(f'Queue handle terminal without complete receipt: {receipt}')
            state['queue_observations'] = observations; save()
            if complete:
                return
            time.sleep(10)

    def stage(name, command):
        state['stage'] = name; save()
        with (output/f'{name}.log').open('x') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            event = {'name':name,'pid':process.pid,'command':command,'started_unix':time.time()}
            state['stages'].append(event); save()
            code = process.wait(); event['exit_code'] = code; save()
        if code:
            raise RuntimeError(f'{name} exited {code}')

    try:
        save()
        engineering = json.loads(Path(plan['engineering_launch']).read_text())
        wait_queues(engineering['queues'])
        stage('verify_engineering',[plan['python'],'verify_row_projected_engineering.py'])
        gate = Path(plan['engineering_verification'])
        assert json.loads(gate.read_text())['full_ce_reference_metrics_histories_dynamics_exact']
        state['engineering_verification_sha256'] = sha256(gate)
        queues = []
        for specification in plan['scientific_queues']:
            qp = Path(specification['plan'])
            data = {'label':specification['label'],'receipt_directory':specification['receipt'],
                    'source_sha256':plan['training_queue_sources'],
                    'input_sha256':{plan['training_plan']:sha256(Path(plan['training_plan'])),str(gate):sha256(gate)},
                    'jobs':specification['jobs']}
            with qp.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
            command=[plan['python'],'-u','run_frozen_research_queue.py','--plan',str(qp)]
            with Path(specification['log']).open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            startticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            queues.append({'pid':process.pid,'startticks':startticks,'plan':str(qp),'plan_sha256':sha256(qp),
                           'receipt':specification['receipt'],'log':specification['log'],'gpu':specification['gpu']})
        launch={'unix':time.time(),'controller_plan_sha256':sha256(args.plan),'training_plan_sha256':sha256(Path(plan['training_plan'])),
                'engineering_verification_sha256':sha256(gate),'queues':queues}
        with Path(plan['scientific_launch']).open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
        state['stage']='waiting_scientific_training';state['scientific_launch']=plan['scientific_launch'];save()
        print('All scientific queues launched',json.dumps(queues),flush=True)
        wait_queues(queues)
        stage('report',[plan['python'],'report_row_projected_training.py'])
        state['status']='complete';state['stage']='complete';save()
        print('Completed frozen row-space experiment and report',flush=True)
    except Exception as error:
        state['status']='failed';state['error']=repr(error);save()
        raise


if __name__=='__main__':main()
