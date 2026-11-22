"""Finish the already-running matching reference, verify it, then launch fixed arms."""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path('runs/paper_merge_v2')
    launch_path=root/'inputs/prehead_reference_replay_launch.json'
    launch=json.loads(launch_path.read_text())
    receipt=root/'seed0/prehead_engineering_finish'
    receipt.mkdir(exist_ok=False)
    scripts=['verify_prehead_engineering_v2.py','launch_prehead_training_v2.py']
    frozen={name:sha(name) for name in [*scripts,__file__]}
    state={'started_unix':time.time(),'reference_launch':str(launch_path),
           'reference_launch_sha256':sha(launch_path),'source_sha256':frozen,
           'status':'waiting_for_reference','steps':[]}
    def record():
        state['updated_unix']=time.time()
        (receipt/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    record()
    try:
        proc=Path(f'/proc/{launch["pid"]}')
        while proc.exists():
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            assert stat[19]==launch['start_ticks'], 'reference PID identity changed'
            argv=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            assert argv==launch['argv'] or stat[0]=='Z'
            state['last_reference_process_state']=stat[0];record()
            time.sleep(5)
        assert (Path(launch['outdir'])/'run.json').exists(), 'reference ended without completion artifact'
        for script in scripts:
            assert sha(script)==frozen[script]
            state['status']='running';state['active_step']=script;record()
            log=receipt/f'{Path(script).stem}.log'
            with log.open('x') as f:
                child=subprocess.Popen(['/home/jingzhou/miniconda3/bin/python','-u',script],stdout=f,stderr=subprocess.STDOUT)
                state['active_pid']=child.pid;record()
                code=child.wait()
            state['steps'].append({'script':script,'exit_code':code,'log':str(log),'log_sha256':sha(log)})
            assert code==0, f'{script} failed; outputs retained, no retries'
        state['status']='complete';state.pop('active_pid',None);state.pop('active_step',None);record()
        print('Completed engineering verification and fixed four-arm queue launch',flush=True)
    except Exception as error:
        state['status']='failed';state['error']=repr(error);record()
        raise


if __name__=='__main__':main()
