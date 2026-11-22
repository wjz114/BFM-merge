"""Launch the four prespecified arms only after both engineering jobs pass."""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    root=Path('runs/paper_merge_v2')
    design_path=root/'inputs/prehead_distillation_design.json'
    design=json.loads(design_path.read_text())
    gate_path=root/'analysis/prehead_engineering_verification.json'
    gate=json.loads(gate_path.read_text())
    assert gate['all_zero_auxiliary_metrics_histories_data_gradients_exact']
    assert gate['paired_smoke_finite_active_gradients_and_native_export']
    for name,digest in gate['sources'].items():assert sha(name)==digest,name
    reference_launch_path=root/'inputs/prehead_reference_replay_launch.json'
    reference_launch=json.loads(reference_launch_path.read_text())
    assert not Path(f'/proc/{reference_launch["pid"]}').exists(), 'reference process not yet terminal'
    assert (Path(reference_launch['outdir'])/'run.json').exists()
    assert gate['engineering_optimizer_updates']==96
    amendment_path=Path(gate['reference_amendment']['path'])
    assert sha(amendment_path)==gate['reference_amendment']['sha256']
    old_launch=json.loads((root/'inputs/prehead_engineering_launch.json').read_text())
    for job in old_launch['jobs']:
        assert not Path(f'/proc/{job["pid"]}').exists(), 'engineering process not yet terminal'
        assert (Path(job['outdir'])/'run.json').exists()
    plan_path=Path(design['training_plan'])
    plan=json.loads(plan_path.read_text())
    assert plan['design_sha256']==sha(design_path)
    for name,digest in plan['source_sha256'].items():assert sha(name)==digest,name
    sources={**plan['source_sha256'],'run_frozen_research_queue.py':sha('run_frozen_research_queue.py'),
             str(Path(__file__).name):sha(__file__)}
    inputs={str(p):sha(p) for p in [design_path,plan_path,gate_path,amendment_path,reference_launch_path,Path(design['target_manifest']),Path(design['scales'])]}
    specs=[]
    for gpu in range(2):
        jobs=[]
        for condition,spec in sorted(design['training_arms'].items(),key=lambda item:item[1]['wave']):
            if spec['gpu']!=gpu:continue
            assert not Path(spec['outdir']).exists() and not Path(spec['log']).exists()
            jobs.append({'name':condition,'outdir':spec['outdir'],'log':spec['log'],
                'command':['/home/jingzhou/miniconda3/bin/python','-u','train_prehead_distillation.py',
                    '--plan',str(plan_path),'--condition',condition,'--outdir',spec['outdir'],'--gpu',str(gpu)]})
        qp=root/f'inputs/prehead_training_gpu{gpu}_queue_plan.json'
        receipt=root/f'seed0/prehead_training_gpu{gpu}_queue'
        write(qp,{'label':f'prehead target controls GPU{gpu}','source_sha256':sources,'input_sha256':inputs,
                  'receipt_directory':str(receipt),'jobs':jobs})
        specs.append({'gpu':gpu,'plan':str(qp),'plan_sha256':sha(qp),'receipt_directory':str(receipt),
                      'log':str(root/f'seed0/logs/prehead_training_gpu{gpu}_queue.log')})
    controllers=[]
    for spec in specs:
        argv=['/home/jingzhou/miniconda3/bin/python','-u','run_frozen_research_queue.py','--plan',spec['plan']]
        with Path(spec['log']).open('x') as f:
            p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        stat=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
        controllers.append({**spec,'pid':p.pid,'start_ticks':stat[19],'argv':argv})
    out=root/'inputs/prehead_training_launch.json'
    write(out,{'unix':time.time(),'design':str(design_path),'design_sha256':sha(design_path),
               'engineering':str(gate_path),'engineering_sha256':sha(gate_path),'controllers':controllers})
    print(json.dumps({'launch':str(out),'sha256':sha(out),'controllers':controllers},indent=2))


if __name__=='__main__':main()
