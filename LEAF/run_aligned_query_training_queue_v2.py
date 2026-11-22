"""Serial resource-checked target-coordinate comparison and its frozen reporter."""
import json
import os
from pathlib import Path
import subprocess
import time
from d_downstream_qformer_ce import sha256


def main():
    root=Path('runs/paper_merge_v2');plan_path=root/'inputs/aligned_query_queue_plan_v2.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['source_sha256'].items():assert sha256(Path(p))==h,p
    status_path=Path(plan['status']);assert not status_path.exists()
    state={'unix':time.time(),'status':'running','plan_sha256':sha256(plan_path),'workers':{}}
    def save():status_path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    for arm in ['original','aligned']:
        spec=plan['arms'][arm];assert sha256(Path(spec['plan']))==spec['plan_sha256']
        assert not Path(spec['outdir']).exists() and not Path(spec['log']).exists()
        cmd=[plan['python'],'-u','train_aligned_query_targets_v2.py','--plan',spec['plan'],'--outdir',spec['outdir'],'--gpu','0']
        with Path(spec['log']).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':plan['cuda_allocator']})
            state['active_arm']=arm;state['workers'][arm]={'pid':process.pid,'startticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],'command':cmd,'started_unix':time.time()};save()
            code=process.wait()
        state['workers'][arm].update(exit_code=code,completed_unix=time.time());save()
        if code!=0:state.update(status='worker_failed',active_arm=None);save();return
        result=json.loads((Path(spec['outdir'])/'wrapper.json').read_text());assert result['status']=='complete'
    state.update(status='reporting',active_arm=None);save()
    with Path(plan['report_log']).open('x') as log:
        code=subprocess.run([plan['python'],plan['reporter']],stdout=log,stderr=subprocess.STDOUT).returncode
    state.update(status='complete' if code==0 else 'reporter_failed',reporter_exit_code=code,completed_unix=time.time());save()


if __name__=='__main__':main()
