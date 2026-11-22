"""Memory-bounded launch of the unchanged paired-prehead distillation trainer."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import train_prehead_distillation as trainer
from d_downstream_qformer_ce import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True);parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert sha256(Path(__file__))==plan['source_sha256'][Path(__file__).name]
    assert sha256(Path(plan['alignment_design']))==plan['alignment_design_sha256']
    start=time.time();torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'],args.gpu)
    sys.argv=[trainer.__file__,'--plan',str(args.plan),'--condition','ce_prehead_paired','--outdir',str(args.outdir),'--gpu',str(args.gpu)]
    trainer.main()
    result={'status':'complete','arm':plan['alignment_arm'],'plan_sha256':sha256(args.plan),
            'wrapper_sha256':sha256(Path(__file__)),'run_sha256':sha256(args.outdir/'run.json'),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(args.gpu),'seconds':time.time()-start,
            'execution':'Unchanged native paired-prehead trainer, optimizer, schedule, objective, original per-task auxiliary scale and mean6 export. Only the frozen teacher target coordinate system differs between arms.'}
    (args.outdir/'wrapper.json').write_text(json.dumps(result,indent=2)+'\n');print('Wrapper completed',plan['alignment_arm'],flush=True)


if __name__=='__main__':main()
