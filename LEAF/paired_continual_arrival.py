"""Paired readouts at the second and third arrivals of order 43."""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from continual_leaf import RecursiveMerge, expert_features
from continual_readout import cold_start, pool_branches, scores
from continual_stage_a import DATA_FOLDER, read_eeg, stratified_order
from d_downstream_qformer_ce import load_leaf, prepare_text, prototype_logits
from pilot_continual_stream import (BANK, BASE, RUN, arrival, bacc, build_model_config,
                                    features, fit_arrival, load_yaml, refit)

ORDER = ['MI_OpenBMI', 'EMO_FACED', 'MI_BCIC_IV2a']
DEST = RUN/'paired_arrival_order43'


@torch.no_grad()
def evaluate(task, seed, n, config, checkpoint, expert, merged, W, d, q, out):
    # Both independent heads use the same new-task samples and the Stage-A cold-start solver.
    P, y = d['P'], d['y']
    weights = [cold_start(features(enc,d['x']),P,q,y)[0] for enc in (expert,merged)]
    data = torch.load(RUN/'stage_a'/f'{task}_features.pt', weights_only=True)
    val = data['validation']
    assert d['ids'] == data['train']['ids'][:n] and set(val['ids']).isdisjoint(d['ids'])
    leaf = load_leaf(config,checkpoint).requires_grad_(False).eval()
    labels, _, instruction, teacher_P = prepare_text(config, task)
    assert labels == data['labels']
    batches = []
    with h5py.File(DATA_FOLDER/f'{task}.h5', locking=False) as f:
        for i in range(0,len(val['y']),4):
            x = torch.from_numpy(np.stack([read_eeg(f,s,k) for s,k in val['ids'][i:i+4]]))
            emb,tokens = leaf(x,instruction.expand(len(x),-1))
            a, b = pool_branches(tokens), merged(x)
            batches.append([prototype_logits(emb,teacher_P),scores(a,weights[0],P),
                            scores(b,weights[1],P),scores(b,W,P)])
    names = ['teacher','expert_independent','merged_independent','merged_shared']
    logits = {k:torch.cat(v) for k,v in zip(names,zip(*batches))}
    probe = stratified_order(val['ids'],val['y'].numpy(),'stream-probe')[:64]
    metric = {k:bacc(val['y'],s.argmax(1)) for k,s in logits.items()}
    probe_metric = {k:bacc(val['y'][probe],s[probe].argmax(1)) for k,s in logits.items()}
    if n==32:
        reference = original(seed)
        prior = next(s['eval'][task] for s in json.loads((reference/'results.json').read_text()) if s['arrival']==task)
        assert abs(prior['bacc']-probe_metric['merged_shared'])<1e-12
        assert abs(prior['teacher_bacc']-probe_metric['teacher'])<1e-12
    if seed==0:
        prior = json.loads((RUN/'stage_a'/f'{task}_results.json').read_text())
        assert abs(next(r['student_bacc'] for r in prior if r['budget']==n)-metric['expert_independent'])<1e-12
    torch.save(dict(calibration_ids=d['ids'],validation_ids=val['ids'],y=val['y'],logits=logits),
               out/f'{task}_scores.pt')
    result = dict(seed=seed,budget=n,task=task,validation=len(val['y']),**metric,probe64=probe_metric)
    (out/f'{task}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(seed,n,task,metric,flush=True)


def original(seed):
    if seed==0:
        return RUN/'stage_b_smoke/43/full'
    return RUN/'stage_b_replication/order43'/f'seed{seed}'/'full'


def run(seed,n):
    cfg=build_model_config(load_yaml())
    base=expert_features(cfg,BASE)
    bank=BANK.parent.parent/f'seed{seed}'/'experts'
    out=DEST/f'seed{seed}'/f'n{n}'
    out.mkdir(parents=True,exist_ok=True)
    memory={}
    for step,task in enumerate(ORDER,1):
        d,q,_,_=arrival(task,n,seed)
        checkpoint=bank/task/'best.ckpt'
        expert=expert_features(cfg,checkpoint)
        if step==1:
            encoder=expert
            W,_=cold_start(features(encoder,d['x']),d['P'],q,d['y'])
        else:
            if n==32:
                state=torch.load(original(seed)/f'coefficients_{step}.pt',weights_only=True)
                merger=RecursiveMerge(base,encoder,expert,step)
                assert state['parameter_names']==list(merger.old_delta)
                refs=[scores(features(encoder,v['x']),W,v['P']) for v in memory.values()]
                encoder=merger.fold(state['mean'])
                W=refit(encoder,W,[*memory.values(),d],[*refs,q])
            else:
                encoder,W,*_=fit_arrival('full',base,encoder,expert,step,memory,W,d,q,
                                         np.random.default_rng([43,step]),out)
            evaluate(task,seed,n,cfg,checkpoint,expert,encoder,W,d,q,out)
        if step<len(ORDER):
            memory[task]=d
            assert sum(len(v['x']) for v in memory.values())<=1024
            assert sum(v['x'].nbytes for v in memory.values())<=2**30


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--budget',type=int,choices=[32,512],required=True)
    args=p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    run(args.seed,args.budget)
