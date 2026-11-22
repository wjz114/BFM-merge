"""Four-task mechanism smoke. 32 calibration EEG per arrival, 64 validation probes per task."""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score

from continual_leaf import RecursiveMerge, expert_features, load_student, save_student
from continual_readout import calibrate, class_weights, cold_start, query_loss, scores, targets
from continual_stage_a import (DATA_FOLDER, ROOT, TASKS, build_model_config, load_yaml,
                              read_eeg, stratified_order, support_indices)
from d_downstream_qformer_ce import load_classifier, prepare_text, sha256

RUN = ROOT/'runs/continual_v12'
BANK = ROOT/'runs/poc16/seed0/experts'
BASE = ROOT/'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'
MANIFEST = ['MI_BCIC_IV2a','MI_OpenBMI','EMO_FACED','EMO_SEED_3_seg4']


def arrival(task, n, expert_seed=0):
    # The external evaluation cache supplies independent fixed validation IDs.
    folder = RUN/('stage_a' if task in TASKS else 'stage_b_reference')
    data = torch.load(folder/f'{task}_features.pt', weights_only=True)
    expert_dir = BANK.parent.parent/f'seed{expert_seed}'/'experts'/task
    run = json.loads((expert_dir/'run.json').read_text())
    assert run['seed'] == expert_seed and run['source_checkpoint_sha256'] == sha256(BASE)
    assert run['labels'][task] == data['labels']
    vocab = torch.load(RUN/'stage_b_reference/prototypes'/f'{task}.pt', weights_only=True)
    assert data['labels'] == vocab['labels']
    tr, val = data['train'], data['validation']
    ids, y = tr['ids'][:n], tr['y'][:n]
    S = np.array(support_indices(ids, y.numpy()))
    Q = np.setdiff1d(np.arange(n), S)
    val_ids = stratified_order(val['ids'], val['y'].numpy(), 'stream-probe')[:64]
    with h5py.File(DATA_FOLDER/f'{task}.h5', locking=False) as f:
        def eeg(rows):
            return torch.from_numpy(np.stack([read_eeg(f,s,i) for s,i in rows]))
        train = eeg(ids)
        heldout = [val['ids'][i] for i in val_ids]
        assert set(ids).isdisjoint(heldout)
        eval_x = eeg(heldout)
    if expert_seed == 0:
        assert data['teacher_ckpt_sha256'] == sha256(expert_dir/'best.ckpt')
        train_logits, eval_logits = tr['logits'][:n], val['logits'][val_ids]
    else:
        cfg = build_model_config(load_yaml())
        labels, prompt, inst, P = prepare_text(cfg, task)
        assert labels == data['labels'] and prompt == run['instruction'][task]
        teacher = load_classifier(cfg, expert_dir/'best.ckpt', inst, P).requires_grad_(False).eval()
        with torch.no_grad():
            train_logits, eval_logits = [torch.cat([teacher(batch) for batch in x.split(4)])
                                         for x in (train, eval_x)]
    evaluation = (eval_x,val['y'][val_ids],eval_logits.argmax(1))
    target,_ = targets(train_logits, y, S)
    return dict(x=train,y=y,P=vocab['prototypes'],S=S,Q=Q,ids=ids), target, vocab, evaluation


@torch.no_grad()
def features(encoder, x):
    return torch.cat([encoder(xx) for xx in x.split(4)])


def sample(d, split, rng):
    ids = d[split]
    y = d['y'][ids].numpy()
    labels = rng.choice(np.unique(y), size=min(4,len(ids)))
    return [int(rng.choice(ids[y==k])) for k in labels]


def refit(encoder, W, tasks, q):
    masses = [.5/(len(tasks)-1)]*(len(tasks)-1) + [.5]
    batches = [(features(encoder,d['x']), d['P'], t, m*class_weights(d['y']))
               for d,t,m in zip(tasks,q,masses)]
    return calibrate(W,batches).detach()


def fit_arrival(arm, base, encoder, expert, step, memory, W, d, target, rng, out):
    active_penalties, drift, updates = [], {}, 0
    merged = RecursiveMerge(base,encoder,expert,step)
    opt = torch.optim.Adam([merged.theta], lr=.01)
    last = []
    references = {t:scores(features(encoder,v['x']),W,v['P']).detach() for t,v in memory.items()}
    oldW=W
    for epoch in range(12):
        for t in rng.permutation(list(memory)):
            v = memory[t]
            if arm=='uniform':
                continue
            s,u = sample(d,'S',rng),sample(v,'S',rng)
            j,k = sample(d,'Q',rng),sample(v,'Q',rng)
            def support(z,idx,q):
                return (merged(z['x'][idx]),z['P'],q[idx],torch.full((len(idx),),.5/len(idx)))
            wk = oldW
            if arm!='ordinary':
                wk = calibrate(oldW,[support(d,s,target),support(v,u,references[t])])
                if arm=='detach':
                    wk=wk.detach()
            ns = scores(merged(d['x'][j]),wk,d['P'])
            os = scores(merged(v['x'][k]),wk,v['P'])
            loss, change = query_loss(ns,d['y'][j],os,v['y'][k],references[t][k])
            if arm=='ordinary':
                loss=.5*(F.cross_entropy(ns,d['y'][j])+F.cross_entropy(os,v['y'][k]))
            opt.zero_grad()
            loss.backward()
            assert torch.isfinite(merged.theta.grad).all()
            opt.step()
            updates+=1
            active_penalties.append(float(change.detach())>0)
        if epoch>=6:
            last.append(merged.theta.detach().clone())
    coeff=torch.stack(last).mean(0)
    encoder=merged.fold(coeff)
    with torch.no_grad():
        torch.save({'parameter_names':list(merged.old_delta), 'rounds_7_12':torch.stack(last),
                    'mean':coeff}, out/f'coefficients_{step}.pt')
        merged.theta.copy_(coeff)
        torch.testing.assert_close(merged(d['x'][:2]), encoder(d['x'][:2]), atol=2e-5, rtol=1e-4)
    W=refit(encoder,oldW,[*memory.values(),d],[*references.values(),target])
    for t,v in memory.items():
        h=features(encoder,v['x'])
        drift[t]=[float((scores(h,w,v['P'])-references[t]).square().mean().sqrt()) for w in [oldW,W]]
    return encoder, W, active_penalties, drift, updates


def main(arm, order_seed, expert_seed=0):
    cfg = load_yaml()['model']
    config = build_model_config({'model':cfg})
    order = np.random.default_rng(order_seed).permutation(MANIFEST).tolist()
    out = RUN/'stage_b_smoke'/str(order_seed)/arm
    if expert_seed:
        out = RUN/'stage_b_replication'/f'order{order_seed}'/f'seed{expert_seed}'/arm
    bank = BANK.parent.parent/f'seed{expert_seed}'/'experts'
    out.mkdir(exist_ok=True, parents=True)
    base = expert_features(config, BASE)
    memory, vocab, validation, rows = {}, {}, {}, []
    for step, task in enumerate(order, 1):
        start = time.monotonic()
        d,target,vocab[task],validation[task] = arrival(task, 32, expert_seed)
        expert = expert_features(config, bank/task/'best.ckpt')
        rng = np.random.default_rng([order_seed,step])
        active_penalties, drift, updates, coeff = [],{},0,None
        if step==1:
            encoder = expert
            W,_ = cold_start(features(encoder,d['x']),d['P'],target,d['y'])
        else:
            encoder, W, active_penalties, drift, updates = fit_arrival(
                arm, base, encoder, expert, step, memory, W, d, target, rng, out)
        memory[task]=d
        for v in memory.values():
            assert not set(v['S']) & set(v['Q'])
        raw_bytes=sum(v['x'].numel()*v['x'].element_size() for v in memory.values())
        assert raw_bytes<=2**30 and sum(len(v['x']) for v in memory.values())<=1024
        path=out/'student.pt'
        save_student(path,encoder,W,vocab,cfg)
        student=load_student(path)
        results={}
        for t,(x,y,teacher) in validation.items():
            logits=scores(features(encoder,x),W,vocab[t]['prototypes'])
            with torch.no_grad():
                torch.testing.assert_close(student(x[:3],t),logits[:3],atol=2e-5,rtol=1e-4)
            results[t]=dict(bacc=bacc(y,logits.argmax(1)),teacher_bacc=bacc(y,teacher))
        rows.append(dict(arrival=task,eval=results,expert_seed=expert_seed,
                         expert_sha256=sha256(bank/task/'best.ckpt'),seconds=time.monotonic()-start,updates=updates,
                         memory_bytes=raw_bytes,old_score_rmse_before_after=drift,
                         penalty_active_fraction=np.mean(active_penalties) if active_penalties else None))
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(arm,step,task,results,flush=True)


def bacc(y,p):
    return float(balanced_accuracy_score(y,p))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['full','detach','ordinary','uniform'],required=True)
    p.add_argument('--order-seed',type=int,default=17)
    p.add_argument('--expert-seed',type=int,choices=[0,1,2],default=0)
    args=p.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    main(args.arm,args.order_seed,args.expert_seed)
