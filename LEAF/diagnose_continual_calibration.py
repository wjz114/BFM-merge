"""Hold the first uniform merge fixed and measure convergence of its shared readout."""
import json
import torch
from continual_readout import calibrate, class_weights, cold_start, scores
from pilot_continual_stream import (BANK, BASE, RUN, ROOT, RecursiveMerge, arrival,
                                    bacc, build_model_config, expert_features, features, load_yaml)


def solve(W0, tasks, ridge):
    def image(W):
        grad = ridge * W
        for h,P,q,w in tasks:
            A = P-P.mean(0)
            grad = grad + A.T @ ((w/len(P))[:,None] * scores(h,W,P)).T @ h
        return grad
    rhs=ridge*W0
    for h,P,q,w in tasks:
        A=P-P.mean(0)
        rhs=rhs+A.T @ ((w/len(P))[:,None]*q).T @ h
    W=W0.clone()
    r=rhs-image(W)
    p=r.clone()
    norm0=r.norm()
    for i in range(100):
        Hp=image(p)
        step=r.square().sum()/(p*Hp).sum()
        W=W+step*p
        new=r-step*Hp
        p=new+new.square().sum()/r.square().sum()*p
        r=new
        if r.norm()<=1e-8*norm0:
            break
    return W, i+1, float((rhs-image(W)).norm()/norm0)


def main():
    torch.set_num_threads(2)
    old,new='MI_BCIC_IV2a','EMO_FACED'
    cfg=build_model_config(load_yaml())
    a,qa,_,va=arrival(old,32)
    ea=expert_features(cfg,BANK/old/'best.ckpt')
    h=features(ea,a['x'])
    W0,_=cold_start(h,a['P'],qa,a['y'])
    old_scores=scores(h,W0,a['P'])
    b,qb,_,vb=arrival(new,32)
    m=RecursiveMerge(expert_features(cfg,BASE),ea,expert_features(cfg,BANK/new/'best.ckpt'),2)
    encoder=m.fold(m.theta.detach())
    tasks=[(features(encoder,d['x']),d['P'],q,.5*class_weights(d['y'])) for d,q in [(a,old_scores),(b,qb)]]
    eval_h=[features(encoder,v[0]) for v in [va,vb]]
    three=calibrate(W0,tasks).detach()
    precise,n,resid=solve(W0.double(),[tuple(z.double() for z in task) for task in tasks],.001)
    rows=[]
    for kind,W in [('old',W0),('three_steps',three),('converged',precise.float())]:
        mse=[float((w*(scores(h,W,P)-q).square().mean(-1)).sum()) for h,P,q,w in tasks]
        rows.append(dict(method=kind,objective=sum(mse)+.001*(W-W0).square().sum().item(),
                         weighted_old_mse=mse[0],weighted_new_mse=mse[1],
                         old_bacc=bacc(va[1],scores(eval_h[0],W,a['P']).argmax(1)),
                         new_bacc=bacc(vb[1],scores(eval_h[1],W,b['P']).argmax(1))))
    result=dict(order=[old,new],n_per_task=32,validation_per_task=64,ridge=.001,rows=rows,
                cg_steps=n,relative_residual=resid)
    (RUN/'stage_b_readout_convergence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    main()
