"""Paired Stage-A probes on frozen EEG features. Select ridge on calibration only."""
import csv
import json
from itertools import product

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.nn import functional as F

from continual_readout import center, class_weights, cold_start, scores, targets
from continual_stage_a import ROOT, TASKS, BUDGETS, support_indices

RUN = ROOT / 'runs/continual_v12'
RIDGES = [10. ** i for i in range(-8, -1)]


def ridge_predictions(h, eval_h, P, q, y):
    # Dual ridge in the EEG sample Gram and the centered class Gram eigenbases.
    # For each class eigenvalue v the effective penalty is lambda*C/v.
    A = P - P.mean(0)
    v, V = torch.linalg.eigh(A @ A.T)
    w = class_weights(y).to(h).sqrt()[:, None]
    X = w * h
    e, U = torch.linalg.eigh(X @ X.T)
    left, right = eval_h @ X.T @ U, U.T @ (w * q) @ V
    for ridge in RIDGES:
        gain = v.clamp_min(0) / (e.clamp_min(0)[:, None]*v.clamp_min(0) + ridge*len(P))
        yield ridge, (left @ (gain * right) @ V.T).argmax(1)


def bacc(y, pred):
    return float(balanced_accuracy_score(y, pred))


def main():
    rows = []
    for task, bins in product(TASKS, (1, 2)):
        folder = RUN / ('stage_a' if bins == 1 else 'stage_a_halves')
        data = torch.load(folder / f'{task}_features.pt', weights_only=True)
        proto = torch.load(RUN/'stage_a/mpnet_descriptions'/f'{task}.pt', weights_only=True)
        P = proto['prototypes'].double()
        tr, val = data['train'], data['validation']
        for n in BUDGETS:
            h, y = tr['h'][:n].double(), tr['y'][:n]
            sup = support_indices(tr['ids'][:n], y.numpy())
            qry = np.setdiff1d(np.arange(n), sup)
            q, _ = targets(tr['logits'][:n].double(), y, sup)
            labels = center(F.one_hot(y, len(P)).double())
            for head, p in [('text', P), ('direct', torch.eye(len(P), dtype=h.dtype))]:
                for target, z in [('mixed', q), ('label', labels)]:
                    tune = ridge_predictions(h[sup], h[qry], p, z[sup], y[sup])
                    final = ridge_predictions(h, torch.cat([h, val['h'].double()]), p, z, y)
                    for (ridge, query_pred), (_, pred) in zip(tune, final, strict=True):
                        rows.append(dict(task=task, bins=bins, n=n, head=head, target=target,
                                         ridge=ridge, selection_bacc=bacc(y[qry],query_pred),
                                         train_bacc=bacc(y,pred[:n]), val_bacc=bacc(val['y'],pred[n:]),
                                         agreement=(pred[n:]==val['logits'].argmax(1)).double().mean().item()))
    for row in rows:
        # Tie-break toward greater regularization.
        peers = [r for r in rows if all(r[k]==row[k] for k in ('task','bins','n','head','target'))]
        row['selected'] = row is max(peers, key=lambda r:(r['selection_bacc'],r['ridge']))
    # Separate ridge selection from the original 20-step finite solve.
    for r in rows:
        if not (r['selected'] and r['head']=='text' and r['target']=='mixed'):
            continue
        task, n = r['task'], r['n']
        directory = RUN/('stage_a' if r['bins']==1 else 'stage_a_halves')
        data = torch.load(directory/f'{task}_features.pt', weights_only=True)
        tr,val = data['train'],data['validation']
        P = torch.load(RUN/'stage_a/mpnet_descriptions'/f'{task}.pt', weights_only=True)['prototypes']
        y=tr['y'][:n]
        q,_=targets(tr['logits'][:n],y,support_indices(tr['ids'][:n],y.numpy()))
        W,trace=cold_start(tr['h'][:n],P,q,y,r['ridge'])
        r['selected_cg20_val_bacc']=bacc(val['y'],scores(val['h'],W,P).argmax(1))
        r['selected_cg20_relative_residual']=trace[-1]/trace[0]
    with (RUN/'stage_a_readout_diagnostic.csv').open('w') as f:
        writer=csv.DictWriter(f, dict.fromkeys(k for row in rows for k in row));writer.writeheader();writer.writerows(rows)
    for r in rows:
        if r['n']==512 and r['bins']==1 and (r['selected'] or r['ridge']==.001):
            print(json.dumps(r),flush=True)


if __name__=='__main__':
    main()
