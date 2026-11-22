"""Fixed-budget CE fitting for projected attention-operator residuals."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_operator_residual import OperatorMerger


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device):
    merger = OperatorMerger(leaf, base, previous, incoming, count, plan['active'], condition, plan['seed'])
    c = merger.coefficients
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1,
        plan['tasks'].index(t), 217]).permutation(len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, schedule, losses = [], [], []
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        values = []
        for index in order_rng.permutation(len(tasks)):
            task = tasks[int(index)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            labels = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = merger.logits(wrappers[task], x[ids].to(device))
            loss = (F.cross_entropy(logits, labels, reduction='none') * data.weights[task][labels]).sum() / bs
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(c.grad).all()
            optimizer.step()
            values.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(values)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    checks = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        functional = {t: merger.logits(wrappers[t], data.train[t][0][:bs].to(device), parameters).cpu() for t in tasks}
        leaf.load_state_dict(state, strict=True)
        for t in tasks:
            native = wrappers[t](data.train[t][0][:bs].to(device)).cpu()
            check = compare(functional[t], native, data.train[t][1][:bs])
            assert check['max_abs_logit_error'] <= 1e-5 and check['changed_predictions'] == 0
            checks[t] = check
    gates = mean[merger.native_coefficient_count:]
    record = dict(updates=len(schedule), training_examples=len(schedule) * bs,
        schedule_hash=array_hash(np.asarray(schedule, dtype=np.int64)), round_losses=losses,
        coefficient_count=mean.numel(), operator_gate_count=merger.gate_count,
        coefficient_min=float(mean.min()), coefficient_max=float(mean.max()),
        gate_min=float(gates.min()), gate_max=float(gates.max()), gate_norm=float(gates.norm()),
        export_checks=checks, training_seconds=time.monotonic() - started,
        residual_spectral_norms=merger.residual_singular_values.norm(dim=-1).tolist())
    payload = dict(snapshots=torch.stack(snapshots), mean=mean,
        residual_left=merger.residual_left.cpu(), residual_right=merger.residual_right.cpu())
    return state, payload, record
