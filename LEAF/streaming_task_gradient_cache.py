"""Known gradient-table recipes as controls inside incremental merge calibration.

Changing minibatches, random task permutations and Adam differ from classical
finite-sum SAG/SAGA assumptions. No inherited convergence or novelty claim.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger


CONDITIONS = ('direct_ce', 'table_mean', 'table_difference')


def update_direction(table, index, gradient, condition):
    assert condition in CONDITIONS and gradient.shape == table.shape[1:]
    if condition == 'table_difference':
        direction = gradient - table[index] + table.mean(0)
    table[index].copy_(gradient)
    if condition == 'table_mean':
        direction = table.mean(0)
    elif condition == 'direct_ce':
        direction = gradient
    return direction


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device,
                audit_directions=False):
    assert condition in CONDITIONS
    merger = StreamingMerger(leaf, base, previous, incoming, None, count, plan['active'], 'old_rows', None)
    c = merger.coefficients
    wrappers = {t:PrototypeClassifier(leaf,*data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c],lr=plan['learning_rate'])
    bs, stage = plan['batch_size'], count+1
    samples = {t:np.random.default_rng([plan['seed'],stage,plan['tasks'].index(t),217]).permutation(512)
               for t in tasks}
    rng = np.random.default_rng([plan['seed'],stage,931])
    schedule, initialization_schedule, snapshots, rounds, updates = [], [], [], [], []
    started = time.monotonic()

    def task_gradient(task, ids):
        x,y = data.train[task]
        labels = y[ids].to(device)
        logits = merger.logits(wrappers[task],x[ids].to(device))
        loss = (F.cross_entropy(logits,labels,reduction='none')*data.weights[task][labels]).sum()/bs
        gradient, = torch.autograd.grad(loss,c)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        return gradient, float(loss.detach())

    table, initial_losses = [], []
    for task in tasks:
        ids = samples[task][:bs]
        gradient,loss = task_gradient(task,ids)
        table.append(gradient)
        initial_losses.append(loss)
        initialization_schedule.append([plan['tasks'].index(task),*ids.tolist()])
    table = torch.stack(table)
    initial_table_hash = array_hash(table.cpu().numpy())
    stamps = np.zeros(stage,dtype=np.int64)
    table_hashes, oracle_errors = [], []
    for rnd in range(plan['rounds']):
        losses = []
        for k in rng.permutation(stage):
            index = int(k);task = tasks[index];ids = samples[task][rnd*bs:(rnd+1)*bs]
            gradient,loss = task_gradient(task,ids)
            before = table.detach().cpu().double().clone() if audit_directions else None
            old_age = len(updates)-int(stamps[index])
            direction = update_direction(table,index,gradient,condition)
            assert torch.isfinite(direction).all()
            if audit_directions:
                fresh = gradient.detach().cpu().double()
                if condition == 'direct_ce':
                    expected = fresh
                elif condition == 'table_mean':
                    expected = (sum(before[j] for j in range(stage) if j!=index)+fresh)/stage
                else:
                    # Independent expression in terms of the UPDATED table mean.
                    mean_after = (sum(before[j] for j in range(stage) if j!=index)+fresh)/stage
                    expected = mean_after+(1-1/stage)*(fresh-before[index])
                error = float((expected-direction.detach().cpu().double()).abs().max())
                scale = max(float(expected.abs().max()),1e-12)
                assert error < 1e-6*scale+1e-10,(condition,error,scale)
                oracle_errors.append(error)
            stamps[index] = len(updates)
            optimizer.zero_grad(set_to_none=True);c.grad = direction;optimizer.step()
            updates.append(dict(task=task,round=rnd+1,refreshed_entry_age=old_age,
                table_entry_ages=(len(updates)-stamps).tolist(),
                current_gradient_norm=float(gradient.norm()),used_gradient_norm=float(direction.norm()),
                current_to_used_cosine=float(F.cosine_similarity(gradient[None],direction[None],eps=1e-12)),
                loss=loss))
            schedule.append([plan['tasks'].index(task),*ids.tolist()]);losses.append(loss)
        rounds.append(float(np.mean(losses)))
        table_hashes.append(array_hash(table.cpu().numpy()))
        if rnd+1 in plan['average_rounds']:snapshots.append(c.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        functional = {t:merger.logits(wrappers[t],data.train[t][0][:bs].to(device),parameters).cpu()
                      for t in tasks}
        leaf.load_state_dict(state,strict=True)
        checks = {t:compare(functional[t],wrappers[t](data.train[t][0][:bs].to(device)).cpu(),
                          data.train[t][1][:bs]) for t in tasks}
    assert all(v['max_abs_logit_error']<=1e-5 and v['changed_predictions']==0 for v in checks.values())
    record = dict(updates=len(updates),training_examples=(len(schedule)+stage)*bs,
        update_training_examples=len(schedule)*bs,initialization_training_examples=stage*bs,
        training_forward_batches=len(schedule)+stage,backward_calls=len(schedule)+stage,
        schedule_hash=array_hash(np.asarray(schedule,dtype=np.int64)),
        initialization_schedule_hash=array_hash(np.asarray(initialization_schedule,dtype=np.int64)),
        initial_table_hash=initial_table_hash,round_table_hashes=table_hashes,
        initial_losses=initial_losses,round_losses=rounds,cache_records=updates,
        cache_entries=stage,cache_bytes=table.numel()*table.element_size(),cache_reset_at_arrival=True,
        export_checks=checks,coefficient_count=mean.numel(),training_seconds=time.monotonic()-started,
        direction_oracle_max_abs_error=max(oracle_errors,default=0.),direction_oracle_checked=audit_directions,
        coefficient_min=float(mean.min()),coefficient_max=float(mean.max()))
    return state,dict(snapshots=torch.stack(snapshots),mean=mean,
        schedule=torch.tensor(schedule,dtype=torch.int64),
        initialization_schedule=torch.tensor(initialization_schedule,dtype=torch.int64)),record
