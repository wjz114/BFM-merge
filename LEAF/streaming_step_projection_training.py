"""Matched simultaneous-task Adam and two projected actual-step controls."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_bilateral_merge import StreamingMerger
from streaming_step_projection import CONDITIONS, RowGeometry, project_proposal


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device):
    assert condition in CONDITIONS
    merger = StreamingMerger(leaf, base, previous, incoming, None, count, plan['active'], 'old_rows', None)
    c = merger.coefficients
    geometry = RowGeometry(base, previous, incoming, plan['active'], plan['metric_damping'])
    assert geometry.count == c.numel() == 347666
    metric, inverse = geometry.matrices(condition)
    wrappers = {t:PrototypeClassifier(leaf,*data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([c],lr=plan['learning_rate'])
    bs, stage = plan['batch_size'], count+1
    samples = {t:np.random.default_rng([plan['seed'],stage,plan['tasks'].index(t),217]).permutation(
        len(data.train[t][1])) for t in tasks}
    rng = np.random.default_rng([plan['seed'],stage,931])
    schedules, snapshots, rounds, updates = [], [], [], []
    started = time.monotonic()
    for rnd in range(plan['rounds']):
        task_order = [tasks[int(k)] for k in rng.permutation(len(tasks))]
        losses_this_round = []
        for repeat in range(stage):
            gradients, losses = [], []
            for task in task_order:
                ids = samples[task][rnd*bs:(rnd+1)*bs]
                assert len(ids)==bs
                x,y = data.train[task]
                truth = y[ids].to(device)
                logits = merger.logits(wrappers[task],x[ids].to(device))
                loss = (F.cross_entropy(logits,truth,reduction='none')*data.weights[task][truth]).mean()
                gradient, = torch.autograd.grad(loss,c)
                assert torch.isfinite(loss) and torch.isfinite(gradient).all()
                gradients.append(gradient)
                losses.append(float(loss.detach()))
                schedules.append([rnd+1,repeat+1,plan['tasks'].index(task),*ids.tolist()])
            stacked = torch.stack(gradients)
            before = c.detach().cpu().clone()
            optimizer.zero_grad(set_to_none=True)
            c.grad = stacked.mean(0)
            optimizer.step()
            proposal_coefficients = before.double()-c.detach().cpu().double()
            proposal = geometry.collapse(proposal_coefficients)
            full_g = stacked.detach().cpu().double()
            row_g = geometry.row_gradients(full_g)
            full_margins = full_g @ proposal_coefficients
            row_margins = (row_g*proposal).sum((1,2))
            scale = max(float(full_margins.abs().max()),float(row_margins.abs().max()),1e-12)
            chain_error = float((full_margins-row_margins).abs().max())/scale
            assert chain_error<2e-4,chain_error
            if condition=='simultaneous_mean_adam':
                record=dict(projected=False,before_margins=row_margins.tolist(),
                    after_margins=row_margins.tolist(),solver_iterations=0)
            else:
                projected,record = project_proposal(proposal,row_g,metric,inverse)
                correction = geometry.lift(projected-proposal)
                with torch.no_grad():
                    c.sub_(correction.to(device=device,dtype=c.dtype))
                realized = geometry.collapse(before.double()-c.detach().cpu().double())
                record['realized_FP32_margins'] = (row_g*realized).sum((1,2)).tolist()
                record['realized_FP32_to_requested_correction_max_abs_error'] = float((realized-projected).abs().max())
            record.update(task_order=task_order,round=rnd+1,repeat=repeat+1,
                task_losses=losses,chain_rule_relative_error=chain_error)
            updates.append(record)
            losses_this_round.extend(losses)
            del stacked,gradients,gradient,full_g,row_g,logits,loss
        rounds.append(float(np.mean(losses_this_round)))
        if rnd+1 in plan['average_rounds']:
            snapshots.append(c.detach().cpu().clone())
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
    record = dict(updates=len(updates),training_examples=len(schedules)*bs,
        training_forward_batches=len(schedules),backward_calls=len(schedules),
        schedule_hash=array_hash(np.asarray(schedules,dtype=np.int64)),
        round_losses=rounds,export_checks=checks,coefficient_count=mean.numel(),
        training_seconds=time.monotonic()-started,projection_records=updates,
        projected_updates=sum(r['projected'] for r in updates),
        max_chain_rule_relative_error=max(r['chain_rule_relative_error'] for r in updates),
        zero_native_gram_rows=int(geometry.zero_rows.sum()),metric_damping=plan['metric_damping'],
        coefficient_min=float(mean.min()),coefficient_max=float(mean.max()))
    return state,dict(snapshots=torch.stack(snapshots),mean=mean,
                      schedule=torch.tensor(schedules,dtype=torch.int64)),record
