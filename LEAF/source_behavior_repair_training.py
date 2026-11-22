"""Shared native correction with source-context or merged-context teacher KL."""
import time
import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_native_complement import StreamingMerger


CONDITIONS = ('plain_ce', 'source_context_kl', 'merged_context_kl')


def train_stage(leaf, base, previous, incoming, count, data, tasks, plan, condition, device, projections=None):
    assert condition in CONDITIONS
    merger = StreamingMerger(leaf, base, previous, incoming, count, plan['active'], 'unprojected_hybrid', projections)
    initial = merger.coefficients.detach().cpu().clone()
    native_parameters = list(merger.residual.values())
    groups = [{'params': native_parameters, 'lr': plan['native_learning_rate']}]
    if condition != 'native_only':
        groups.append({'params': [merger.coefficients], 'lr': plan['learning_rate']})
    trainable = [p for group in groups for p in group['params']]
    optimizer = torch.optim.Adam(groups)
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1, plan['tasks'].index(t), 217]).permutation(
        len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, residual_snapshots, losses, schedule = [], [], [], []
    start = time.monotonic()
    auxiliary_losses = []
    source_parameters = {}
    initial_source_errors = {}
    if condition == 'source_context_kl':
        for task in tasks:
            source_parameters[task] = {n: data.experts[task][n].to(device) for n in merger.parameter_names}
            with torch.no_grad():
                check = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), source_parameters[task]).cpu()
                initial_source_errors[task] = float((check - data.teacher_logits[task][:bs]).abs().max())
                assert initial_source_errors[task] < 1e-5
    for rnd in range(plan['rounds']):
        round_loss = []
        for k in order_rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            truth = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = merger.logits(wrappers[task], x[ids].to(device))
            loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
            auxiliary = None
            if condition != 'plain_ce':
                if condition == 'source_context_kl':
                    params = source_parameters[task].copy()
                    for n in plan['active']:
                        params[n] = params[n] + merger.residual[n]
                    auxiliary_logits = merger.logits(wrappers[task], x[ids].to(device), params)
                else:
                    auxiliary_logits = logits
                target = data.teacher_logits[task][ids].to(device).log_softmax(-1)
                per_example = F.kl_div(auxiliary_logits.log_softmax(-1), target,
                                       reduction='none', log_target=True).sum(-1)
                auxiliary = (per_example * data.weights[task][truth]).mean()
                loss = loss + plan['kl_weight'] * auxiliary
            auxiliary_losses.append(0. if auxiliary is None else float(auxiliary.detach()))
            assert torch.isfinite(loss)
            loss.backward()
            # The established native control uses this finite-gradient check;
            # infinite clipping threshold leaves finite gradients unchanged.
            torch.nn.utils.clip_grad_norm_(trainable, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            merger.constrain_residual()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(merger.coefficients.detach().cpu().clone())
            residual_snapshots.append({n: r.detach().cpu().clone() for n, r in merger.residual.items()})
    assert len(snapshots) == len(plan['average_rounds'])
    mean = torch.stack(snapshots).double().mean(0).float()
    residual_mean = {n: torch.stack([r[n] for r in residual_snapshots]).double().mean(0).float() for n in plan['active']}
    residual_gpu = {n: r.to(device) for n, r in residual_mean.items()}
    residual_audit = merger.residual_audit(residual_gpu)
    if condition == 'complement_hybrid':
        assert residual_audit['max_relative_row_projection'] < 2e-6
    if condition == 'native_only':
        assert all(torch.equal(s, initial) for s in snapshots)
    state = merger.materialize(mean.to(device), residual_gpu)
    functional_probes = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device), residual_gpu)
        for task in tasks:
            functional_probes[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    optimizer_bytes = sum(v.numel() * v.element_size() for s in optimizer.state.values()
                          for v in s.values() if isinstance(v, torch.Tensor))
    basis_audit = merger.basis_audit
    trainable_count = sum(p.numel() for p in trainable)
    del parameters, logits, loss, optimizer, wrappers, merger, residual_gpu, groups, trainable, native_parameters
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            native = wrapper(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(functional_probes[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5 and checks[task]['changed_predictions'] == 0
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_total_loss': losses, 'auxiliary_losses': auxiliary_losses,
              'source_initial_logit_errors': initial_source_errors,
              'auxiliary_student_forward_examples': len(schedule) * bs if condition == 'source_context_kl' else 0,
              'auxiliary_KL_examples': len(schedule) * bs if condition != 'plain_ce' else 0,
              'export_checks': checks,
              'coefficient_count': mean.numel(), 'native_residual_count': 25751552,
              'trainable_parameters': trainable_count, 'training_seconds': time.monotonic() - start,
              'update_kind': 'native_only' if condition == 'native_only' else 'joint_coefficient_and_native',
              'initial_coefficient_hash': array_hash(initial.numpy()),
              'basis_audit': basis_audit, 'native_residual_audit': residual_audit,
              'optimizer_state_bytes': optimizer_bytes, 'native_residual_bytes': 103006208,
              'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    return state, {'initial': initial, 'snapshots': torch.stack(snapshots), 'mean': mean,
                   'native_snapshots': residual_snapshots, 'native_mean': residual_mean}, record
