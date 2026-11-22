"""Strict balanced-CE training with matrix or diagonal row-plane statistics."""
import time
import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_row_matrix import StreamingMerger, RowMatrixAdam, transport_state


def train_stage(leaf, base, previous, incoming, history, count, data, tasks, plan, condition, device):
    started = time.monotonic()
    merger = StreamingMerger(leaf, base, previous, incoming, count, plan['active'])
    initial_native = merger.materialize()
    assert all(torch.equal(v, initial_native[n]) for n, v in merger.initial_native.items())
    del initial_native
    geometry = merger.geometry
    initial_state, transport = None, None
    if condition == 'matrix_transport' and history is not None:
        initial_state, transport = transport_state(history, merger.basis, merger.scale, plan['active'])
    optimizer = RowMatrixAdam(merger.coefficients, condition, initial_state, lr=plan['learning_rate'])
    initial_optimizer = optimizer.state()
    setup_seconds = time.monotonic() - started
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1, plan['tasks'].index(t), 217]).permutation(len(data.train[t][1])) for t in tasks}
    rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, losses, schedule = [], [], []
    for rnd in range(plan['rounds']):
        round_loss = []
        for k in rng.permutation(len(tasks)):
            task = tasks[int(k)]
            ids = permutations[task][rnd * bs:(rnd + 1) * bs]
            assert len(ids) == bs
            x, y = data.train[task]
            truth = y[ids].to(device)
            optimizer.zero_grad()
            logits = merger.logits(wrappers[task], x[ids].to(device))
            loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(merger.coefficients.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    final_optimizer = optimizer.state()
    payload = {'snapshots': torch.stack(snapshots), 'mean': mean,
               'basis': merger.basis, 'scale': merger.scale,
               'optimizer_initial': initial_optimizer, 'optimizer_final': final_optimizer}
    next_history = {'basis': merger.basis, 'scale': merger.scale, **final_optimizer} if condition == 'matrix_transport' else None
    functional = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        for task in tasks:
            functional[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    del parameters, logits, loss, wrappers, optimizer, merger
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            native = wrapper(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(functional[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5 and checks[task]['changed_predictions'] == 0
    assert final_optimizer['step'] - initial_optimizer['step'] == len(schedule)
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_balanced_ce': losses, 'export_checks': checks,
              'coefficient_count': mean.numel(), 'coefficient_mean_abs': float(mean.abs().mean()),
              'training_and_export_seconds': time.monotonic() - started,
              'setup_seconds': setup_seconds, 'geometry': geometry, 'transport': transport,
              'initial_native_weights_exactly_match_bilateral': True,
              'optimizer_initial_step': initial_optimizer['step'],
              'optimizer_final_step': final_optimizer['step'],
              'optimizer_moment_bytes': sum(final_optimizer[k].numel() * final_optimizer[k].element_size() for k in ('m', 'v'))}
    record['persistent_auxiliary_history_bytes'] = (geometry['basis_bytes'] + geometry['scale_bytes'] + record['optimizer_moment_bytes'] + 8) if next_history is not None else 0
    assert mean.numel() == 346900
    return state, payload, next_history, record
