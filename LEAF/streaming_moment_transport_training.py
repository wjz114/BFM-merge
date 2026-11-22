"""Strict calibration with fresh or transported original-moment coordinates."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import PrototypeClassifier
from streaming_moment_transport import build_merger

def train_stage(leaf, base, previous, incoming, history, count, data, tasks, plan, condition, device, projections, previous_coefficients=None):
    merger = build_merger(leaf, base, previous, incoming, history, count, plan['active'],
                          condition, projections, previous_coefficients)
    initial = merger.coefficients.detach().cpu().clone()
    wrappers = {t: PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks}
    optimizer = torch.optim.Adam([merger.coefficients], lr=plan['learning_rate'])
    bs = plan['batch_size']
    permutations = {t: np.random.default_rng([plan['seed'], count + 1, plan['tasks'].index(t), 217]).permutation(
        len(data.train[t][1])) for t in tasks}
    order_rng = np.random.default_rng([plan['seed'], count + 1, 931])
    snapshots, losses, schedule = [], [], []
    start = time.monotonic()
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
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(merger.coefficients.grad).all()
            optimizer.step()
            round_loss.append(float(loss.detach()))
            schedule.append([plan['tasks'].index(task), *ids.tolist()])
        losses.append(float(np.mean(round_loss)))
        if rnd + 1 in plan['average_rounds']:
            snapshots.append(merger.coefficients.detach().cpu().clone())
    mean = torch.stack(snapshots).double().mean(0).float()
    state = merger.materialize(mean.to(device))
    # Native export must preserve actual task logits, including historical tasks.
    functional_probes = {}
    with torch.no_grad():
        parameters = merger.compose(mean.to(device))
        for task in tasks:
            functional_probes[task] = merger.logits(wrappers[task], data.train[task][0][:bs].to(device), parameters).cpu()
    del parameters, logits, loss, optimizer, wrappers, merger
    leaf.load_state_dict(state, strict=True)
    checks = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            native = wrapper(data.train[task][0][:bs].to(device)).cpu()
            checks[task] = compare(functional_probes[task], native, data.train[task][1][:bs])
            assert checks[task]['max_abs_logit_error'] <= 1e-5
            assert checks[task]['changed_predictions'] == 0
    record = {'updates': len(schedule), 'training_examples': len(schedule) * bs,
              'schedule_hash': array_hash(np.asarray(schedule, dtype=np.int64)),
              'round_balanced_ce': losses, 'export_checks': checks,
              'coefficient_count': mean.numel(), 'training_seconds': time.monotonic() - start,
              'coefficient_min': float(mean.min()), 'coefficient_max': float(mean.max())}
    record['row_gate_mean_abs'] = float(mean[2 * len(plan['active']):174216].abs().mean())
    if mean.numel() > 174216:
        record['retention_gate_mean_abs'] = float(mean[174216:347666].abs().mean())
    record['initial_coefficient_hash'] = array_hash(initial.numpy())
    record['initialization_from_previous_coefficients'] = condition == 'moment_transport' and count > 1
    assert mean.numel() == plan['per_arm_coefficients'][condition]
    return state, {'initial': initial, 'snapshots': torch.stack(snapshots), 'mean': mean}, record

