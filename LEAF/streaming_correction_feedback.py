"""Compress normalized calibration corrections with arriving expert vectors."""
import math

import torch

from streaming_complement_memory import RANK, compress_rows
from streaming_unit_feedback import update_memory as update_unit_memory


@torch.no_grad()
def update_memory(history, base, previous, incoming, current, count, active, condition):
    assert condition == 'correction_feedback' and count >= 0
    if count <= 1:
        stats = update_unit_memory(history, base, previous, incoming, current,
                                   count, active, 'unit_feedback')
        stats['feedback_source'] = 'initialization'
        return stats
    correction_energy, incoming_energy = 0., 0.
    native_energies = [0.] * 4
    for name in active:
        old = (previous[name] - base[name]).double()
        new = (incoming[name] - base[name]).double()
        calibrated = (current[name] - base[name]).double()
        correction = calibrated - (count * old + new) / (count + 1)
        correction_energy += float(correction.square().sum())
        incoming_energy += float(new.square().sum())
        for j, state in enumerate((base, previous, incoming, current)):
            native_energies[j] += float(state[name].double().square().sum())
    rounding_floor = 8 * torch.finfo(torch.float32).eps * sum(math.sqrt(x) for x in native_energies)
    significant = math.sqrt(correction_energy) > rounding_floor
    scale = math.sqrt(incoming_energy / correction_energy) if significant else 0.
    total = {'dictionary_squared_norm': 0., 'projected_dictionary_squared_norm': 0.,
             'anchor_removed_squared_norm': 0., 'retained_squared_norm': 0.,
             'discarded_eigenvalue_sum': 0., 'max_stored_anchor_cosine_abs': 0.,
             'retained_rank_counts': [0, 0, 0]}
    for name in active:
        old = (previous[name] - base[name]).double()
        new = (incoming[name] - base[name]).double()
        anchor = (current[name] - base[name]).double()
        correction = anchor - (count * old + new) / (count + 1)
        rows = old.shape[0] if old.ndim else 1
        dictionary = torch.cat([
            scale * correction.reshape(rows, 1, -1),
            history[name].reshape(RANK, rows, -1).transpose(0, 1).double(),
            new.reshape(rows, 1, -1)], dim=1)
        kept, stats = compress_rows(dictionary, anchor.reshape(rows, -1), 'joint_memory')
        history[name].copy_(kept.transpose(0, 1).reshape_as(history[name]))
        for key in total:
            if key == 'retained_rank_counts':
                total[key] = [a + b for a, b in zip(total[key], stats[key])]
            elif key == 'max_stored_anchor_cosine_abs':
                total[key] = max(total[key], stats[key])
            else:
                total[key] += stats[key]
    assert all(torch.isfinite(x).all() for x in history.values())
    total.update(state_bytes=sum(x.numel() * x.element_size() for x in history.values()),
                 actual_previous_task_count=count, feedback_source='current_calibration_correction',
                 calibration_correction_squared_norm=correction_energy,
                 incoming_squared_norm=incoming_energy, feedback_global_scale=scale,
                 correction_rounding_floor=rounding_floor, correction_above_rounding_floor=significant)
    return total
