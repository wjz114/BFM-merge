"""One equally weighted calibrated-model contribution per memory update.

The frozen joint-memory updater uses count only for initialization and its
sqrt(count) dictionary multiplier. Passing one removes that multiplier while
preserving the exact compression arithmetic. Calibration uses the real count.
"""
from streaming_complement_memory import RANK, update_memory as update_joint_memory


def update_memory(history, base, previous, incoming, current, count, active, condition):
    assert condition == 'unit_feedback' and count >= 0
    compression_count = 0 if count == 0 else 1
    stats = update_joint_memory(history, base, previous, incoming, current,
                                compression_count, active, 'joint_memory')
    stats['actual_previous_task_count'] = count
    stats['feedback_dictionary_multiplier'] = float(compression_count)
    return stats
