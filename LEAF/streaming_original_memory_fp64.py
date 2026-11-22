"""Original-only history with the joint-memory arithmetic and initial prefix.

The first expert is held by the unchanged current model with zero auxiliary
memory. At the second arrival, include that first expert once. Thereafter,
only the incoming original expert is appended to the two-slot history.
"""
from streaming_complement_memory import RANK, update_memory as update_joint_memory


def update_memory(history, base, previous, incoming, current, count, active, condition):
    assert condition == 'original_memory'
    # Only the dictionary anchor changes; the calibration merger still uses
    # its actual previous merged model. Zero is represented by base-base.
    dictionary_previous = base if count > 1 else previous
    return update_joint_memory(history, base, dictionary_previous, incoming,
                               current, count, active, 'joint_memory')
