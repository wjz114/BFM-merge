"""Original-expert mean and calibration-correction propagation rules.

The retained state is an FP64 sum of original active expert parameters.
It never contains calibrated or merged parameters. Mean/correction separation
is not itself claimed as novel; the comparison changes only propagation.
"""
import torch


@torch.no_grad()
def initialize(previous, incoming, active, count, condition, original_sum=None):
    assert count >= 1 and condition in ('recursive', 'retain_correction')
    assert previous.keys() == incoming.keys()
    active = set(active)
    assert all(torch.equal(previous[n], incoming[n]) for n in previous if n not in active)
    if condition == 'retain_correction':
        assert set(original_sum) == active
    result = {n: v.clone() for n, v in previous.items()}
    for name in active:
        old = previous[name].double()
        mixed = (count * old + incoming[name].double()) / (count + 1)
        if condition == 'retain_correction':
            correction = old - original_sum[name] / count
            mixed.add_(correction / (count + 1))
        result[name] = mixed.float()
        assert torch.isfinite(result[name]).all()
    return result


@torch.no_grad()
def add_original(original_sum, incoming, active):
    for name in active:
        if name not in original_sum:
            original_sum[name] = incoming[name].double().clone()
        else:
            original_sum[name].add_(incoming[name].double())


def average_snapshots(snapshots, fixed_state, active):
    """Use exactly the same FP64-mean/FP32-export convention as prior runs."""
    result = {n: v.clone() for n, v in fixed_state.items()}
    for name in active:
        result[name] = torch.stack([s[name].double() for s in snapshots]).mean(0).float()
        assert torch.isfinite(result[name]).all()
    return result
