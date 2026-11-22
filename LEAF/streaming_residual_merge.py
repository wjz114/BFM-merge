"""Sequential expert merging with a fixed-size historical parameter mean.

The history is a parameter statistic, not a cached functional residual. All
coefficients are folded into one ordinary model before inference.
"""
import torch
from torch.func import functional_call


class StreamingMerger:
    def __init__(self, leaf, base, previous, incoming, history_sum, count,
                 active, condition):
        assert condition in ('recursive', 'history')
        self.leaf = leaf.eval().requires_grad_(False)
        self.active = active
        self.condition = condition
        self.base = base
        self.parameter_names = [name for name, _ in leaf.named_parameters()]
        device = next(leaf.parameters()).device
        self.anchor = {name: base[name].to(device) for name in self.parameter_names}
        self.directions = {}
        fixed = set(base) - set(active)
        for state in (previous, incoming):
            assert state.keys() == base.keys()
            assert all(torch.equal(state[n], base[n]) for n in fixed)
        for name in active:
            old = previous[name] - base[name]
            new = incoming[name] - base[name]
            pieces = [old, new]
            if condition == 'history':
                pieces.append((history_sum[name] / count).float() - old)
            self.directions[name] = torch.stack(pieces).to(device)
        initial = torch.zeros(2 if condition == 'recursive' else 3, len(active), device=device)
        initial[0] = count / (count + 1)
        initial[1] = 1 / (count + 1)
        self.coefficients = torch.nn.Parameter(initial)

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = self.anchor.copy()
        for j, name in enumerate(self.active):
            result[name] = self.anchor[name] + torch.tensordot(c[:, j], self.directions[name], dims=1)
        return result

    def logits(self, wrapper, x, parameters=None):
        parameters = self.compose() if parameters is None else parameters
        return functional_call(wrapper, {'leaf.' + k: v for k, v in parameters.items()}, (x,))

    @torch.no_grad()
    def materialize(self, coefficients=None):
        parameters = self.compose(coefficients)
        result = {name: parameters.get(name, value).detach().cpu().clone()
                  for name, value in self.base.items()}
        assert all(torch.isfinite(result[name]).all() for name in self.active)
        return result


def add_history(history_sum, incoming, base, active):
    """Sum original FP32 task vectors in FP64; never sum merged checkpoints."""
    for name in active:
        history_sum[name].add_((incoming[name] - base[name]).double())
