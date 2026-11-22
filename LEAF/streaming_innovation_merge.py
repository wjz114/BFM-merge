"""Row-wise merging along incoming-minus-current innovations.

Changing this direction changes the row-constrained model family: each new
row contribution also removes the corresponding previous-model contribution.
The unprojected variant has exactly the incoming-row baseline's coefficient
count. This is a candidate operator, not a claim of newness over prior art.
"""
import torch

from streaming_subspace_merge import StreamingMerger as SplitMerger
from streaming_subspace_merge import collect_split as collect_absolute_split


CONDITIONS = {'innovation_rows': 'incoming_rows',
              'innovation_split': 'activation_split',
              'innovation_random': 'random_split'}


def collect_split(leaf, base, previous, incoming, data, old_tasks, plan, stage, condition, device):
    # The reference argument is used only to form the projected difference.
    projections, split, audit = collect_absolute_split(
        leaf, previous, previous, incoming, data, old_tasks, plan, stage, CONDITIONS[condition], device)
    audit['direction_reference'] = 'incoming minus current merged model'
    return projections, split, audit


class StreamingMerger(SplitMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition, projections):
        super().__init__(leaf, base, previous, incoming, history_sum, count, active, CONDITIONS[condition], projections)
        self.condition = condition
        with torch.no_grad():
            for name in active:
                innovation = (incoming[name] - previous[name]).to(self.coefficients.device)
                self.directions[name][1].copy_(innovation)
                self.directions[name][2].copy_(innovation)
            # base + old_delta + (incoming-current)/t starts at the same
            # count-weighted interpolation as the previous merger, up to FP32 rounding.
            self.coefficients[:len(active)].fill_(1.)
