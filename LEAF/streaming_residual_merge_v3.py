"""Combine row-wise incoming adaptation with tensor-wise history compensation."""
import torch

from streaming_residual_merge_v2 import StreamingMerger as IncomingRowMerger


class StreamingMerger(IncomingRowMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition):
        assert condition == 'history_incoming_rows'
        super().__init__(leaf, base, previous, incoming, history_sum, count, active, 'incoming_rows')
        self.condition = condition
        device = self.coefficients.device
        self.incoming_coefficient_count = self.coefficients.numel()
        self.history_direction = {n: ((history_sum[n] / count).float() - (previous[n] - base[n])).to(device)
                                  for n in active}
        self.coefficients = torch.nn.Parameter(torch.cat([
            self.coefficients.detach(), torch.zeros(len(active), device=device)]))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.incoming_coefficient_count])
        for j, name in enumerate(self.active):
            result[name] = result[name] + c[self.incoming_coefficient_count + j] * self.history_direction[name]
        return result
