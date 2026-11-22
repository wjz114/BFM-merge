"""First-dimension residual gates with a matched incoming-direction control."""
import torch

from streaming_residual_merge import StreamingMerger as TensorMerger


class StreamingMerger(TensorMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition):
        assert condition in ('history_rows', 'incoming_rows')
        super().__init__(leaf, base, previous, incoming, history_sum, count, active,
                         'history' if condition == 'history_rows' else 'recursive')
        self.condition = condition
        device = self.coefficients.device
        initial = [self.coefficients[:2].detach().flatten()]
        offset = 2 * len(active)
        self.row_slices = {}
        for name in active:
            rows = base[name].shape[0] if base[name].ndim else 1
            self.row_slices[name] = slice(offset, offset + rows)
            offset += rows
            initial.append(torch.zeros(rows, device=device))
            if condition == 'incoming_rows':
                # Same number of gates, applied to the incoming direction.
                d = self.directions[name]
                self.directions[name] = torch.cat([d, d[1:2]], dim=0)
        self.coefficients = torch.nn.Parameter(torch.cat(initial))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        n = len(self.active)
        scalar = c[:2 * n].view(2, n)
        result = self.anchor.copy()
        for j, name in enumerate(self.active):
            value = self.anchor[name]
            shape = (-1,) + (1,) * (value.ndim - 1) if value.ndim else ()
            gates = c[self.row_slices[name]].view(shape)
            d = self.directions[name]
            result[name] = value + torch.tensordot(scalar[:, j], d[:2], dims=1) + gates * d[2]
        return result
