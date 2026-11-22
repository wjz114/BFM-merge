"""Independent row-level retention and acquisition, with a matched random control."""
import torch

from streaming_residual_merge_v2 import StreamingMerger as IncomingRowMerger
from streaming_subspace_merge import collect_split as collect_reference_statistics


def collect_split(leaf, base, previous, incoming, data, old_tasks, plan, stage, condition, device):
    # Read-only exposure matched to the existing calibrated baseline. These
    # statistics are not used to construct either bilateral direction.
    return collect_reference_statistics(leaf, base, previous, incoming, data, old_tasks,
                                        plan, stage, 'incoming_rows', device)


class StreamingMerger(IncomingRowMerger):
    def __init__(self, leaf, base, previous, incoming, history_sum, count, active, condition, projections):
        assert condition in ('old_rows', 'random_old_rows')
        super().__init__(leaf, base, previous, incoming, history_sum, count, active, 'incoming_rows')
        self.condition = condition
        self.original_count = self.coefficients.numel()
        self.retention_slices, self.retention_directions = {}, {}
        pieces = [self.coefficients.detach()]
        offset = self.original_count
        for j, name in enumerate(active):
            old = self.directions[name][0]
            rows = old.shape[0] if old.ndim else 1
            self.retention_slices[name] = slice(offset, offset + rows)
            offset += rows
            pieces.append(torch.zeros(rows, device=old.device))
            if condition == 'old_rows':
                self.retention_directions[name] = old
            else:
                generator = torch.Generator(device=old.device).manual_seed(17039 + count * 1000 + j)
                direction = torch.randn(old.shape, device=old.device, generator=generator)
                norms = old.reshape(rows, -1).norm(dim=1)
                unit = direction.reshape(rows, -1)
                unit.mul_((norms / unit.norm(dim=1).clamp_min(1e-30))[:, None])
                self.retention_directions[name] = direction
        self.coefficients = torch.nn.Parameter(torch.cat(pieces))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.original_count])
        for name, direction in self.retention_directions.items():
            shape = (-1,) + (1,) * (direction.ndim - 1) if direction.ndim else ()
            result[name] = result[name] + c[self.retention_slices[name]].view(shape) * direction
        return result
