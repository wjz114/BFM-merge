"""Transport calibration coordinates over additive original-expert moments."""
import torch

from fisher_moment_merge import MomentMerger


def build_merger(leaf, base, previous, incoming, moments, count, active,
                 condition, projections, previous_coefficients=None):
    assert condition in ('moment_restart', 'moment_transport')
    merger = MomentMerger(leaf, base, previous, incoming, moments, count,
                          active, 'moment_restart', projections)
    if condition == 'moment_transport' and count > 1:
        assert previous_coefficients is not None
        assert previous_coefficients.shape == merger.coefficients.shape
        assert previous_coefficients.dtype == torch.float32
        assert torch.isfinite(previous_coefficients).all()
        with torch.no_grad():
            merger.coefficients.copy_(previous_coefficients.to(merger.coefficients.device))
    elif count == 1:
        assert previous_coefficients is None
    return merger
