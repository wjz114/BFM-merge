"""Add a frozen local-gradient or matched-random basis to residual tensor merging."""
from pathlib import Path

import torch

from coefficient_control_support import ResidualTensorMerger
from d_downstream_qformer_ce import sha256


class AugmentedResidualTensorMerger(ResidualTensorMerger):
    def __init__(self, leaf, source, basis_path, expected_sha256, geometry):
        super().__init__(leaf, source)
        assert sha256(Path(basis_path)) == expected_sha256
        extra = torch.load(basis_path, map_location='cpu', weights_only=True)
        assert set(extra) == set(self.active) == set(geometry)
        device = self.beta.device
        for name in self.active:
            value = extra[name]
            assert value.shape == self.deltas[name].shape and value.dtype == torch.float32
            assert torch.isfinite(value).all()
            rank = geometry[name]['rank']
            assert 0 <= rank <= 16 and torch.count_nonzero(value[rank:]) == 0
            self.deltas[name] = torch.cat([self.deltas[name], value.to(device)], dim=0)
        self.beta = torch.nn.Parameter(torch.zeros(32, len(self.active), device=device))
        self.effective_coefficient_count = source['num_coefficients']+sum(v['rank'] for v in geometry.values())
        self.basis_info = {'path': str(basis_path), 'sha256': expected_sha256,
                           'effective_added_coefficients': self.effective_coefficient_count-source['num_coefficients'],
                           'nominal_added_coefficients': 16*len(self.active)}
