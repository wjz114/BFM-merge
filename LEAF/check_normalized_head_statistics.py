"""Check the normalized ridge identity on cached fitting rows before launch."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_normalized_head_statistics import statistics
from d_downstream_qformer_ce import sha256


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    source = root/'seed0/head_calibration_features/run.json'
    cache = json.loads(source.read_text())
    geometry_path = root/'seed0/prototype_head_geometry/contrast_bases.pt'
    geometry = torch.load(geometry_path, map_location='cpu', weights_only=True)
    gen = torch.Generator().manual_seed(20261593)
    records = {}
    for task in ['MI_OpenBMI', 'EMO_FACED']:
        row = cache['records'][task]
        assert sha256(Path(row['path'])) == row['sha256']
        data = torch.load(row['path'], map_location='cpu', weights_only=True)
        n = data['n_calibration']
        h, z, t = (data[k][:n].double() for k in ['features', 'raw_output', 'teacher_raw_calibration'])
        truth, cw = data['truth'][:n], data['class_weights']
        x = torch.cat([h, torch.ones((n, 1), dtype=torch.float64)], dim=1)
        norm = z.norm(dim=-1, keepdim=True)
        residual = norm*F.normalize(t, dim=-1)-z
        delta = torch.randn((768, x.shape[1]), generator=gen, dtype=torch.float64)*.01
        a = geometry['own_bases'][task].double()
        weights = cw[truth].double()
        error = (x@delta.T-residual)@a.T
        direct = (weights*error.square().sum(-1)/norm[:, 0].square()).mean()/len(a)
        st = statistics(h, z, t, truth, cw, normalize=True)
        covariance = delta@st['S']@delta.T-delta@st['F'].T-st['F']@delta.T+st['R']
        reconstructed = ((a.T@a/len(a))*covariance.T).sum()
        relative = float((direct-reconstructed).abs()/direct.abs())
        assert relative < 1e-12, (task, relative)
        raw = z+x@delta.T
        assert torch.equal((raw@a.T).argmax(-1), ((raw/norm)@a.T).argmax(-1))
        # The augmented bias column must be divided by the same per-row norm.
        assert torch.allclose(st['S'][-1, -1], (weights/norm[:, 0].square()).mean(), atol=1e-14, rtol=1e-12)
        records[task] = {'examples': n, 'direct_loss': float(direct),
                         'statistics_loss': float(reconstructed), 'relative_error': relative,
                         'positive_scalar_argmax_identity': True, 'bias_normalization_checked': True}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'worker_sha256': sha256(Path('cache_normalized_head_statistics.py')),
              'source': str(source), 'source_sha256': sha256(source),
              'geometry_sha256': sha256(geometry_path), 'records': records,
              'scope': 'Binary and nine-class original512 fitting rows; direct projected objective versus normalized sufficient statistics. No fitting, native inference, validation statistics or test reads.'}
    path = root/'analysis/normalized_head_statistics_verification.json'
    with path.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
