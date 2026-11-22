"""Check anchored covariance reconstruction and its local retention bound."""
import hashlib
import json
from pathlib import Path

import torch

from streaming_anchored_covariance import (compress_anchored, covariance_factor,
                                           decode_history, initialize_memory,
                                           unit_rows, update_memory)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(8471)
    output = Path('runs/streaming_anchored_covariance_v1/algebra.json')
    assert not output.exists()
    records = []
    for width in (1, 2, 3, 9):
        dictionary = torch.randn(20, 4, width, dtype=torch.float64)
        anchor = torch.randn(20, width, dtype=torch.float64)
        anchor[0].zero_()
        dictionary[1].zero_()
        stored, covariance, stats = compress_anchored(dictionary, anchor)
        q = unit_rows(anchor)
        basis = torch.cat([q[:, None], stored.double()], dim=1)
        factor = covariance_factor(covariance, basis)
        projected = dictionary - (dictionary * q[:, None]).sum(2, keepdim=True) * q[:, None]
        singular = torch.linalg.svdvals(projected)
        expected = singular[:, :2].square().sum(1) + (dictionary * q[:, None]).sum(2).square().sum(1)
        actual = factor.square().sum((1, 2))
        error = float(((actual - expected).abs() / dictionary.square().sum((1, 2)).clamp_min(1e-300)).max())
        assert error < 1e-7
        assert torch.isfinite(covariance).all()
        records.append({'width': width, 'max_relative_retained_trace_error_vs_SVD': error, 'stats': stats})

    # Width3 has enough anchor+two-slot capacity to retain all covariance.
    # Repeatedly change the native anchor; verify the original sum survives.
    base = {'w': torch.zeros(2, 3)}
    memory = initialize_memory(base, ['w'])
    expected = torch.zeros(2, 3, 3, dtype=torch.float64)
    previous, maximum_error, maximum_decode_error, cross_norm = None, 0., 0., 0.
    for count in range(8):
        incoming = {'w': torch.randn(2, 3)}
        current = incoming if count == 0 else {'w': torch.randn(2, 3)}
        update_memory(memory, base, previous, incoming, current, count, ['w'], 'anchored_covariance')
        delta = incoming['w'].double()
        expected += delta[:, :, None] @ delta[:, None, :]
        q = unit_rows(current['w'].double())
        basis = torch.cat([q[:, None], memory['basis']['w'].transpose(0, 1).double()], dim=1)
        factor = covariance_factor(memory['covariance']['w'], basis)
        reconstructed = factor.transpose(1, 2) @ factor
        error = float((reconstructed - expected).norm() / expected.norm())
        maximum_error = max(maximum_error, error)
        assert error < 1e-6
        projector = torch.eye(3, dtype=torch.float64)[None] - q[:, :, None] @ q[:, None, :]
        decoded = decode_history(memory)['w'].transpose(0, 1).double()
        left = projector @ reconstructed @ projector
        right = projector @ (decoded.transpose(1, 2) @ decoded) @ projector
        decode_error = float((left - right).norm() / expected.norm())
        maximum_decode_error = max(maximum_decode_error, decode_error)
        assert decode_error < 1e-6
        cross_norm = max(cross_norm, float(memory['covariance']['w'][:, 0, 1:].norm()))
        if count == 0:
            assert torch.count_nonzero(decoded) == 0
        previous = current
    assert cross_norm > .1
    result = {'passed': True, 'random_dictionary_checks': records,
              'max_relative_covariance_error_over8_reanchors': maximum_error,
              'max_relative_decoded_complement_covariance_error': maximum_decode_error,
              'nonzero_anchor_complement_cross_covariance_norm': cross_norm,
              'zero_first_calibration_history': True, 'optimizer_updates': 0, 'datasets_read': False,
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                ['streaming_anchored_covariance.py', 'check_anchored_covariance_algebra.py']},
              'scope': 'Local reconstruction and trace-retention checks; no accuracy or global streaming-optimality claim.'}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'random_dictionary_checks'}, indent=2))


if __name__ == '__main__':
    main()
