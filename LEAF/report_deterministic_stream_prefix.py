"""Compare the actual saved tensors from two strict training-prefix processes."""
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.set_num_threads(4)
    root = Path('runs/deterministic_stream_prefix_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in plan['sha256'].items():
        assert sha256(Path(path)) == digest
    runs = [json.loads((root / f'trial{i}/run.json').read_text()) for i in range(2)]
    assert all(r['status'] == 'completed' and r['optimizer_updates'] == r['score_VJPs'] == 60 for r in runs)
    assert runs[0]['stages'][0]['first_expert_sha256'] == runs[1]['stages'][0]['first_expert_sha256']
    checks = {}
    files = ['history.pt'] + [f'stage_{stage:02d}/{filename}' for stage in (2, 3)
                             for filename in ('merged.ckpt', 'coefficients.pt', 'responses.pt')]
    for filename in files:
        values = [torch.load(root / f'trial{i}' / filename, map_location='cpu', weights_only=True) for i in range(2)]
        checks[filename] = identical(*values)
    for index in (1, 2):
        a, b = [r['stages'][index] for r in runs]
        for key in ('schedule_hash', 'probe_signs_hash', 'round_balanced_ce'):
            checks[f'stage_{index + 1:02d}/{key}'] = a[key] == b[key]
        checks[f'stage_{index + 1:02d}/auxiliary_ids'] = a['input_split']['selected_ids_hash'] == b['input_split']['selected_ids_hash']
    result = {'unix': time.time(), 'passed': all(checks.values()), 'checks': checks,
              'source_sha256': {str(root / f'trial{i}/run.json'): sha256(root / f'trial{i}/run.json') for i in range(2)},
              'optimizer_updates': 120, 'CE_example_passes': 1920,
              'score_VJPs': 120, 'score_VJP_example_passes': 1920,
              'auxiliary_forward_examples': sum(r['auxiliary_forward_examples'] for r in runs),
              'elapsed_seconds': [r['elapsed_seconds'] for r in runs],
              'peak_reserved_bytes': [r['peak_reserved_bytes'] for r in runs],
              'development_validation_read': False, 'official_test_read': False,
              'scope': 'Independent processes, same GPU and same seed, three-expert full-round prefix. Exact tensor equality supports using strict settings in the next version; it is not full16task, cross-device or cross-version reproducibility evidence.'}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))
    assert result['passed']


if __name__ == '__main__':
    main()
