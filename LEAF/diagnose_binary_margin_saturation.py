"""Inspect binary cosine-margin saturation from saved training logits only."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source_path = root/'analysis/input_response_summary.json'
    source = json.loads(source_path.read_text())
    head_path = root/'seed0/head_calibration_features/run.json'
    head = json.loads(head_path.read_text())
    records = {}
    source_hashes = {str(source_path): sha256(source_path), str(head_path): sha256(head_path)}
    for task in source['datasets']:
        shard = source['datasets'].index(task) % 2
        data_path = root/f'seed0/input_response_v3_shard{shard}/{task}.pt'
        data = torch.load(data_path, map_location='cpu', weights_only=True)
        if data['logits']['teacher'].shape[1] != 2:
            continue
        spec = head['records'][task]
        assert sha256(Path(spec['path'])) == spec['sha256']
        cache = torch.load(spec['path'], map_location='cpu', weights_only=True, mmap=True)
        proto = cache['prototypes'].double()
        bound = float((proto[1]-proto[0]).norm())
        task_result = {'maximum_absolute_binary_margin': bound, 'pools': {}}
        n = data['n_original']
        for pool, select in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            task_result['pools'][pool] = {}
            for model, logits in data['logits'].items():
                logits = logits[select].double()
                ratio = (logits[:, 1]-logits[:, 0]).abs()/bound
                assert float(ratio.max()) < 1+1e-5
                slack = 1-ratio
                tangent_fraction = (1-ratio.square()).clamp_min(0)
                task_result['pools'][pool][model] = {
                    'normalized_margin_slack_quantiles_p10_p50_p90': torch.quantile(slack, torch.tensor([.1, .5, .9], dtype=torch.float64)).tolist(),
                    'normalized_head_tangent_squared_quantiles_p10_p50_p90': torch.quantile(tangent_fraction, torch.tensor([.1, .5, .9], dtype=torch.float64)).tolist(),
                    'minimum_raw_slack': float(slack.min()),
                    'rounding_above_geometric_bound_examples': int((slack < 0).sum()),
                    'accuracy_on_selected_rows': float((logits.argmax(-1) == data['truth'][select]).double().mean())}
        records[task] = task_result
        source_hashes[str(data_path)] = sha256(data_path)
        source_hashes[spec['path']] = spec['sha256']
    target = root/'analysis/binary_margin_saturation.json'
    result = {'code_sha256': sha256(Path(__file__)), 'source_sha256': source_hashes, 'records': records,
              'identity': 'For u=h/||h|| and d=p1-p0, margin=u^T d and ||P_(u-perp)d||^2/||d||^2 = 1-(margin/||d||)^2. Input derivatives additionally contain the feature Jacobian and 1/||h||.',
              'scope': 'Post hoc descriptive geometric check on every binary task in the saved training diagnostic. Tiny readout tangent factors can contribute to tiny expert input Jacobians, but this is not a complete causal decomposition of those Jacobians or BAcc. Roundoff can put margins slightly above the analytic bound; report it and clamp only the squared tangent factor. No EEG read, inference or fitting.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target))
    for task, record in records.items():
        for pool, models in record['pools'].items():
            print(task, pool, {m: r['normalized_margin_slack_quantiles_p10_p50_p90'][1] for m, r in models.items()})


if __name__ == '__main__':
    main()
