"""Partition initial feature-distillation output gradients by task readout spaces."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    path = root/'seed0/full_model_distillation_targets/run.json'
    cache = json.loads(path.read_text())
    bp = root/'seed0/prototype_head_geometry/contrast_bases.pt'
    bases = torch.load(bp, map_location='cpu', weights_only=True)
    union = bases['union_basis'].double()
    schedule = np.load(cache['plan']['schedule'])
    rows, hashes = {}, {str(p): sha256(p) for p in [path, bp]}
    for index, task in enumerate(cache['datasets']):
        row = cache['records'][task]
        p = Path(row['target_path'])
        assert sha256(p) == row['target_sha256']
        hashes[str(p)] = row['target_sha256']
        data = torch.load(p, map_location='cpu', weights_only=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == index].reshape(-1).astype(np.int64))
        positions = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][positions], ids)
        norm = data['initial_norm'][positions].double()
        weights = torch.tensor(row['class_weights'], dtype=torch.float64)[data['truth'][positions], None]
        gradient = weights*(data['initial_raw'][positions].double()/norm-data['teacher_unit'][positions].double())/norm
        own = bases['own_bases'][task].double()
        go = (gradient@own.T)@own
        gu = (gradient@union.T)@union
        components = {'own_contrast': go, 'other_contrasts_orthogonal_to_own': gu-go, 'all_contrasts_null': gradient-gu}
        total = float(gradient.square().sum())
        energies = {k: float(v.square().sum())/total for k, v in components.items()}
        assert abs(sum(energies.values())-1) < 1e-10
        rows[task] = energies
    means = {k: float(np.mean([v[k] for v in rows.values()])) for k in rows[cache['datasets'][0]]}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'per_task_energy_fractions': rows, 'mean_task_energy_fractions': means,
              'scope': 'Exact fitting multiset and full-pool class weights at frozen initialization. Orthogonal raw-output gradient partition only; scalar loss factors cancel. Global contrast-null directions preserve every task raw-score difference and argmax, but affect normalized confidence. Other-contrast components can affect other readouts on these inputs, not necessarily other-domain samples. Neither fraction proves harmful transfer, parameter-gradient interference or efficacy. No new inference or fitting.'}
    output = root/'analysis/full_distillation_null_partition.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
