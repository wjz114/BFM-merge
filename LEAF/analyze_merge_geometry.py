"""Measure actual parameter displacement of completed runs, without coefficient-sum proxies."""
import argparse
import json
import math
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, load_leaf, sha256
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES, group_for_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/aligned_merge_v1/seed0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = json.loads((args.root / 'merge_block_mseed0/run.json').read_text())
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(reference['base_checkpoint']))
    base = {name: tensor.detach() for name, tensor in leaf.named_parameters()
            if not name.startswith(UNUSED_PREFIXES)}
    uniform = {name: torch.zeros_like(tensor) for name, tensor in base.items()}
    paths = reference['experts'].values()
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=True)
        for name in uniform:
            uniform[name] += (state[name] - base[name]) / len(paths)
    groups = list(dict.fromkeys(group_for_name(name) for name in base))
    output = {}
    for folder in sorted(args.root.iterdir()):
        if not (folder / 'run.json').exists() or not (folder / 'merged.ckpt').exists():
            continue
        state = torch.load(folder / 'merged.ckpt', map_location='cpu', weights_only=True)
        sums = {group: {'delta2': 0., 'uniform2': 0., 'base2': 0., 'dot_uniform': 0.} for group in groups}
        for name in base:
            delta = (state[name] - base[name]).double()
            ref = uniform[name].double()
            row = sums[group_for_name(name)]
            row['delta2'] += delta.square().sum().item()
            row['uniform2'] += ref.square().sum().item()
            row['base2'] += base[name].double().square().sum().item()
            row['dot_uniform'] += (delta * ref).sum().item()
        measured = {}
        for group, row in sums.items():
            dn, un, bn = (math.sqrt(row[key]) for key in ['delta2', 'uniform2', 'base2'])
            measured[group] = {'delta_l2': dn, 'original_uniform_delta_l2': un,
                               'relative_to_base_l2': dn / bn if bn else None,
                               'cosine_to_original_uniform_delta': row['dot_uniform'] / (dn * un) if dn * un else None}
        output[folder.name] = {
            'checkpoint_sha256': sha256(folder / 'merged.ckpt'), 'groups': measured,
            'backbone_delta_l2': math.sqrt(sum(v['delta2'] for g, v in sums.items() if g.startswith('tower.'))),
            'interface_delta_l2': math.sqrt(sum(v['delta2'] for g, v in sums.items() if g.startswith('qformer.'))),
        }
    destination = args.root / 'analysis/parameter_geometry.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({'base_sha256': reference['base_checkpoint_sha256'],
        'reference': 'mean of original expert task vectors; active parameters only', 'runs': output}, indent=2) + '\n')
    print(f'Measured actual parameter displacements for {len(output)} completed checkpoints: {destination}')


if __name__ == '__main__':
    main()
