"""Descriptive RI task-vector norm and direction changes; no inference or fitting."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, load_leaf, sha256
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES, group_for_name


def describe(sums):
    original, adapted, dot, change = sums
    return {'original_norm': math.sqrt(original), 'adapted_norm': math.sqrt(adapted),
            'norm_ratio': math.sqrt(adapted/original) if original else None,
            'cosine': dot/math.sqrt(original*adapted) if original*adapted else None,
            'adaptation_norm': math.sqrt(change),
            'adaptation_over_original_norm': math.sqrt(change/original) if original else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    plan = json.loads((root/'inputs/ri_adaptation_plan.json').read_text())
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).eval()
    names = [name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)]
    base = leaf.state_dict()
    completed = [task for task in source['datasets'] if (root/'seed0/ri_experts'/task/'run.json').exists()]
    if not args.partial:
        assert len(completed) == len(source['datasets']) == 16
    records = {}
    for task in completed:
        original_path = Path(source['experts'][task])
        adapted_path = root/'seed0/ri_experts'/task/'best.ckpt'
        original = torch.load(original_path, map_location='cpu', weights_only=True)
        adapted = torch.load(adapted_path, map_location='cpu', weights_only=True)
        blocks, tensors, total = {}, {}, [0.0]*4
        for name in names:
            u = original[name].double()-base[name].double()
            v = adapted[name].double()-base[name].double()
            values = [float(u.square().sum()), float(v.square().sum()),
                      float((u*v).sum()), float((v-u).square().sum())]
            tensors[name] = describe(values)
            group = blocks.setdefault(group_for_name(name), [0.0]*4)
            for index, value in enumerate(values):
                group[index] += value
                total[index] += value
        records[task] = {'total': describe(total), 'blocks': {k: describe(v) for k, v in blocks.items()},
                         'tensors': tensors, 'original_sha256': sha256(original_path),
                         'adapted_sha256': sha256(adapted_path)}
    result = {'records': records, 'partial': len(completed) != len(source['datasets']),
              'active_parameter_count': sum(base[name].numel() for name in names),
              'scope': 'Euclidean geometry of active task vectors relative to original base; coordinate-dependent, descriptive, no model forwards or fitting.',
              'interpretation': 'Norm or cosine changes do not establish causal interference reduction or merge improvement. A norm-matched control would be a separate intervention.',
              'code_sha256': sha256(Path(__file__)), 'seconds': time.time()-started}
    suffix = 'progress' if args.partial else 'summary'
    (root/f'analysis/ri_geometry_{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({task: record['total'] for task, record in records.items()}, indent=2))


if __name__ == '__main__':
    main()
