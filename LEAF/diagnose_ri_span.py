"""Measure RI adaptation inside/outside the original tensorwise expert span.

This is parameter geometry, not an accuracy bound or a functional intervention.
No model inference, labels, fitting, or checkpoint export is performed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import ROOT, load_leaf, sha256
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES, group_for_name


def project(matrix, targets):
    gram = matrix @ matrix.T
    rhs = matrix @ targets.T
    values, vectors = torch.linalg.eigh(gram)
    cutoff = max(float(values[-1]), 0.)*1e-10
    keep = values > cutoff
    coefficients = vectors[:, keep] @ ((vectors[:, keep].T @ rhs)/values[keep, None])
    projection = coefficients.T @ matrix
    residual = targets-projection
    stats = {'rank': int(keep.sum()), 'eigenvalue_cutoff': cutoff,
             'normal_equation_relative_residual': float((gram@coefficients-rhs).norm()/rhs.norm().clamp_min(1e-30))}
    return projection, residual, coefficients, stats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha256(Path(__file__)) == plan['code_sha256']
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).eval()
    active = [name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)]
    base = leaf.state_dict()
    originals = {}
    for task, path in source['experts'].items():
        assert sha256(Path(path)) == plan['original_sha256'][task]
        originals[task] = torch.load(path, map_location='cpu', weights_only=True)
    adapted = {}
    for task, expected in plan['adapted_sha256'].items():
        path = Path(plan['experts_dir'])/task/'best.ckpt'
        assert sha256(path) == expected
        adapted[task] = torch.load(path, map_location='cpu', weights_only=True)
    subjects, bank_tasks = list(adapted), list(originals)
    measured = {task: {'total': {key: 0. for key in ['adaptation_energy', 'projection_energy',
                  'residual_energy', 'projection_residual_dot']}, 'blocks': {}} for task in subjects}
    solutions, solvers = [], {}
    max_self_reconstruction = 0.
    for name in active:
        # FP32 subtraction agrees with the unchanged actual merger basis.
        matrix = torch.stack([(originals[t][name]-base[name]).flatten() for t in bank_tasks]).double()
        changes = torch.stack([(adapted[t][name]-originals[t][name]).flatten() for t in subjects]).double()
        projection, residual, coefficients, stats = project(matrix, changes)
        # A source basis vector must reconstruct in its own span.
        _, self_residual, _, _ = project(matrix, matrix[:1])
        self_error = float(self_residual.norm()/matrix[:1].norm().clamp_min(1e-30))
        max_self_reconstruction = max(max_self_reconstruction, self_error)
        assert self_error < 1e-6, (name, self_error)
        solutions.append(coefficients.numpy())
        solvers[name] = stats
        for index, task in enumerate(subjects):
            values = {'adaptation_energy': float(changes[index].square().sum()),
                      'projection_energy': float(projection[index].square().sum()),
                      'residual_energy': float(residual[index].square().sum()),
                      'projection_residual_dot': float((projection[index]*residual[index]).sum())}
            group = measured[task]['blocks'].setdefault(group_for_name(name), {key: 0. for key in values})
            for key, value in values.items():
                measured[task]['total'][key] += value
                group[key] += value
    for task, record in measured.items():
        for row in [record['total'], *record['blocks'].values()]:
            energy = row['adaptation_energy']
            row['outside_fraction'] = row['residual_energy']/energy if energy else None
            assert abs(energy-row['projection_energy']-row['residual_energy']) <= max(energy*1e-7,1e-12)
    np.savez(args.outdir/'projection_coefficients.npz', coefficients=np.stack(solutions),
             tensor_names=np.array(active), original_tasks=np.array(bank_tasks), adapted_tasks=np.array(subjects))
    result = {'records': measured, 'solver_stats': solvers, 'source_expert_order': bank_tasks,
              'adapted_expert_order': subjects, 'active_parameter_count': sum(base[n].numel() for n in active),
              'max_self_reconstruction_relative_error': max_self_reconstruction,
              'scope': 'Euclidean projection of adaptation epsilon=(RI-original), separately for every active tensor, onto all16 original expert deltas. FP64 Gram solve with relative eigenvalue cutoff1e-10. No checkpoints produced.',
              'interpretation': 'A nonzero outside component changes the available merging subspace; its energy is coordinate-dependent and does not prove functional necessity, causal interference reduction, better accuracy or novelty.',
              'seconds': time.time()-started, 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'test_inference_performed': False}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'total_per_expert': {task: row['total'] for task,row in measured.items()},
                      'seconds': result['seconds'], 'max_self_reconstruction_relative_error': max_self_reconstruction}, indent=2),flush=True)


if __name__ == '__main__':
    main()
