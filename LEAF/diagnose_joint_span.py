"""Oracle Joint-to-expert-span projection, evaluated only on validation.

A poor Euclidean projection is not an accuracy upper bound. Joint is used
only to diagnose the coefficient family, never as a candidate method input.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import ROOT, PrototypeClassifier, evaluate, load_leaf, make_loader, prepare_text, selection_key, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES, group_for_name


def solve(gram, rhs):
    values, vectors = np.linalg.eigh(gram)
    cutoff = max(float(values[-1]), 0.) * 1e-10
    keep = values > cutoff
    alpha = vectors[:, keep] @ ((vectors[:, keep].T @ rhs) / values[keep])
    residual = gram @ alpha - rhs
    return alpha, {'rank': int(keep.sum()), 'eigenvalue_cutoff': cutoff,
                   'eigenvalues': values.tolist(),
                   'normal_equation_relative_residual': float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'runs/paper_merge_v2/inputs/joint_span_diagnostic_plan.json')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'runs/paper_merge_v2/seed0/joint_span_diagnostic')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    plan = json.loads(args.plan.read_text())
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    model = load_leaf(config, Path(plan['base'])).to(device).eval().requires_grad_(False)
    base = {name: value.detach().clone() for name, value in model.state_dict().items()}
    active = [name for name, _ in model.named_parameters() if not name.startswith(UNUSED_PREFIXES)]
    fixed = [name for name in base if name not in active]
    joint = torch.load(plan['joint_checkpoint'], map_location=device, weights_only=True)
    assert set(joint) == set(base) and all(joint[k].shape == base[k].shape for k in base)
    assert all(torch.equal(joint[name], base[name]) for name in fixed)
    deltas = {name: [] for name in active}
    source_hashes = {'base': sha256(Path(plan['base'])), 'joint': sha256(Path(plan['joint_checkpoint'])),
                     'prior_candidate': sha256(Path(plan['prior_candidate']))}
    for task, checkpoint in plan['expert_paths'].items():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        assert set(state) == set(base) and all(torch.equal(state[k], base[k]) for k in fixed)
        for name in active:
            deltas[name].append(state[name].float() - base[name].float())
        source_hashes[task] = sha256(Path(checkpoint))
    del state
    deltas = {name: torch.stack(values) for name, values in deltas.items()}
    groups = list(dict.fromkeys(group_for_name(name) for name in active))
    k = len(plan['expert_paths'])
    block_gram = {group: np.zeros((k, k), dtype=np.float64) for group in groups}
    block_rhs = {group: np.zeros(k, dtype=np.float64) for group in groups}
    tensor_solutions, tensor_stats = {}, {}
    for name in active:
        matrix = deltas[name].flatten(1).double()
        target = (joint[name].float() - base[name].float()).flatten().double()
        gram = (matrix @ matrix.T).cpu().numpy()
        rhs = (matrix @ target).cpu().numpy()
        alpha, statistics = solve(gram, rhs)
        tensor_solutions[name] = alpha
        tensor_stats[name] = statistics
        group = group_for_name(name)
        block_gram[group] += gram
        block_rhs[group] += rhs
    block_solutions, block_stats = {}, {}
    for group in groups:
        block_solutions[group], block_stats[group] = solve(block_gram[group], block_rhs[group])
    projections, geometry = {}, {}
    for layout, solutions, names, statistics in [
        ('block', block_solutions, groups, block_stats), ('tensor', tensor_solutions, active, tensor_stats)]:
        state = {name: value.cpu().clone() for name, value in base.items()}
        measured = {group: {'target_energy': 0., 'projection_energy': 0., 'residual_energy': 0., 'projection_residual_dot': 0.} for group in groups}
        alpha_table = np.stack([solutions[name] for name in names], axis=1)
        for name in active:
            column = solutions[name if layout == 'tensor' else group_for_name(name)]
            vector = torch.as_tensor(column, dtype=torch.float32, device=device)
            displacement = torch.tensordot(vector, deltas[name], dims=1)
            state[name] = (base[name] + displacement).cpu()
            target = (joint[name] - base[name]).double()
            residual = target - displacement.double()
            row = measured[group_for_name(name)]
            row['target_energy'] += target.square().sum().item()
            row['projection_energy'] += displacement.double().square().sum().item()
            row['residual_energy'] += residual.square().sum().item()
            row['projection_residual_dot'] += (displacement.double() * residual).sum().item()
        assert all(torch.equal(state[name], base[name].cpu()) for name in fixed)
        assert all(torch.isfinite(state[name]).all() for name in active)
        checkpoint = args.outdir / f'oracle_{layout}_projection.ckpt'
        torch.save(state, checkpoint)
        np.save(args.outdir / f'{layout}_alpha.npy', alpha_table)
        total = {key: sum(row[key] for row in measured.values()) for key in next(iter(measured.values()))}
        geometry[layout] = {'coefficient_groups': len(names), 'coefficients': alpha_table.size,
                            'max_abs_coefficient': float(np.abs(alpha_table).max()), 'groups': measured,
                            'total': total, 'captured_energy_fraction': 1 - total['residual_energy'] / total['target_energy'],
                            'solvers': statistics, 'checkpoint_sha256': sha256(checkpoint)}
        projections[layout] = checkpoint
        print(layout, 'captured energy fraction', geometry[layout]['captured_energy_fraction'], flush=True)
    assert geometry['tensor']['total']['residual_energy'] <= geometry['block']['total']['residual_energy'] + 1e-6
    (args.outdir / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    del deltas, joint
    torch.cuda.empty_cache()
    loaders, readouts = {}, {}
    for task in plan['expert_paths']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(task)
        del train_x, train_y, test_x, test_y
        loaders[task] = make_loader(val_x, val_y, config, len(labels), 128, False)
        readouts[task] = instruction.to(device), prototypes.to(device)
    evaluations = {}
    conditions = {'joint': Path(plan['joint_checkpoint']), 'prior_candidate': Path(plan['prior_candidate']),
                  'oracle_block_projection': projections['block'], 'oracle_tensor_projection': projections['tensor']}
    for name, checkpoint in conditions.items():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        metrics = {task: evaluate(PrototypeClassifier(model, *readouts[task]).to(device), loader, device)
                   for task, loader in loaders.items()}
        bacc, loss, _ = selection_key(metrics)
        evaluations[name] = {'macro_bacc': bacc, 'macro_ce': loss, 'per_task': metrics,
                             'checkpoint_sha256': sha256(checkpoint)}
        (args.outdir / 'evaluations.json').write_text(json.dumps(evaluations, indent=2) + '\n')
        print(name, 'validation BAcc', bacc, 'CE', loss, flush=True)
    result = {'diagnostic_only': True, 'joint_assisted': True, 'test_inference_performed': False,
              'not_an_accuracy_upper_bound': True, 'interpretation': plan['interpretation'],
              'protocol': 'Complete original source validation, training instruction, FP32, eval, TF32 off',
              'geometry': geometry, 'evaluations': evaluations, 'source_hashes': source_hashes,
              'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'seconds': time.time() - start, 'fixed_tensors_exact': len(fixed)}
    (args.outdir / 'run.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
