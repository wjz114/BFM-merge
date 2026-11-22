"""Measure fixed-instruction FiLM ambiguity using checkpoints and text only."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, normalized, prepare_text, sha256
from load_config import build_model_config, load_yaml


W, B = 'qformer.film.weight', 'qformer.film.bias'


def main():
    root = Path('runs/paper_merge_v2')
    started = time.time()
    torch.set_num_threads(4)
    source_dir = root/'seed0/original_fp32_tensor_balanced_ce'
    avg_dir = root/'seed0/original_fp32_tensor_balanced_avg'
    source = json.loads((source_dir/'run.json').read_text())
    alpha = np.load(avg_dir/'best_alpha.npy')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    tasks = source['datasets']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    merged = torch.load(avg_dir/'merged.ckpt', map_location='cpu', weights_only=True)
    instructions, group_for_task, groups = [], {}, []
    for task in tasks:
        labels, text, instruction, prototypes = prepare_text(config, task)
        cache = torch.load(root/f'seed0/ri_cache/{task}.pt', map_location='cpu', weights_only=True)
        meta = cache['metadata']
        assert array_hash(instruction) == meta['instruction_hash']
        assert array_hash(prototypes) == meta['prototypes_hash']
        assert labels == meta['labels'] and text == meta['instruction']
        actual = normalized(instruction, 'actual classifier instruction')
        same = next((i for i,v in enumerate(instructions) if torch.equal(actual, v)), None)
        if same is None:
            groups.append([task]); group_for_task[task] = len(groups)-1
        else:
            group_for_task[task] = group_for_task[tasks[same]]
            groups[group_for_task[task]].append(task)
        instructions.append(actual)
    u = torch.stack(instructions, dim=1).double()
    q = torch.cat([u, torch.ones(1, len(tasks), dtype=torch.float64)])
    uu, su, _ = torch.linalg.svd(u, full_matrices=False)
    rank = int((su > su[0]*1e-10).sum())
    union = uu[:, :rank]@uu[:, :rank].T
    own_targets, records, hashes = [], {}, {}
    base_affine = torch.cat([base[W].double(), base[B].double().unsqueeze(1)], dim=1)
    merged_affine = torch.cat([merged[W].double(), merged[B].double().unsqueeze(1)], dim=1)
    for i, task in enumerate(tasks):
        state = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
        hashes[task] = sha256(Path(source['experts'][task]))
        delta = state[W].double()-base[W].double()
        delta_affine = torch.cat([delta, (state[B].double()-base[B].double()).unsqueeze(1)], dim=1)
        own = u[:, i:i+1]
        own_projected = delta@own@own.T/float(own.square().sum())
        removed = delta-own_projected
        union_removed = delta@(torch.eye(u.shape[0], dtype=torch.float64)-union)
        visible_removed = removed@union
        old_responses = delta@u
        response_change = -removed@u
        aug = q[:, i:i+1]
        aug_projected = delta_affine@aug@aug.T/float(aug.square().sum())
        affine_change = aug_projected-delta_affine
        aw, ab = [float(alpha[i, source['groups'].index(k)]) for k in [W, B]]
        weighted_own_error = (aw*affine_change[:, :-1]@own[:, 0]+ab*affine_change[:, -1]).norm()
        affine = torch.cat([state[W].double(), state[B].double().unsqueeze(1)], dim=1)
        own_targets.append(affine@aug[:, 0])
        off = [j for j in range(len(tasks)) if j!=i]
        norm2 = float(delta.square().sum())
        records[task] = {
            'instruction_group': group_for_task[task], 'weight_delta_norm': float(delta.norm()),
            'own_null_weight_energy_fraction': float(removed.square().sum())/norm2,
            'union_null_weight_energy_fraction': float(union_removed.square().sum())/norm2,
            'own_null_but_union_visible_weight_energy_fraction': float(visible_removed.square().sum())/norm2,
            'orthogonal_energy_check_error': abs(float(removed.square().sum()-union_removed.square().sum()-visible_removed.square().sum()))/norm2,
            'weight_only_own_response_max_error': float(response_change[:, i].abs().max()),
            'weight_only_off_condition_change_relative_to_original_delta_response': float(response_change[:, off].norm()/old_responses[:, off].norm()),
            'weight_only_off_condition_absolute_rms_change': float(response_change[:, off].square().mean().sqrt()),
            'affine_own_response_max_error': float((affine_change@aug).abs().max()),
            'affine_null_energy_fraction': float(affine_change.square().sum()/delta_affine.square().sum()),
            'affine_independently_weighted_own_contribution_error_norm': float(weighted_own_error),
            'original_merger_weight_coefficient': aw, 'original_merger_bias_coefficient': ab,
            'own_expert_film_output_norm': float(own_targets[-1].norm())}
    targets = torch.stack(own_targets, dim=1)
    inverse = torch.linalg.pinv(q, rtol=1e-10)
    fit_affine = merged_affine+(targets-merged_affine@q)@inverse
    residual = targets-fit_affine@q
    duplicate_floor = torch.zeros_like(targets)
    for group in groups:
        ids = [tasks.index(t) for t in group]
        duplicate_floor[:, ids] = targets[:, ids]-targets[:, ids].mean(1, keepdim=True)
    conflicts = [{'tasks': group, 'own_film_output_rms_disagreement': float(duplicate_floor[:, [tasks.index(t) for t in group]].square().mean().sqrt())}
                 for group in groups]
    sq = torch.linalg.svdvals(q)
    qrank = int((sq>sq[0]*1e-10).sum())
    result = {'tasks': tasks, 'instruction_groups': groups, 'instruction_rank': rank,
              'instruction_singular_values': su.tolist(), 'augmented_instruction_rank': qrank,
              'augmented_instruction_singular_values': sq.tolist(),
              'augmented_nonzero_condition_number': float(sq[0]/sq[qrank-1]),
              'per_expert': records, 'same_instruction_conflicts': conflicts,
              'joint_affine_fit': {'original_response_residual_norm': float((targets-merged_affine@q).norm()),
                                   'fitted_response_residual_norm': float(residual.norm()),
                                   'duplicate_instruction_irreducible_residual_norm': float(duplicate_floor.norm()),
                                   'residual_minus_duplicate_floor_max_abs': float((residual-duplicate_floor).abs().max()),
                                   'weight_displacement_from_merged_norm': float((fit_affine-merged_affine).norm()),
                                   'folded_fp32_response_residual_norm': float((fit_affine.float().double()@q-targets).norm())},
              'expert_sha256': hashes, 'initial_checkpoint_sha256': sha256(avg_dir/'merged.ckpt'),
              'code_sha256': sha256(Path(__file__)), 'design_sha256': sha256(root/'inputs/film_identifiability_design.json'),
              'seconds': time.time()-started, 'eeg_data_read': False, 'model_inference_performed': False,
              'interpretation': 'Text/weight identifiability only. Large null energy does not establish relevant interference or merge performance. Directions outside the instruction union are invisible to every current condition. Equality/least-squares editing has prior art; no novelty claim.'}
    output = root/'analysis/film_identifiability_geometry.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez(root/'analysis/film_identifiability_arrays.npz', instructions=u.numpy(), augmented_instructions=q.numpy(),
             own_expert_outputs=targets.numpy(), fitted_affine=fit_affine.numpy(), original_merged_affine=merged_affine.numpy())
    print(json.dumps({'groups': groups, 'rank': rank, 'fit': result['joint_affine_fit'],
                      'own_null_fractions': {t:v['own_null_weight_energy_fraction'] for t,v in records.items()},
                      'off_condition_response_change': {t:v['weight_only_off_condition_change_relative_to_original_delta_response'] for t,v in records.items()}}, indent=2))


if __name__ == '__main__':
    main()
