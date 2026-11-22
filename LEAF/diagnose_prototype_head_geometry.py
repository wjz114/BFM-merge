"""Audit final-head drift in own, other-task and globally invisible contrast spaces."""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, normalized, prepare_text, sha256
from load_config import build_model_config, load_yaml


def row_basis(matrix):
    _, singular, rows = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > singular[0]*1e-10
    return rows[keep], singular


def projected(rows, matrix):
    return rows.T@(rows@matrix)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(filename)) == expected, filename
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    contrasts, bases, singulars, labels_by_task = {}, {}, {}, {}
    for task in source['datasets']:
        labels, text, _, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        # Reproduce the classifier buffer's additional normalization.
        prototypes = normalized(prototypes, 'prototypes').double()
        contrasts[task] = prototypes-prototypes.mean(0, keepdim=True)
        bases[task], singulars[task] = row_basis(contrasts[task])
        assert len(bases[task]) == len(labels)-1
        labels_by_task[task] = labels
    union, union_singular = row_basis(torch.cat(list(contrasts.values())))
    names = ['qformer.output_embed.2.weight', 'qformer.output_embed.2.bias']
    records = {}
    for task in source['datasets']:
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        expert = torch.load(path, map_location='cpu', weights_only=True)
        own = bases[task]
        nesting_error = float((own-projected(union, own.T).T).norm())
        assert nesting_error < 1e-10
        tensors = {}
        for name in names:
            original = base[name].double().reshape(768, -1)
            current = expert[name].double().reshape(768, -1)
            delta = current-original
            own_delta = projected(own, delta)
            union_delta = projected(union, delta)
            cross_delta = union_delta-own_delta
            invisible_delta = delta-union_delta
            total = float(delta.square().sum())
            energies = {key: float(value.square().sum()) for key, value in [
                ('own_contrast', own_delta), ('other_contrasts', cross_delta),
                ('global_null', invisible_delta)]}
            assert abs(sum(energies.values())-total) <= max(1e-12, total*1e-8)
            base_null = original-projected(own, original)
            expert_null = current-projected(own, current)
            delta_null = expert_null-base_null
            candidate = (current-cross_delta).float().double()
            difference = contrasts[task]@(candidate-current)
            tensors[name] = {
                'update_energy': total, 'component_energy': energies,
                'component_energy_fraction': {k: v/total for k, v in energies.items()},
                'own_null_expert_to_base_norm_ratio': float(expert_null.norm()/base_null.norm()),
                'own_null_update_cosine_with_base': float((delta_null*base_null).sum()
                                                          /(delta_null.norm()*base_null.norm())),
                'cross_reset_fp32_own_contrast_matrix_max_abs_error': float(difference.abs().max()),
                'cross_reset_fp32_own_contrast_matrix_relative_error': float(difference.norm()
                    /(contrasts[task]@current).norm().clamp_min(1e-30)),
            }
        records[task] = {'own_contrast_rank': len(own), 'labels': labels_by_task[task],
                          'singular_values': singulars[task].tolist(),
                          'own_basis_in_union_error': nesting_error, 'tensors': tensors}
    torch.save({'own_bases': bases, 'union_basis': union, 'contrasts': contrasts}, args.outdir/'contrast_bases.pt')
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'union_contrast_rank': len(union), 'union_singular_values': union_singular.tolist(),
        'records': records, 'basis_path': str(args.outdir/'contrast_bases.pt'),
        'basis_sha256': sha256(args.outdir/'contrast_bases.pt'), 'seconds': time.time()-started,
        'model_inference_performed': False, 'EEG_data_read': False, 'optimizer_updates': 0,
        'test_data_read': False,
        'scope': 'Final linear head and fixed normalized text prototypes only. Own centered prototype rows determine within-task class-score differences before output normalization. Resetting expert-minus-base drift in union-minus-own contrast directions preserves own unnormalized class differences in exact arithmetic; positive norm rescaling preserves argmax, not necessarily probabilities/CE. Matrix checks quantify FP32 error, not native prediction retention. No new checkpoint, accuracy result, novelty or generalization claim.',
    }
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed prototype-head geometry, union rank', len(union), flush=True)


if __name__ == '__main__':
    main()
