"""Fold a fixed prototype-contrast head intervention and two controls into LEAF."""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256
from diagnose_prototype_head_geometry import projected


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
    geometry = json.loads(Path(plan['geometry']).read_text())
    source = json.loads(Path(geometry['plan']['source_fit']).read_text())
    average = json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    basis = torch.load(geometry['basis_path'], map_location='cpu', weights_only=True)
    assert sha256(Path(geometry['basis_path'])) == geometry['basis_sha256']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    alpha = torch.tensor(average['best_alpha'], dtype=torch.float64)
    assert alpha.shape == (16, 383) and average['groups'] == source['groups']
    names = ['qformer.output_embed.2.weight', 'qformer.output_embed.2.bias']
    corrections = {kind: {n: torch.zeros_like(initial[n], dtype=torch.float64) for n in names}
                   for kind in ['cross_contrast_restore', 'global_null_restore']}
    reconstructed = {n: base[n].double().clone() for n in names}
    own_patches, checks = {}, {}
    union = basis['union_basis']
    for task_index, task in enumerate(source['datasets']):
        path = Path(source['experts'][task])
        assert sha256(path) == geometry['plan']['expert_sha256'][task]
        expert = torch.load(path, map_location='cpu', weights_only=True)
        own = basis['own_bases'][task]
        own_patches[task], checks[task] = {}, {}
        for name in names:
            current = expert[name].double().reshape(768, -1)
            delta = current-base[name].double().reshape(768, -1)
            within_union = projected(union, delta)
            cross = within_union-projected(own, delta)
            outside = delta-within_union
            weight = alpha[task_index, source['groups'].index(name)]
            reconstructed[name] += weight*delta.reshape(initial[name].shape)
            for kind, removed in [('cross_contrast_restore', cross), ('global_null_restore', outside)]:
                patched = (current-removed).reshape(expert[name].shape).float()
                corrections[kind][name] += weight*(patched.double()-expert[name].double())
                if kind == 'cross_contrast_restore':
                    own_patches[task][name] = patched
                    error = basis['contrasts'][task]@(patched.double().reshape(768, -1)-current)
                    checks[task][name] = {'own_raw_contrast_matrix_max_abs_error': float(error.abs().max()),
                                          'fixed_merge_coefficient': float(weight)}
    original_errors = {n: float((reconstructed[n]-initial[n].double()).abs().max()) for n in names}
    assert max(original_errors.values()) <= 1e-5
    outputs = {}
    for kind in ['cross_contrast_restore', 'global_null_restore', 'base_head']:
        state = dict(initial)
        for name in names:
            state[name] = (base[name].clone() if kind == 'base_head'
                           else (initial[name].double()+corrections[kind][name]).float())
        assert all(torch.equal(state[n], initial[n]) for n in state if n not in names)
        assert all(torch.isfinite(state[n]).all() for n in names)
        output = args.outdir/f'{kind}.ckpt'
        torch.save(state, output)
        outputs[kind] = {'path': str(output), 'sha256': sha256(output),
                         'head_update_norm': {n: float((state[n]-initial[n]).norm()) for n in names}}
    patches = args.outdir/'own_cross_contrast_heads.pt'
    torch.save(own_patches, patches)
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'datasets': source['datasets'], 'outputs': outputs,
        'own_patches_path': str(patches), 'own_patches_sha256': sha256(patches),
        'own_contrast_matrix_checks': checks, 'original_head_recomposition_error': original_errors,
        'all_other_tensors_exact': True, 'seconds': time.time()-started,
        'optimizer_updates': 0, 'EEG_data_read': False, 'model_inference_performed': False,
        'scope': 'Fixed original averaged merger and its per-tensor coefficients. Change only the final projection weight/bias by composing expert head corrections. Cross-contrast restore removes expert-minus-base drift in union-minus-own prototype contrast directions; global-null restore is a prediction-invariance control, and base-head is a generic reset baseline. This reuses the fitted merger; it adds no training/data. Own argmax preservation follows only in exact arithmetic and must be checked with actual EEG/native exports. No measured performance or novelty claim yet.',
    }
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed fixed prototype-head checkpoints', list(outputs), flush=True)


if __name__ == '__main__':
    main()
