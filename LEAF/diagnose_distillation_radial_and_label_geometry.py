"""Readout-level initial distillation geometry on the exact fitting multiset."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256
from distillation_control_support import auxiliary, readout


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    cp = root/'seed0/full_model_distillation_targets/run.json'
    bp = root/'seed0/prototype_head_geometry/contrast_bases.pt'
    sp = root/'analysis/full_model_distillation_scales.json'
    cache, scales = [json.loads(p.read_text()) for p in [cp, sp]]
    assert scales['target_run_sha256'] == sha256(cp)
    basis = torch.load(bp, map_location='cpu', weights_only=True)
    schedule = np.load(cache['plan']['schedule'])
    hashes = {str(p): sha256(p) for p in [cp, bp, sp]}
    rows = {}
    for ti, task in enumerate(cache['datasets']):
        record = cache['records'][task]
        path = Path(record['target_path'])
        assert sha256(path) == record['target_sha256']
        hashes[str(path)] = record['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64))
        pos = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][pos], ids)
        z = data['initial_raw'][pos].double().requires_grad_(True)
        unit = F.normalize(z.detach(), dim=-1)
        y = data['truth'][pos]
        w = torch.tensor(record['class_weights'], dtype=torch.float64)[y]
        teacher = {k: data[k][pos].double() for k in ['initial_norm', 'teacher_unit', 'teacher_logits']}
        logits = readout(z, data['prototypes'].double())
        gc, = torch.autograd.grad((w*F.cross_entropy(logits, y, reduction='none')).sum(), z, retain_graph=True)
        teacher_correct = teacher['teacher_logits'].argmax(-1) == y
        initial_correct = logits.argmax(-1) == y
        masks = {'teacher_correct': teacher_correct, 'teacher_wrong': ~teacher_correct,
                 'teacher_wrong_initial_correct': ~teacher_correct & initial_correct,
                 'teacher_correct_initial_wrong': teacher_correct & ~initial_correct}
        row = {'weighted_teacher_correct_fraction': float(w[teacher_correct].sum()/w.sum()),
               'weighted_initial_correct_fraction': float(w[initial_correct].sum()/w.sum()),
               'conditions': {}}
        for condition in ['ce_teacher_kl', 'ce_full_feature', 'ce_prototype_contrast']:
            aux = auxiliary(logits, z, teacher, condition, basis['own_bases'][task].double())
            ga, = torch.autograd.grad((w*aux).sum(), z, retain_graph=True)
            ga = ga*scales['per_task'][task][condition]['scale']
            e = ga.square().sum(-1)
            radial = (ga*unit).sum(-1)
            dot = (gc*ga).sum(-1)
            conflict = dot < 0
            total = float(e.sum())
            assert total > 0
            radial_fraction = float(radial.square().sum()/total)
            if condition == 'ce_teacher_kl':
                assert radial_fraction < 1e-20
            condition_row = {'radial_output_gradient_energy_fraction': radial_fraction,
                             'weighted_ce_conflict_fraction': float(w[conflict].sum()/w.sum()),
                             'aux_gradient_energy_on_ce_conflict_fraction': float(e[conflict].sum()/total),
                             'flattened_cosine_with_ce': float((gc*ga).sum()/(gc.norm()*ga.norm())),
                             'label_groups': {}}
            for name, mask in masks.items():
                mass = float(w[mask].sum())
                condition_row['label_groups'][name] = {
                    'weighted_mass_fraction': mass/float(w.sum()),
                    'aux_gradient_energy_fraction': float(e[mask].sum()/total),
                    'weighted_ce_conflict_fraction_within_group': float(w[mask & conflict].sum())/mass if mass else None,
                }
            row['conditions'][condition] = condition_row
        rows[task] = row
    macro = {c: {k: float(np.mean([rows[t]['conditions'][c][k] for t in rows]))
                 for k in rows[cache['datasets'][0]]['conditions'][c] if k != 'label_groups'}
             for c in ['ce_teacher_kl', 'ce_full_feature', 'ce_prototype_contrast']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'per_task': rows, 'macro': macro, 'seconds': time.time()-started,
              'model_inference_performed': False, 'fitting_performed': False,
              'validation_used_in_statistics': False, 'test_data_read': False,
              'scope': 'Exact existing full-training multiset at initialization. Class-weighted per-example raw-output gradients, with frozen auxiliary scales. Radial directions are locally invisible to the cosine readout; their shared-parameter updates need not be radial on other examples. Negative per-example CE/auxiliary dot products indicate local loss disagreement, not harmful generalization or full-network task conflict. This is posthoc diagnosis of a frozen objective pilot, with no hyperparameter selection.'}
    path = root/'analysis/full_distillation_radial_and_label_geometry.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
