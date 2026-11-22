"""Calibrate frozen angular auxiliaries and verify their readout geometry."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from angular_distillation_support import CONDITIONS, auxiliary, initial_gradient_scales, objective, readout
from d_downstream_qformer_ce import sha256


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    cp = root/'seed0/full_model_distillation_targets/run.json'
    bp = root/'seed0/prototype_head_geometry/contrast_bases.pt'
    dp = root/'inputs/angular_distillation_design.json'
    cache = json.loads(cp.read_text())
    assert not cache['validation_used_in_statistics']
    bases = torch.load(bp, map_location='cpu', weights_only=True)['own_bases']
    schedule = np.load(cache['plan']['schedule'])
    per_task, checks = {}, {}
    for ti, task in enumerate(cache['datasets']):
        record = cache['records'][task]
        assert sha256(Path(record['target_path'])) == record['target_sha256']
        data = torch.load(record['target_path'], map_location='cpu', weights_only=True)
        ids = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert array_hash(ids) == record['stream_indices_hash']
        pos = torch.searchsorted(data['pool_indices'], torch.from_numpy(ids))
        assert torch.equal(data['pool_indices'][pos], torch.from_numpy(ids))
        weights = torch.tensor(record['class_weights'], dtype=torch.float64)
        basis = bases[task].double()
        per_task[task] = initial_gradient_scales(data, pos, weights, basis)
        idx = pos[:128]
        teacher = {'teacher_unit': data['teacher_unit'][idx].double()}
        truth = data['truth'][idx]
        prototypes = data['prototypes'].double()
        centered = prototypes-prototypes.mean(0)
        gram_inverse = torch.linalg.pinv(centered@centered.T, hermitian=True, rtol=1e-10)
        row = {}
        for factor in [1., 1.7]:
            raw = (factor*data['initial_raw'][idx].double()).requires_grad_(True)
            unit = F.normalize(raw, dim=-1)
            logits = readout(raw, prototypes)
            reference = (weights[truth]*F.cross_entropy(logits, truth, reduction='none')).mean()
            reference_gradient, = torch.autograd.grad(reference, raw, retain_graph=True)
            for condition in CONDITIONS[1:]:
                zero = objective(logits, raw, truth, weights, teacher, condition, 0., basis)
                zero_gradient, = torch.autograd.grad(zero, raw, retain_graph=True)
                assert torch.equal(zero, reference) and torch.equal(zero_gradient, reference_gradient)
                values = auxiliary(logits, raw, teacher, condition, basis)
                grad, = torch.autograd.grad((weights[truth]*values).mean(), raw, retain_graph=True)
                residual = unit-teacher['teacher_unit']
                if condition == 'ce_angular_prototype_contrast':
                    euclidean = ((residual@basis.T)@basis)/len(basis)
                    centered_logits = residual@centered.T
                    logit_quadratic = ((centered_logits@gram_inverse)*centered_logits).sum(-1)/len(basis)
                    equivalence_error = float((values-logit_quadratic).norm()/values.norm())
                    assert equivalence_error < 2e-6, (task, equivalence_error)
                else:
                    euclidean = residual/raw.shape[-1]
                    equivalence_error = None
                analytic = 2*(weights[truth]/len(truth))[:, None]*(euclidean-unit*(euclidean*unit).sum(-1, keepdim=True))/raw.norm(dim=-1, keepdim=True)
                relative = float((grad-analytic).norm()/grad.norm())
                radial = float((grad*unit).sum(-1).square().sum()/grad.square().sum())
                invariance = float((values-auxiliary(logits, 2.3*raw, teacher, condition, basis)).abs().max())
                assert relative < 1e-10 and radial < 1e-20 and invariance < 1e-12
                row[f'{condition}_raw_scale_{factor}'] = {'analytic_gradient_relative_error': relative,
                    'radial_gradient_energy_fraction': radial, 'scale_invariance_max_abs_error': invariance,
                    'centered_logit_quadratic_relative_error': equivalence_error,
                    'zero_auxiliary_ce_loss_and_gradient_exact': True}
        checks[task] = row
    original = Path('train_distillation_control.py').read_text()
    current = Path('train_angular_distillation.py').read_text()
    assert current == original.replace('from distillation_control_support import CONDITIONS, objective',
                                       'from angular_distillation_support import CONDITIONS, objective')
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'support_sha256': sha256(Path('angular_distillation_support.py')),
              'target_run': str(cp), 'target_run_sha256': sha256(cp),
              'geometry_basis': str(bp), 'geometry_basis_sha256': sha256(bp),
              'design_path': str(dp), 'design_sha256': sha256(dp),
              'datasets': cache['datasets'], 'per_task': per_task, 'objective_checks': checks,
              'trainer_is_import_only_change_from_verified_worker': True,
              'trainer_sha256': sha256(Path('train_angular_distillation.py')),
              'seconds': time.time()-started, 'validation_used': False, 'test_data_read': False,
              'scope': 'Same fixed initial fitting-multiset raw-output gradient-RMS rule. Objective and analytical geometry checks use real cached training rows at initial and scaled output norms. Ordinary CE delegates to the previously verified objective; the training worker has only its support import changed. No parameter-gradient matching, extra target forwards or validation selection.'}
    path = root/'analysis/angular_distillation_scales.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'analytic_gradient_max_relative_error': max(v['analytic_gradient_relative_error'] for t in checks.values() for v in t.values()), 'radial_energy_max': max(v['radial_gradient_energy_fraction'] for t in checks.values() for v in t.values())}, indent=2))


if __name__ == '__main__':
    main()
