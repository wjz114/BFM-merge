"""Freeze taskwise auxiliary strength from fitting-stream output gradients only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from distillation_control_support import CONDITIONS, auxiliary, initial_gradient_scales, objective, readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.time()
    torch.set_num_threads(4)
    target_run = json.loads(args.targets.read_text())
    assert not target_run['validation_used_in_statistics']
    geometry = json.loads(args.geometry.read_text())
    assert sha256(Path(geometry['basis_path'])) == geometry['basis_sha256']
    bases = torch.load(geometry['basis_path'], map_location='cpu', weights_only=True)['own_bases']
    schedule = np.load(target_run['plan']['schedule'])
    per_task, checks = {}, {}
    for index, task in enumerate(target_run['datasets']):
        row = target_run['records'][task]
        assert sha256(Path(row['target_path'])) == row['target_sha256']
        data = torch.load(row['target_path'], map_location='cpu', weights_only=True)
        ids = schedule['local_indices'][schedule['task'] == index].reshape(-1).astype(np.int64)
        assert array_hash(ids) == row['stream_indices_hash']
        positions = torch.searchsorted(data['pool_indices'], torch.from_numpy(ids))
        assert torch.equal(data['pool_indices'][positions], torch.from_numpy(ids))
        weights = torch.tensor(row['class_weights'], dtype=torch.float64)
        basis = bases[task].double()
        per_task[task] = initial_gradient_scales(data, positions, weights, basis)
        # Independent analytic derivatives on a real batch check KL, projected and full NMSE.
        idx = positions[:128]
        raw = data['initial_raw'][idx].double().requires_grad_(True)
        prototypes = data['prototypes'].double()
        logits = readout(raw, prototypes)
        truth = data['truth'][idx]
        teacher = {k: data[k][idx].double() for k in ['initial_norm', 'teacher_unit', 'teacher_logits']}
        weighted = weights[truth]/len(idx)
        reference = (F.cross_entropy(logits, truth, reduction='none')*weights[truth]).mean()
        reference_gradient, = torch.autograd.grad(reference, raw, retain_graph=True)
        rows = {}
        for condition in CONDITIONS[1:]:
            zero = objective(logits, raw, truth, weights, teacher, condition, 0., basis)
            zero_gradient, = torch.autograd.grad(zero, raw, retain_graph=True)
            assert torch.equal(zero, reference) and torch.equal(zero_gradient, reference_gradient)
            loss = (weights[truth]*auxiliary(logits, raw, teacher, condition, basis)).mean()
            automatic, = torch.autograd.grad(loss, raw, retain_graph=True)
            if condition == 'ce_teacher_kl':
                probability_error = F.softmax(logits, -1)-F.softmax(teacher['teacher_logits'], -1)
                normalized = F.normalize(raw, dim=-1)
                embedding_grad = probability_error@prototypes
                expected = weighted[:, None]*(embedding_grad-normalized*(embedding_grad*normalized).sum(-1, keepdim=True))/raw.norm(dim=-1, keepdim=True)
            else:
                residual = raw/teacher['initial_norm']-teacher['teacher_unit']
                if condition == 'ce_prototype_contrast':
                    expected = ((residual@basis.T)@basis)/len(basis)
                else:
                    expected = residual/raw.shape[-1]
                expected = 2*weighted[:, None]*expected/teacher['initial_norm']
            relative = float((automatic-expected).norm()/automatic.norm())
            assert relative < 1e-10, (task, condition, relative)
            scale = per_task[task][condition]['scale']
            assert np.isfinite(scale) and scale > 0
            rows[condition] = {'analytic_gradient_relative_error': relative, 'zero_auxiliary_ce_loss_and_gradient_exact': True}
        checks[task] = rows
        print(task, {c: v['scale'] for c, v in per_task[task].items()}, flush=True)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'support_sha256': sha256(Path('distillation_control_support.py')),
              'target_run': str(args.targets), 'target_run_sha256': sha256(args.targets),
              'geometry': str(args.geometry), 'geometry_sha256': sha256(args.geometry),
              'datasets': target_run['datasets'], 'per_task': per_task, 'objective_checks': checks,
              'seconds': time.time()-started, 'validation_used': False, 'test_data_read': False,
              'scope': 'One fixed initial raw-output gradient-RMS match per task/auxiliary objective on the exact fitting multiset, including original class weights and repetitions. No model parameter gradients, adaptive rescaling, validation fitting or hyperparameter sweep. This is a conventional magnitude control, not a GradNorm reproduction or novelty claim.'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
