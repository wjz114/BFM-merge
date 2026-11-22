"""Describe the fixed distillation gradients without fitting or new EEG forwards."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256
from distillation_control_support import CONDITIONS, auxiliary, readout


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    paths = {'targets': root/'seed0/full_model_distillation_targets/run.json',
             'features': root/'seed0/head_calibration_features/run.json',
             'scales': root/'analysis/full_model_distillation_scales.json',
             'basis': root/'seed0/prototype_head_geometry/contrast_bases.pt'}
    targets, features, scales = [json.loads(paths[k].read_text()) for k in ['targets', 'features', 'scales']]
    bases = torch.load(paths['basis'], map_location='cpu', weights_only=True)['own_bases']
    assert targets['datasets'] == features['datasets'] == scales['datasets']
    assert scales['target_run_sha256'] == sha256(paths['targets'])
    schedule = np.load(targets['plan']['schedule'])
    hashes = {str(p): sha256(p) for p in paths.values()}
    per_task, head_gradients = {}, {c: [] for c in CONDITIONS}
    for task_index, task in enumerate(targets['datasets']):
        record = targets['records'][task]
        tp, fp = Path(record['target_path']), Path(features['records'][task]['path'])
        assert sha256(tp) == record['target_sha256']
        assert sha256(fp) == features['records'][task]['sha256']
        hashes.update({str(p): sha256(p) for p in [tp, fp]})
        data = torch.load(tp, map_location='cpu', weights_only=True)
        old = torch.load(fp, map_location='cpu', weights_only=True)
        ids = schedule['local_indices'][schedule['task'] == task_index].reshape(-1).astype(np.int64)
        positions = torch.searchsorted(data['pool_indices'], torch.from_numpy(ids))
        assert torch.equal(data['pool_indices'][positions], torch.from_numpy(ids))
        basis = bases[task].double()
        assert (basis@basis.T-torch.eye(len(basis))).abs().max() < 1e-10
        weights = torch.tensor(record['class_weights'], dtype=torch.float64)
        norm = data['initial_norm'][positions].double()
        residual = data['initial_raw'][positions].double()/norm-data['teacher_unit'][positions].double()
        gradient = weights[data['truth'][positions], None]*residual/norm
        relevant = (gradient@basis.T)@basis
        invisible = gradient-relevant
        assert abs(float((relevant*invisible).sum())) < 1e-10*float(gradient.square().sum())
        full_energy = float(gradient.square().sum())
        output_stats = {'fitting_multiset_examples': len(positions),
                        'own_contrast_rank': len(basis),
                        'full_feature_aux_raw_gradient_own_contrast_energy_fraction': float(relevant.square().sum())/full_energy,
                        'full_feature_aux_raw_gradient_own_contrast_null_energy_fraction': float(invisible.square().sum())/full_energy}
        # The last affine head is exactly linear in these original512 inputs.
        n = old['n_calibration']
        raw = old['raw_output'][:n].double().requires_grad_(True)
        truth = old['truth'][:n]
        teacher = {'initial_norm': old['raw_output'][:n].norm(dim=-1, keepdim=True).double(),
                   'teacher_unit': F.normalize(old['teacher_raw_calibration'], dim=-1).double(),
                   'teacher_logits': old['teacher_logits_calibration'].double()}
        logits = readout(raw, old['prototypes'].double())
        assert torch.equal(logits.argmax(-1), old['initial_logits'][:n].argmax(-1))
        x = torch.cat([old['features'][:n].double(), torch.ones((n, 1), dtype=torch.float64)], dim=-1)
        ce = (weights[truth]*F.cross_entropy(logits, truth, reduction='none')).mean()
        ce_raw, = torch.autograd.grad(ce, raw, retain_graph=True)
        ce_head = ce_raw.T@x
        head_gradients['balanced_ce'].append(ce_head.reshape(-1))
        head_stats = {}
        for condition in CONDITIONS[1:]:
            alpha = scales['per_task'][task][condition]['scale']
            loss = alpha*(weights[truth]*auxiliary(logits, raw, teacher, condition, basis)).mean()
            aux_raw, = torch.autograd.grad(loss, raw, retain_graph=True)
            aux_head = aux_raw.T@x
            combined = ce_head+aux_head
            head_gradients[condition].append(combined.reshape(-1))
            head_stats[condition] = {'scaled_aux_to_ce_head_gradient_norm': float(aux_head.norm()/ce_head.norm()),
                                     'head_aux_ce_cosine': float((aux_head*ce_head).sum()/(aux_head.norm()*ce_head.norm())),
                                     'combined_to_ce_head_gradient_norm': float(combined.norm()/ce_head.norm())}
        per_task[task] = {'full_stream_output_gradient': output_stats, 'original512_head_gradients': head_stats}
        print(task, 'own-null gradient fraction', output_stats['full_feature_aux_raw_gradient_own_contrast_null_energy_fraction'], flush=True)
    pairwise = {}
    for condition, gradients in head_gradients.items():
        stacked = torch.stack(gradients)
        unit = F.normalize(stacked, dim=-1)
        cosines = unit@unit.T
        upper = cosines[torch.triu_indices(len(unit), len(unit), offset=1).unbind()]
        pairwise[condition] = {'matrix': cosines.tolist(), 'mean_off_diagonal_cosine': float(upper.mean()),
                               'negative_pairs': int((upper < 0).sum()), 'total_pairs': len(upper),
                               'gradient_norms': stacked.norm(dim=-1).tolist()}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'support_sha256': sha256(Path('distillation_control_support.py')),
              'input_sha256': hashes, 'datasets': targets['datasets'], 'per_task': per_task,
              'head_pairwise_geometry': pairwise, 'seconds': time.time()-started,
              'fitting_performed': False, 'model_inference_performed': False, 'validation_used': False, 'test_data_read': False,
              'scope': 'Initial-output auxiliary energy uses the exact full fitting multiset. Head parameter gradients use cached original512 features, full-pool class weights and fixed full-stream alpha; they do not represent full-network gradients or a full-stream head mean. Own-contrast-null perturbations preserve raw class-score differences and argmax, but can change normalized cosine confidence. Pairwise head-gradient signs and energy fractions are descriptive, not proof of harmful interference or causal attribution. No scales or training choices changed.'}
    output = root/'analysis/full_distillation_gradient_geometry.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({c: {k: v for k, v in row.items() if k not in ['matrix', 'gradient_norms']}
                      for c, row in pairwise.items()}, indent=2))


if __name__ == '__main__':
    main()
