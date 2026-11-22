"""Audit fixed cosine-readout geometry and cached calibration gradients, without fitting."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256


def describe(value):
    return {'mean': float(value.mean()), 'median': float(value.median()),
            'min': float(value.min()), 'max': float(value.max())}


def gradient_statistics(raw, prototypes, truth, weights):
    norms = raw.norm(dim=-1, keepdim=True)
    unit = raw / norms
    logits = unit @ prototypes.T
    error = logits.softmax(-1) - F.one_hot(truth, prototypes.shape[0])
    unit_gradient = error @ prototypes
    tangent_gradient = unit_gradient - (unit_gradient * unit).sum(-1, keepdim=True) * unit
    raw_gradient = tangent_gradient / norms
    correct = logits.argmax(-1) == truth
    energy = (weights[:, None] * raw_gradient).square().sum(-1)
    ce = F.cross_entropy(logits, truth, reduction='none')
    return raw_gradient, logits, {
        'raw_output_norm': describe(norms[:, 0]),
        'balanced_ce': float((weights * ce).mean()),
        'balanced_probability_of_true_class': float((weights * logits.softmax(-1).gather(1, truth[:, None])[:, 0]).mean()),
        'balanced_unit_gradient_rms': float((weights * unit_gradient.square().sum(-1)).mean().sqrt()),
        'balanced_tangent_gradient_rms': float((weights * tangent_gradient.square().sum(-1)).mean().sqrt()),
        'balanced_raw_gradient_rms': float((weights * raw_gradient.square().sum(-1)).mean().sqrt()),
        'weighted_per_example_raw_gradient_energy_correct_fraction': float(energy[correct].sum() / energy.sum()),
        'balanced_bacc': float((weights * correct).mean()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--gradient-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.time()
    torch.set_num_threads(4)
    cache = json.loads(args.feature_cache.read_text())
    gradients = json.loads(args.gradient_cache.read_text())
    assert cache['datasets'] == gradients['datasets']
    records, inputs = {}, {str(p): sha256(p) for p in [args.feature_cache, args.gradient_cache]}
    for task in cache['datasets']:
        c, g = cache['records'][task], gradients['records'][task]
        assert c['data_hashes']['calibration_eeg'] == g['eeg_hash']
        assert c['data_hashes']['calibration_labels'] == g['labels_hash']
        for item in [c, g]:
            assert sha256(Path(item['path'])) == item['sha256']
            inputs[item['path']] = item['sha256']
        data = torch.load(c['path'], map_location='cpu', weights_only=True)
        native_gradient = torch.load(g['path'], map_location='cpu', weights_only=True)
        n = data['n_calibration']
        assert n == 512
        truth = data['truth'][:n].long()
        weights = data['class_weights'][truth].double()
        p = data['prototypes'].double()
        assert float((p.norm(dim=1)-1).abs().max()) < 1e-6
        center = p-p.mean(0)
        singular = torch.linalg.svdvals(center)
        rank = int((singular > singular[0]*1e-8).sum())
        distances = torch.cdist(p, p)
        offdiag = distances[~torch.eye(len(p), dtype=torch.bool)]
        # Cauchy-Schwarz bounds every possible unit-feature pairwise margin.
        # Summing these bounds gives a CE lower bound, generally not attainable.
        lower_bound = torch.logsumexp(-distances, dim=1)
        raw_grad, logits, initial = gradient_statistics(data['raw_output'][:n].double(), p, truth, weights)
        _, teacher_logits, teacher = gradient_statistics(data['teacher_raw_calibration'].double(), p, truth, weights)
        assert float((logits-data['initial_logits'][:n]).abs().max()) < 1e-6
        assert float((teacher_logits-data['teacher_logits_calibration']).abs().max()) < 1e-6
        assert abs(initial['balanced_ce']-g['weighted_ce']) < 1e-6
        x = data['features'][:n].double()
        head_gradients = {
            'qformer.output_embed.2.weight': raw_grad.T @ (weights[:, None]*x)/n,
            'qformer.output_embed.2.bias': (weights[:, None]*raw_grad).mean(0),
        }
        replay = {}
        for name, value in head_gradients.items():
            reference = native_gradient[name].double()
            error = float((value-reference).norm()/reference.norm())
            assert error < 1e-4, (task, name, error)
            replay[name] = error
        initial['full_parameter_gradient_norm'] = math.sqrt(sum(v*v for v in g['component_gradient_norm'].values()))
        initial['component_gradient_norm'] = g['component_gradient_norm']
        initial['head_gradient_norm'] = math.sqrt(sum(float(v.square().sum()) for v in head_gradients.values()))
        records[task] = {
            'labels': c['labels'], 'class_count': len(p), 'contrast_rank': rank,
            'contrast_singular_values': singular[:rank].tolist(),
            'contrast_condition_number': float(singular[0]/singular[rank-1]),
            'contrast_rms_radius': float((center.square().sum()/len(p)).sqrt()),
            'pairwise_prototype_distance': describe(offdiag),
            'pairwise_bound_ce_per_class': lower_bound.tolist(),
            'pairwise_bound_ce_mean': float(lower_bound.mean()),
            'initial': initial, 'own_teacher': teacher, 'head_gradient_replay_relative_error': replay,
        }
        print(task, 'radius', records[task]['contrast_rms_radius'], 'raw gradient RMS', initial['balanced_raw_gradient_rms'],
              'full parameter gradient norm', initial['full_parameter_gradient_norm'], flush=True)
        del data, native_gradient, x
    ranges = {}
    fields = ['contrast_rms_radius', 'initial.balanced_unit_gradient_rms', 'initial.balanced_raw_gradient_rms',
              'initial.full_parameter_gradient_norm', 'initial.head_gradient_norm']
    for field in fields:
        values = {}
        for t, row in records.items():
            value = row
            for key in field.split('.'):
                value = value[key]
            values[t] = value
        ranges[field] = {'minimum_task': min(values, key=values.get), 'maximum_task': max(values, key=values.get),
                         'min': min(values.values()), 'max': max(values.values()),
                         'maximum_to_minimum_ratio': max(values.values())/min(values.values())}
    result = {
        'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': inputs,
        'datasets': cache['datasets'], 'records': records, 'across_task_ranges': ranges,
        'max_head_gradient_replay_relative_error': max(max(v['head_gradient_replay_relative_error'].values()) for v in records.values()),
        'seconds': time.time()-started, 'model_inference_performed': False, 'optimizer_updates': 0,
        'validation_used_in_statistics': False, 'test_data_read': False,
        'scope': 'Only original512 calibration prefixes enter statistics; validation tensors are loaded in existing cache artifacts but unused. '
                 'Analytic FP64 cosine-head gradients are checked against previously computed native FP32 autograd. '
                 'Per-example gradient energy omits vector cancellation; aggregate parameter gradient additionally reflects the entire model Jacobian. '
                 'Prototype CE bound is loose in general and is not an accuracy or optimization bound. '
                 'Task ranges are descriptive; they do not isolate prototype geometry as the cause of gradient differences or merging loss.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(ranges, indent=2))


if __name__ == '__main__':
    main()
