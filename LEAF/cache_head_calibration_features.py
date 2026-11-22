"""Cache native penultimate features and calibration teacher directions for head repair."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, prototype_logits, sha256,
)
from diagnose_sensor_projection import calibration_and_validation
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == expected, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    source = json.loads(Path(plan['source_fit']).read_text())
    initial_run = json.loads(Path(plan['initial_run']).read_text())
    own_reference = json.loads(Path(plan['own_reference']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    indices = np.load(plan['calibration'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    initial = {n: v.detach().cpu().clone() for n, v in leaf.state_dict().items()}
    head = leaf.qformer.output_embed[2]
    assert isinstance(head, torch.nn.Linear) and head.out_features == 768
    records = {}
    for task in source['datasets']:
        labels, text, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        audit = coverage['audits'][task]
        hashes = {k: array_hash(v) for k, v in [('calibration_eeg', cx), ('calibration_labels', cy),
                                               ('validation_eeg', vx), ('validation_labels', vy)]}
        for key, value in hashes.items():
            assert value == audit[key+'_hash']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        leaf.load_state_dict(initial, strict=True)
        captured = {}

        def hook(module, inputs, output):
            captured['features'] = inputs[0].detach()
            captured['raw'] = output.detach()

        handle = head.register_forward_hook(hook)
        features, raw, logits, reconstruction_errors = [], [], [], []
        try:
            for xx, _ in loader:
                prediction = wrapper(xx.to(device))
                h, z = captured['features'], captured['raw']
                reproduced = F.linear(h, head.weight, head.bias)
                assert torch.equal(z, reproduced)
                rebuilt_logits = prototype_logits(F.normalize(reproduced, dim=-1), wrapper.prototypes)
                error = float((prediction-rebuilt_logits).abs().max())
                assert error <= 1e-6
                assert torch.equal(prediction.argmax(1), rebuilt_logits.argmax(1))
                reconstruction_errors.append(error)
                features.append(h.cpu()); raw.append(z.cpu()); logits.append(prediction.cpu())
            features, raw, logits = torch.cat(features), torch.cat(raw), torch.cat(logits)
            metrics = {s: classification_metrics(logits[ids], truth[ids])
                       for s, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]}
            assert metrics['validation']['BAcc'] == initial_run['folded_val'][task]['BAcc']
            assert metrics['calibration']['BAcc'] == own_reference['evaluations']['initial'][task]['calibration']['BAcc']
            expert_path = Path(source['experts'][task])
            assert sha256(expert_path) == plan['expert_sha256'][task]
            leaf.load_state_dict(torch.load(expert_path, map_location='cpu', weights_only=True), strict=True)
            teacher_raw, teacher_logits = [], []
            for start in range(0, n, plan['batch_size']):
                pred = wrapper(torch.from_numpy(x[start:start+plan['batch_size']]).to(device))
                teacher_raw.append(captured['raw'].cpu())
                teacher_logits.append(pred.cpu())
            teacher_raw, teacher_logits = torch.cat(teacher_raw), torch.cat(teacher_logits)
            teacher_metrics = classification_metrics(teacher_logits, truth[:n])
            assert teacher_metrics['BAcc'] == own_reference['own_checks'][task]['calibration']['original']['BAcc']
        finally:
            handle.remove()
        counts = torch.bincount(truth[:n], minlength=len(labels))
        weights = n/(len(labels)*counts.float())
        assert weights.tolist() == source['ce_class_weights'][task]
        path = args.outdir/f'{task}.pt'
        torch.save({'features': features, 'raw_output': raw, 'initial_logits': logits,
                    'teacher_raw_calibration': teacher_raw, 'teacher_logits_calibration': teacher_logits,
                    'truth': truth, 'n_calibration': n, 'class_weights': weights,
                    'instruction': wrapper.instruction.cpu(), 'prototypes': wrapper.prototypes.cpu()}, path)
        records[task] = {
            'path': str(path), 'sha256': sha256(path), 'feature_shape': list(features.shape),
            'calibration_size': n, 'validation_size': len(truth)-n,
            'data_hashes': hashes, 'initial_metrics': metrics, 'teacher_calibration_metrics': teacher_metrics,
            'native_head_reconstruction_max_abs_logit_error': max(reconstruction_errors),
            'expert_sha256': plan['expert_sha256'][task], 'labels': labels, 'instruction_text': text}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'features', list(features.shape), 'native replay exact', flush=True)
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': source['datasets'], 'records': records,
              'initial_validation_and_calibration_bacc_replayed': True,
              'own_teacher_calibration_bacc_replayed': True,
              'seconds': time.time()-started, 'optimizer_updates': 0,
              'test_data_read': False, 'test_inference_performed': False,
              'scope': 'Native frozen initial-merger penultimate features on original512 calibration and all16 exposed validation. Teacher raw outputs are computed on calibration only. No fitting, model selection or new unique training rows. Cache separates calibration prefixes explicitly; later fitting must use only these prefixes. Ordinary final-head folding must receive native export validation.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed head-feature cache', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
