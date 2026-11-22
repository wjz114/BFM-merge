"""Evaluate both prespecified SWUDI-A matrix policies on exposed validation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from coverage_support import training_and_validation
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, digest in plan['source_sha256'].items():
        assert sha256(Path(filename)) == digest, filename
    outdir = Path(plan['outdir'])
    assert not (outdir/'run.json').exists()
    built = json.loads((outdir/'build.json').read_text())
    assert built['plan'] == plan and built['plan_sha256'] == sha256(args.plan)
    assert built['fixed_tensors_exact'] and built['native_state_reload_exact']
    started = time.time()
    device = configure(args.gpu, 0)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source = json.loads(Path(plan['source_fit']).read_text())
    reference = json.loads(Path(plan['initial_reference']).read_text())
    audit = json.loads(Path(plan['coverage_audit']).read_text())['audits']
    checkpoints = {'initial_merger': {'path': plan['initial_checkpoint'], 'sha256': plan['initial_checkpoint_sha256']},
                   **built['checkpoints']}
    for entry in checkpoints.values():
        assert sha256(Path(entry['path'])) == entry['sha256']
    states = {key: torch.load(entry['path'], map_location='cpu', weights_only=True) for key, entry in checkpoints.items()}
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    evaluations = {key: {} for key in states}
    data_checks = {}
    for task in source['datasets']:
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        tx, ty, vx, vy = training_and_validation(task)
        del tx, ty
        assert array_hash(vx) == audit[task]['validation_eeg_hash']
        assert array_hash(vy) == audit[task]['validation_labels_hash']
        loader = make_loader(vx, vy, config, len(labels), 128, False)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        truth = torch.from_numpy(vy)
        for key, state in states.items():
            leaf.load_state_dict(state, strict=True)
            logits = torch.cat([wrapper(x.to(device)).cpu() for x, _ in loader])
            assert torch.isfinite(logits).all()
            evaluations[key][task] = classification_metrics(logits, truth)
        assert evaluations['initial_merger'][task]['BAcc'] == reference['folded_val'][task]['BAcc']
        data_checks[task] = {'validation_size': len(vy), 'validation_hashes_exact': True, 'initial_BAcc_replay_exact': True}
        (outdir/'evaluation_progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {key: values[task]['BAcc'] for key, values in evaluations.items()}, flush=True)
    tasks = source['datasets']
    values = {key: np.array([rows[t]['BAcc'] for t in tasks]) for key, rows in evaluations.items()}
    contrasts = {}
    for treatment, control in [('all_2d', 'initial_merger'), ('affine_only', 'initial_merger'), ('affine_only', 'all_2d')]:
        delta = (values[treatment]-values[control])*100
        deleted = [np.delete(delta, i).mean() for i in range(len(tasks))]
        contrasts[treatment+'_vs_'+control] = {'macro_gain_pp': float(delta.mean()),
            'per_task_gain_pp': dict(zip(tasks, delta.tolist())), 'improved': int((delta > 0).sum()),
            'declined': int((delta < 0).sum()), 'ties': int((delta == 0).sum()),
            'task_deletion_range_pp': [float(min(deleted)), float(max(deleted))]}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'build_sha256': sha256(outdir/'build.json'), 'checkpoints': checkpoints, 'datasets': tasks,
              'evaluations': evaluations, 'macro_bacc': {key: float(v.mean()) for key, v in values.items()},
              'contrasts': contrasts, 'data_checks': data_checks,
              'seconds': {'build': built['seconds'], 'evaluation': time.time()-started},
              'optimizer_updates': 0, 'test_data_read': False, 'test_inference_performed': False,
              'scope': plan['scope']+' All performance is exposed validation; no independent confirmation. Task-deletion ranges are sensitivities, not confidence intervals.'}
    (outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', result['macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
