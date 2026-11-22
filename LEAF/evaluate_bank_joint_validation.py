"""Evaluate all three existing Joint banks in the repair protocol's FP32 validation path."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from coverage_support import training_and_validation
from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, sha256,
)
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, expected in plan['source_sha256'].items():
        assert sha256(Path(filename)) == expected, filename
    for key in ['coverage_audit', 'seed0_reference', 'source_fit_metadata']:
        assert sha256(Path(plan[key])) == plan[key+'_sha256'], key
    for bank in plan['banks'].values():
        assert sha256(Path(bank['checkpoint'])) == bank['sha256']
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source = json.loads(Path(plan['source_fit_metadata']).read_text())
    audit = json.loads(Path(plan['coverage_audit']).read_text())['audits']
    reference = json.loads(Path(plan['seed0_reference']).read_text())['evaluations']['joint']['per_task']
    leaf = load_leaf(config, Path(plan['banks']['0']['checkpoint'])).to(device).eval().requires_grad_(False)
    states = {seed: torch.load(bank['checkpoint'], map_location='cpu', weights_only=True)
              for seed, bank in plan['banks'].items()}
    evaluations, data_checks = {seed: {} for seed in states}, {}
    for task in source['datasets']:
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        tx, ty, vx, vy = training_and_validation(task)
        del tx, ty
        assert len(vy) == audit[task]['validation_size']
        assert array_hash(vx) == audit[task]['validation_eeg_hash']
        assert array_hash(vy) == audit[task]['validation_labels_hash']
        loader = make_loader(vx, vy, config, len(labels), plan['batch_size'], False)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        truth = torch.from_numpy(vy)
        for seed, state in states.items():
            leaf.load_state_dict(state, strict=True)
            logits = torch.cat([wrapper(x.to(device)).cpu() for x, _ in loader])
            assert torch.isfinite(logits).all()
            evaluations[seed][task] = classification_metrics(logits, truth)
        assert evaluations['0'][task]['BAcc'] == reference[task]['BAcc'], task
        data_checks[task] = {'validation_size': len(vy), 'validation_hashes_exact': True,
                             'seed0_Joint_BAcc_replay_exact': True}
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {seed: rows[task]['BAcc'] for seed, rows in evaluations.items()}, flush=True)
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'datasets': source['datasets'], 'evaluations': evaluations, 'data_checks': data_checks,
        'macro_bacc': {seed: float(np.mean([row['BAcc'] for row in rows.values()]))
                       for seed, rows in evaluations.items()},
        'seconds': time.time()-started, 'precision': 'FP32,TF32off,eval mode',
        'optimizer_updates': 0, 'test_data_read': False, 'test_inference_performed': False,
        'scope': 'Existing Joint checkpoints evaluated on exposed full16 validation with the repair loader/readouts. Seed0 replays all16 known Joint BAcc values. Bank-specific descriptive reference only; no training, selection, or independent-test evidence.',
    }
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', result['macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
