"""Cache per-task balanced-CE gradients at the fixed merged model, without updating it."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(filename)) == expected, filename
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    source = json.loads(Path(plan['source_fit']).read_text())
    reference = json.loads(Path(plan['calibration_reference']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    indices = np.load(plan['calibration'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    initial = {n: v.detach().cpu().clone() for n, v in leaf.state_dict().items()}
    active = {n: v for n, v in leaf.named_parameters() if not n.startswith(UNUSED_PREFIXES)}
    assert list(active) == source['groups'] and len(active) == 383
    for name, value in leaf.named_parameters():
        value.requires_grad_(name in active)
    records = {}
    for task in source['datasets']:
        task_started = time.time()
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            rows, full_truth = row_map(h, task, 'calibration')
            full_truth = np.asarray(full_truth, dtype=np.int64)
            assert array_hash(full_truth) == coverage['audits'][task]['train_labels_hash']
            selected = [rows[int(i)] for i in indices[task]]
            memory = {key: h[key][:] for key in dict.fromkeys(row[0] for row in selected)}
        x = torch.from_numpy(read_rows(memory, selected))
        truth = torch.from_numpy(full_truth[indices[task]])
        del memory, rows, full_truth
        assert len(truth) == 512
        assert array_hash(x) == coverage['audits'][task]['calibration_eeg_hash']
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        counts = torch.bincount(truth, minlength=len(labels))
        assert counts.tolist() == source['calibration_class_counts'][task]
        weights = (len(truth)/(len(labels)*counts.float())).to(device)
        assert weights.cpu().tolist() == source['ce_class_weights'][task]
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        leaf.zero_grad(set_to_none=True)
        logits, loss_sum = [], 0.
        for start in range(0, len(x), plan['batch_size']):
            target = truth[start:start+plan['batch_size']].to(device)
            prediction = wrapper(x[start:start+plan['batch_size']].to(device))
            # Sum over examples then divide by512: exact empirical class-balanced risk.
            loss = (F.cross_entropy(prediction, target, reduction='none')*weights[target]).sum()/len(x)
            assert torch.isfinite(loss)
            loss.backward()
            loss_sum += float(loss.detach())
            logits.append(prediction.detach().cpu())
        metrics = classification_metrics(torch.cat(logits), truth)
        assert metrics['BAcc'] == reference['evaluations'][task]['initial_merger']['old']['BAcc'], task
        gradients = {n: v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v, device='cpu')
                     for n, v in active.items()}
        assert all(torch.isfinite(v).all() for v in gradients.values())
        components = {component: float(sum(v.double().square().sum() for n, v in gradients.items()
                                          if n.startswith(component+'.')).sqrt())
                      for component in ['tower', 'qformer']}
        assert all(v > 0 for v in components.values())
        path = args.outdir/f'{task}.pt'
        torch.save(gradients, path)
        records[task] = {'path': str(path), 'sha256': sha256(path), 'weighted_ce': loss_sum,
                         'calibration_metrics': metrics, 'component_gradient_norm': components,
                         'no_gradient_parameter_names': [n for n, v in active.items() if v.grad is None],
                         'eeg_hash': array_hash(x), 'labels_hash': array_hash(truth),
                         'instruction_hash': array_hash(instruction), 'prototypes_hash': array_hash(prototypes),
                         'seconds': time.time()-task_started}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'weightedCE', loss_sum, 'gradient norms', components, flush=True)
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in leaf.state_dict().items())
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': source['datasets'], 'parameter_names': list(active), 'records': records,
              'backpropagated_examples': 8192, 'backward_batches': 64, 'optimizer_updates': 0,
              'initial_state_exactly_unchanged': True, 'old_calibration_bacc_replayed': True,
              'validation_inference_performed': False, 'test_data_read': False, 'test_inference_performed': False,
              'seconds': time.time()-started,
              'scope': 'Per-task empirical balancedCE gradients at the fixed initial merger, all original512 rows/task. No candidate fitting, validation selection, Joint assistance or extra unique training rows. These64 backward batches are additional method construction cost.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed local task-gradient cache', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
