"""Exploratory transfer-preservation intervention on the fixed RI adaptation.

Protect original expert predictions on calibration-helpful off-domains, or
on a random same-cardinality domain set; restore base on all other domains.
This is a mechanism experiment, not an established novel merging algorithm.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES


def divergence(student, teacher):
    """Per-example KL(teacher || student), T=1, with detached teacher logits."""
    teacher_logp = teacher.detach().log_softmax(-1)
    return (teacher_logp.exp() * (teacher_logp-student.log_softmax(-1))).sum(-1)


def choose_protected(tasks, own, gains, condition, seed):
    other = [task for task in tasks if task != own]
    helpful = [task for task in other if gains[task] > 0]
    if condition == 'helpful':
        return helpful
    if condition == 'random':
        rng = np.random.default_rng([seed, tasks.index(own), 7381])
        selected = set(rng.choice(other, size=len(helpful), replace=False).tolist())
        return [task for task in other if task in selected]
    assert condition == 'ri_reference'
    return []


@torch.no_grad()
def predictions(leaf, readout, x, device, bs=128):
    wrapper = PrototypeClassifier(leaf, *readout).to(device).eval()
    return torch.cat([wrapper(x[i:i+bs].to(device)).cpu()
                      for i in range(0, len(x), bs)])


def configure(gpu, seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    return device


def adapt(plan, cache_dir, outdir, device, expert_names, transfer_plan, condition, smoke=False):
    started = time.time()
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    cache_run = json.loads((cache_dir/'run.json').read_text())
    assert cache_run['plan'] == plan
    assert cache_run['code_sha256'] == plan['code_sha256']
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    tasks = source['datasets']
    assert set(expert_names) <= set(tasks)
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    active_names = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    fixed = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()
             if name not in active_names}
    for name, value in leaf.named_parameters():
        value.requires_grad_(name in active_names)
    readouts, data, caches = {}, {}, {}
    for task in tasks:
        cache = torch.load(cache_dir/f'{task}.pt', map_location='cpu', weights_only=True)
        meta = cache['metadata']
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == meta['labels'] and text == meta['instruction']
        readouts[task] = instruction.to(device), prototypes.to(device)
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(task)
        ci, vi = meta['calibration_indices'], meta['validation_indices']
        x = torch.from_numpy(np.asarray(np.concatenate([train_x[ci], val_x[vi]]), dtype=np.float32))
        y = torch.from_numpy(np.concatenate([train_y[ci], val_y[vi]]).astype(np.int64))
        del train_x, train_y, val_x, val_y, test_x, test_y
        assert array_hash(x) == meta['eeg_hash'] and array_hash(y) == meta['labels_hash']
        assert array_hash(instruction) == meta['instruction_hash']
        assert array_hash(prototypes) == meta['prototypes_hash']
        data[task], caches[task] = x, cache
    setup_seconds = time.time()-started
    for own in expert_names:
        expert_started = time.time()
        folder = outdir/own
        folder.mkdir(parents=True, exist_ok=False)
        source_path = Path(source['experts'][own])
        assert sha256(source_path) == caches[own]['metadata']['expert_sha256']
        leaf.load_state_dict(torch.load(source_path, map_location=device, weights_only=True), strict=True)
        assert all(torch.equal(leaf.state_dict()[name].cpu(), value) for name, value in fixed.items())
        active = [value for value in leaf.parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(active, lr=plan['lr'], weight_decay=0)
        expert_index = tasks.index(own)
        rng = np.random.default_rng([plan['seed'], expert_index, 1147])
        samplers = {task: np.random.default_rng([plan['seed'], expert_index, i, 247])
                    for i, task in enumerate(tasks)}
        wrappers = {task: PrototypeClassifier(leaf, *readouts[task]).to(device).eval() for task in tasks}
        measured_tasks = tasks

        @torch.no_grad()
        def measure(label):
            report, saved = {}, {}
            for task in measured_tasks:
                values = predictions(leaf, readouts[task], data[task], device)
                cache = caches[task]
                n = cache['metadata']['n_calibration']
                report[task] = {}
                for split, select in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
                    pred, truth = values[select], cache['truth'][select]
                    base_logits = cache['base_logits'][select]
                    own_logits = cache['own_expert_logits'][select]
                    row = classification_metrics(pred, truth)
                    row['kl_from_base'] = float(divergence(pred, base_logits).mean())
                    row['kl_from_target_domain_expert'] = float(divergence(pred, own_logits).mean())
                    row['base_disagreement'] = float((pred.argmax(-1) != base_logits.argmax(-1)).float().mean())
                    report[task][split] = row
                saved[task] = values
            torch.save(saved, folder/f'{label}_logits.pt')
            (folder/f'{label}_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
            return report, saved

        before, before_logits = measure('before')
        initial_error = float((before_logits[own]-caches[own]['own_expert_logits']).abs().max())
        assert initial_error <= 1e-5, initial_error
        other_tasks = [task for task in tasks if task != own]
        gate_source = Path(transfer_plan['ri_bank'])/own/'before_logits.pt'
        original_predictions = torch.load(gate_source, map_location='cpu', weights_only=True)
        assert set(original_predictions) == set(tasks)
        assert max(float((original_predictions[t]-before_logits[t]).abs().max()) for t in tasks) <= 1e-5
        calibration_gains = {}
        for task in other_tasks:
            cache = caches[task]
            n = cache['metadata']['n_calibration']
            base_bacc = classification_metrics(cache['base_logits'][:n], cache['truth'][:n])['BAcc']
            original_bacc = classification_metrics(original_predictions[task][:n], cache['truth'][:n])['BAcc']
            calibration_gains[task] = original_bacc-base_bacc
        protected = choose_protected(tasks, own, calibration_gains, condition, transfer_plan['gate_seed'])
        if condition == 'ri_reference':
            assert smoke, 'Reference condition is an engineering check only'
        print('Protected domains', own, condition, protected, flush=True)
        counts = {task: 0 for task in other_tasks}
        steps = (4 if condition == 'ri_reference' else transfer_plan['engineering_smoke_updates']) if smoke else plan['steps_per_expert']
        bs = plan['batch_size_per_term']
        fit_started = time.time()
        for step in range(steps):
            other = other_tasks[int(rng.integers(len(other_tasks)))]
            counts[other] += 1
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for task, target_key in [(own, 'own_expert_logits'), (other, 'base_logits')]:
                n = caches[task]['metadata']['n_calibration']
                ids = samplers[task].choice(n, size=bs, replace=False)
                predicted = wrappers[task](data[task][ids].to(device))
                # The original expert's own predictions on this other domain
                # are frozen before adaptation; no validation sample selects it.
                teacher_values = original_predictions[task] if task in protected else caches[task][target_key]
                teacher = teacher_values[ids].to(device)
                loss = divergence(predicted, teacher).mean()
                assert torch.isfinite(loss), (own, task, step)
                loss.backward()
                losses.append(float(loss.detach()))
            gradient_norm = torch.nn.utils.clip_grad_norm_(active, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step+1) % 50 == 0 or step+1 == steps:
                row = {'step': step+1, 'other_task': other, 'own_kl': losses[0],
                       'other_kl': losses[1], 'gradient_norm': float(gradient_norm)}
                with (folder/'dynamics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(own, row, flush=True)
        fit_seconds = time.time()-fit_started
        after, after_logits = measure('after')
        state = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
        assert all(torch.equal(state[name], value) for name, value in fixed.items())
        # The exact fixed causal-mask buffer intentionally contains -inf.
        assert all(torch.isfinite(state[name]).all() for name in active_names)
        torch.save(state, folder/'best.ckpt')
        reloaded = load_leaf(config, folder/'best.ckpt').to(device).eval()
        checks = {}
        for task in measured_tasks:
            # Actual full-checkpoint forwards, fixed probe from every domain.
            ids = torch.cat([torch.where(caches[task]['truth'] == c)[0][:8]
                             for c in caches[task]['truth'].unique()])
            values = predictions(reloaded, readouts[task], data[task][ids], device)
            checks[task] = compare(after_logits[task][ids], values, caches[task]['truth'][ids])
            assert checks[task]['max_abs_logit_error'] <= 1e-5
            assert all(row['margin_within_twice_error'] for row in checks[task]['changed_rows'])
        del reloaded
        metadata = json.loads(source_path.with_name('run.json').read_text())
        metadata['transfer_adaptation'] = {
            'condition': condition, 'transfer_plan': transfer_plan,
            'gate_source': str(gate_source), 'gate_source_sha256': sha256(gate_source),
            'calibration_bacc_gains_before': calibration_gains,
            'protected_other_domains': protected,
            'calibration_helpful_domains': [task for task in other_tasks if calibration_gains[task] > 0],
            'rule': 'Use pre-adaptation calibration BAcc only. Protected other domains retain this expert; all other domains restore base. Own preservation unchanged.',
            'source_expert': str(source_path), 'source_sha256': sha256(source_path),
            'checkpoint_sha256': sha256(folder/'best.ckpt'),
            'plan': plan, 'cache_manifest_sha256': sha256(cache_dir/'run.json'),
            'initial_teacher_max_error': initial_error,
            'updates': steps, 'other_domain_update_counts': counts,
            'before': before, 'after': after, 'export_checks': checks,
            'fixed_buffers_and_inactive_exact': True, 'test_inference_performed': False,
            'code_sha256': sha256(Path(__file__)), 'engineering_smoke': smoke,
            'seconds': {'worker_setup_shared': setup_seconds, 'fit': fit_seconds,
                        'expert_total': time.time()-expert_started},
            'selection': 'Fixed final adaptation step; validation is diagnostic only',
        }
        metadata['original_training_metrics_note'] = (
            'Original expert training metrics retained as provenance; adapted model metrics '
            'and checkpoint identity are in transfer_adaptation. No validation selection of adaptation.')
        (folder/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('Completed', own, 'seconds', time.time()-expert_started, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--transfer-plan', type=Path, required=True)
    p.add_argument('--condition', choices=['helpful', 'random', 'ri_reference'], required=True)
    p.add_argument('--outdir', type=Path)
    p.add_argument('--experts', help='Comma-separated expert names assigned to this worker')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    transfer_plan = json.loads(args.transfer_plan.read_text())
    assert sha256(Path(__file__)) == transfer_plan['code_sha256']
    assert sha256(args.plan) == transfer_plan['ri_plan_sha256']
    device = configure(args.gpu, plan['seed'])
    assert args.outdir is not None and args.experts
    adapt(plan, args.cache, args.outdir, device, args.experts.split(','),
          transfer_plan, args.condition, args.smoke)


if __name__ == '__main__':
    main()
