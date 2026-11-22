"""Fixed-duration CE control with exact replay of all six prior prefix checkpoints."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, selection_key, sha256
from load_config import build_model_config, load_yaml
from coverage_support import training_and_validation
from merge_poc import UNUSED_PREFIXES


def objective(logits, truth, weights, teacher, condition):
    assert condition == 'balanced_ce'
    return (F.cross_entropy(logits, truth, reduction='none')*weights[truth]).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=['balanced_ce'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha256(Path(__file__)) == plan['source_sha256'][Path(__file__).name]
    for name, expected in plan['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    assert sha256(Path(plan['initial_checkpoint'])) == plan['initial_checkpoint_sha256']
    assert sha256(Path(plan['calibration'])) == plan['calibration_sha256']
    assert sha256(Path(plan['schedule'])) == plan['schedule_sha256']
    assert args.condition == 'balanced_ce'
    assert plan['training_pool'] in ['original512', 'full_training']
    assert sha256(Path(plan['coverage_audit'])) == plan['coverage_audit_sha256']
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    source = json.loads((Path(plan['source_fit'])/'run.json').read_text())
    reference_path = plan.get('initial_reference')
    initial_run = json.loads(Path(reference_path).read_text()) if reference_path else None
    tasks = source['datasets']
    indices, schedule = np.load(plan['calibration']), np.load(plan['schedule'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    active_names = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    for name, value in leaf.named_parameters():
        value.requires_grad_(name in active_names)
    fixed = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items() if name not in active_names}
    train, validation, readouts, weights, probes, dataset_audits = {}, {}, {}, {}, {}, {}
    calibration_data = {}
    for task in tasks:
        labels, text, instruction, prototypes = prepare_text(config, task)
        readouts[task] = instruction.to(device), prototypes.to(device)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        tx, ty, vx, vy = training_and_validation(task)
        audit = coverage['audits'][task]
        assert len(ty) == audit['train_size'] and len(vy) == audit['validation_size']
        assert array_hash(tx) == audit['train_eeg_hash'] and array_hash(ty) == audit['train_labels_hash']
        assert array_hash(vx) == audit['validation_eeg_hash'] and array_hash(vy) == audit['validation_labels_hash']
        n = len(indices[task])
        x = torch.from_numpy(np.asarray(tx[indices[task]], dtype=np.float32))
        y = torch.from_numpy(np.asarray(ty[indices[task]], dtype=np.int64))
        assert len(y) == 512 and torch.bincount(y, minlength=len(labels)).tolist() == source['calibration_class_counts'][task]
        calibration_data[task] = x, y
        if plan['training_pool'] == 'full_training':
            x, y = torch.from_numpy(tx), torch.from_numpy(ty)
        train[task] = x, y
        counts = torch.bincount(y, minlength=len(labels))
        weights[task] = (len(y)/(len(labels)*counts.float())).to(device)
        assert (counts > 0).all()
        if plan['training_pool'] == 'original512':
            assert weights[task].cpu().tolist() == source['ce_class_weights'][task]
        else:
            assert counts.tolist() == audit['train_class_counts']
        ids = schedule['local_indices'][schedule['task'] == tasks.index(task)]
        assert ids.min() >= 0 and ids.max() < len(y)
        probe_ids = np.concatenate([np.flatnonzero(vy==c)[:8] for c in range(len(labels))])
        probes[task] = torch.from_numpy(np.asarray(vx[probe_ids], dtype=np.float32)), torch.from_numpy(vy[probe_ids].astype(np.int64))
        if args.smoke:
            vx, vy = vx[probe_ids], vy[probe_ids]
        validation[task] = make_loader(vx, vy, config, len(labels), plan['batch_size'], False)
        dataset_audits[task] = {'calibration_size': n, 'validation_size': len(vy),
                               'training_pool': plan['training_pool'], 'training_pool_size': len(y),
                               'class_counts': counts.tolist(), 'class_weights': weights[task].cpu().tolist(),
                               'eeg_hash': array_hash(calibration_data[task][0]),
                               'training_pool_eeg_hash': array_hash(x),
                               'labels': labels, 'instruction_hash': array_hash(instruction),
                               'prototypes_hash': array_hash(prototypes)}
        del tx, ty, vx, vy
    wrappers = {task: PrototypeClassifier(leaf, *readouts[task]).to(device).eval() for task in tasks}
    steps = 32 if args.smoke else plan['steps']
    eval_every = 16 if args.smoke else plan['eval_every']
    average_steps = [16, 32] if args.smoke else plan['average_steps']
    history, snapshots = [], []
    prefix_checks = {}
    best = {'key': (-float('inf'), -float('inf'))}
    gradient_max = {'tower': 0.0, 'qformer': 0.0}

    def cpu_state():
        state = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
        assert all(torch.equal(state[name], value) for name, value in fixed.items())
        assert all(torch.isfinite(state[name]).all() for name in active_names)
        return state

    @torch.no_grad()
    def evaluate():
        result = {}
        for task in tasks:
            logits, truth = [], []
            for x, y in validation[task]:
                logits.append(wrappers[task](x.to(device)).cpu())
                truth.append(y)
            result[task] = classification_metrics(torch.cat(logits), torch.cat(truth))
        return result

    def record(step):
        metrics = evaluate()
        bacc, loss, key = selection_key(metrics)
        if step == 0 and not args.smoke and initial_run is not None:
            assert all(abs(metrics[t]['BAcc']-initial_run['folded_val'][t]['BAcc']) < 1e-12 for t in tasks)
        history.append({'step': step, 'validation': metrics, 'macro_bacc': bacc, 'macro_loss': loss})
        if not args.smoke and str(step) in plan['prefix_reference_snapshots']:
            spec = plan['prefix_reference_snapshots'][str(step)]
            assert sha256(Path(spec['path'])) == spec['sha256']
            reference = torch.load(spec['path'], map_location='cpu', weights_only=True)
            current = cpu_state()
            assert reference.keys() == current.keys()
            changed = {name: float((current[name].double()-reference[name].double()).abs().max())
                       for name in current if not torch.equal(current[name], reference[name])}
            old_history = json.loads(Path(plan['prefix_reference_history']).read_text())
            reference_metrics = next(row['validation'] for row in old_history if row['step'] == step)
            prefix_checks[str(step)] = {'reference': spec, 'nonidentical_state_tensors': changed,
                                        'all_validation_metrics_exact': metrics == reference_metrics}
            (args.outdir/'prefix_replay.json').write_text(json.dumps(prefix_checks, indent=2)+'\n')
            assert not changed, (step, changed)
            assert metrics == reference_metrics, step
            print('Exact prefix checkpoint and metrics verified', step, flush=True)
            del reference, current
        if key > best['key']:
            best.update(key=key, step=step, macro_bacc=bacc, validation=metrics)
            torch.save(cpu_state(), args.outdir/'selected.ckpt')
        if step in average_steps:
            snapshot = args.outdir/f'step_{step}.ckpt'
            torch.save(cpu_state(), snapshot)
            snapshots.append({'step': step, 'path': str(snapshot), 'sha256': sha256(snapshot)})
        (args.outdir/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print('step', step, 'validation macro BAcc', bacc, flush=True)

    setup_seconds = time.time()-started
    record(0)
    optimizer = torch.optim.Adam([value for value in leaf.parameters() if value.requires_grad], lr=plan['lr'], weight_decay=0)
    fit_started = time.time()
    for step in range(steps):
        task = tasks[int(schedule['task'][step])]
        ids = schedule['local_indices'][step]
        x, y = train[task]
        logits = wrappers[task](x[ids].to(device))
        truth = y[ids].to(device)
        teacher = None  # CE does not use teacher predictions; IDs now index the specified pool.
        loss = objective(logits, truth, weights[task], teacher, args.condition)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([v for v in leaf.parameters() if v.requires_grad], float('inf'), error_if_nonfinite=True)
        assert torch.isfinite(loss)
        if step == 0 or (step+1) % 50 == 0:
            component_norms = {}
            for component in gradient_max:
                squares = [v.grad.square().sum() for name, v in leaf.named_parameters()
                           if name.startswith(component+'.') and v.grad is not None]
                value = float(torch.stack(squares).sum().sqrt())
                gradient_max[component] = max(gradient_max[component], value)
                component_norms[component] = value
            with (args.outdir/'dynamics.jsonl').open('a') as f:
                f.write(json.dumps({'step': step+1, 'task': task, 'loss': float(loss), 'gradient_norm': float(norm),
                                    'component_gradient_norms': component_norms})+'\n')
        optimizer.step()
        if (step+1) % eval_every == 0 or step+1 == steps:
            record(step+1)
    fit_seconds = time.time()-fit_started
    if not args.smoke:
        assert set(prefix_checks) == set(plan['prefix_reference_snapshots'])
    assert all(value>0 for value in gradient_max.values())
    torch.save(cpu_state(), args.outdir/'final.ckpt')
    del optimizer
    average_started = time.time()
    assert [row['step'] for row in snapshots] == average_steps
    states = [torch.load(row['path'], map_location='cpu', weights_only=True) for row in snapshots]
    averaged = {name: (torch.stack([state[name].double() for state in states]).mean(0).float()
                       if name in active_names else value.clone()) for name, value in states[0].items()}
    assert all(torch.equal(averaged[name], value) for name, value in fixed.items())
    torch.save(averaged, args.outdir/'merged.ckpt')
    leaf.load_state_dict(averaged, strict=True)
    reloaded = load_leaf(config, args.outdir/'merged.ckpt').to(device).eval()
    checks = {}
    with torch.no_grad():
        for task in tasks:
            x, truth = probes[task]
            before = wrappers[task](x.to(device)).cpu()
            after = PrototypeClassifier(reloaded, *readouts[task]).to(device).eval()(x.to(device)).cpu()
            checks[task] = compare(before, after, truth)
            assert checks[task]['max_abs_logit_error'] <= 1e-5
            assert all(row['margin_within_twice_error'] for row in checks[task]['changed_rows'])
    del reloaded, states
    averaged_metrics = evaluate()
    calibration_metrics = {}
    with torch.no_grad():
        for task in tasks:
            x, y = calibration_data[task]
            logits = torch.cat([wrappers[task](x[i:i+plan['batch_size']].to(device)).cpu()
                                for i in range(0, len(x), plan['batch_size'])])
            calibration_metrics[task] = classification_metrics(logits, y)
    result = {'condition': args.condition, 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'datasets': tasks, 'engineering_smoke': args.smoke,
              'updates': steps, 'training_examples': steps*plan['batch_size'], 'schedule_sha256': plan['schedule_sha256'],
              'trainable_parameters': sum(v.numel() for v in leaf.parameters() if v.requires_grad),
              'component_gradient_max': gradient_max, 'fixed_tensors_exact': True,
              'initial_reference_used': reference_path,
              'initial_validation': history[0], 'selected_step': best['step'], 'selected_validation': best['validation'],
              'final_validation': history[-1]['validation'], 'average_steps': average_steps,
              'average_snapshots': snapshots, 'average_validation': averaged_metrics,
              'average_validation_macro_bacc': selection_key(averaged_metrics)[0],
              'average_calibration': calibration_metrics, 'dataset_audits': dataset_audits,
              'export_checks': checks, 'checkpoint_sha256': sha256(args.outdir/'merged.ckpt'),
              'seconds': {'setup': setup_seconds, 'fit_including_validation': fit_seconds,
                          'averaging_export_evaluation': time.time()-average_started, 'total': time.time()-started},
              'training_pool': plan['training_pool'],
              'calibration_evaluation_scope': 'Always the same original512 reference examples per task.',
              'test_data_read': False, 'test_inference_performed': False,
              'prefix_replay': prefix_checks,
              'teacher_usage': 'None. BalancedCE updates need no expert cache; the unused teacher-cache input checks and final KL diagnostic were removed.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', args.condition, 'fixed average BAcc', result['average_validation_macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
