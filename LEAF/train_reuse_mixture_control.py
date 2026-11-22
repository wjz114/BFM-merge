"""Matched full-weight CE mixing old and replacement rows at fixed batch labels."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure, divergence
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, selection_key, sha256
from load_config import build_model_config, load_yaml
from coverage_support import training_and_validation
from merge_poc import UNUSED_PREFIXES


def objective(logits, truth, weights, teacher, condition):
    if condition == 'balanced_ce':
        return (F.cross_entropy(logits, truth, reduction='none')*weights[truth]).mean()
    return divergence(logits, teacher).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=['balanced_ce', 'own_teacher_kl'], required=True)
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
    assert sha256(Path(plan['teacher_cache'])/'run.json') == plan['teacher_cache_sha256']
    assert args.condition == 'balanced_ce'
    assert plan['training_pool'] == 'reuse_mixture512'
    assert sha256(Path(plan['mixture_mask'])) == plan['mixture_mask_sha256']
    assert sha256(Path(plan['mixture_audit'])) == plan['mixture_audit_sha256']
    mixture = np.load(plan['mixture_mask'])['use_old']
    assert mixture.dtype == np.bool_ and mixture.shape == (plan['steps'], plan['batch_size'])
    mixture_audit = json.loads(Path(plan['mixture_audit']).read_text())
    assert sha256(Path(plan['training_indices'])) == plan['training_indices_sha256']
    assert sha256(Path(plan['fresh_data_audit'])) == plan['fresh_data_audit_sha256']
    fresh_audit = json.loads(Path(plan['fresh_data_audit']).read_text())
    assert sha256(Path(plan['coverage_audit'])) == plan['coverage_audit_sha256']
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    source = json.loads((Path(plan['source_fit'])/'run.json').read_text())
    initial_run = json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    tasks = source['datasets']
    indices, schedule = np.load(plan['calibration']), np.load(plan['schedule'])
    training_indices = np.load(plan['training_indices'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    active_names = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    for name, value in leaf.named_parameters():
        value.requires_grad_(name in active_names)
    fixed = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items() if name not in active_names}
    train, validation, readouts, weights, teachers, probes, dataset_audits = {}, {}, {}, {}, {}, {}, {}
    calibration_data = {}
    for task in tasks:
        labels, text, instruction, prototypes = prepare_text(config, task)
        readouts[task] = instruction.to(device), prototypes.to(device)
        cache = torch.load(Path(plan['teacher_cache'])/f'{task}.pt', map_location='cpu', weights_only=True)
        meta = cache['metadata']
        assert meta['calibration_indices'] == indices[task].tolist()
        assert meta['expert_sha256'] == sha256(Path(source['experts'][task]))
        assert meta['labels'] == labels and meta['instruction'] == text
        assert array_hash(instruction) == meta['instruction_hash'] and array_hash(prototypes) == meta['prototypes_hash']
        tx, ty, vx, vy = training_and_validation(task)
        audit = coverage['audits'][task]
        assert len(ty) == audit['train_size'] and len(vy) == audit['validation_size']
        assert array_hash(tx) == audit['train_eeg_hash'] and array_hash(ty) == audit['train_labels_hash']
        assert array_hash(vx) == audit['validation_eeg_hash'] and array_hash(vy) == audit['validation_labels_hash']
        n = len(indices[task])
        x = torch.from_numpy(np.asarray(tx[indices[task]], dtype=np.float32))
        y = torch.from_numpy(np.asarray(ty[indices[task]], dtype=np.int64))
        # Verify teacher-cache data identity, including its fixed validation probe.
        cache_x = torch.from_numpy(np.asarray(np.concatenate([tx[indices[task]], vx[meta['validation_indices']]]), dtype=np.float32))
        cache_y = torch.from_numpy(np.concatenate([ty[indices[task]], vy[meta['validation_indices']]]).astype(np.int64))
        assert array_hash(cache_x) == meta['eeg_hash'] and array_hash(cache_y) == meta['labels_hash']
        assert torch.equal(y, cache['truth'][:n])
        calibration_data[task] = x, y
        fresh_ids = training_indices[task]
        assert len(fresh_ids) == n == 512 and len(np.unique(fresh_ids)) == n
        assert fresh_ids.min() >= 0 and fresh_ids.max() < len(ty)
        assert np.array_equal(ty[fresh_ids], ty[indices[task]])
        overlap = len(np.intersect1d(fresh_ids, indices[task]))
        assert overlap == fresh_audit['datasets'][task]['minimum_overlap']
        assert array_hash(ty[fresh_ids]) == fresh_audit['datasets'][task]['selected_label_hash']
        x = torch.from_numpy(np.asarray(tx[fresh_ids], dtype=np.float32))
        y = torch.from_numpy(np.asarray(ty[fresh_ids], dtype=np.int64))
        train[task] = x, y
        teachers[task] = cache['own_expert_logits'][:n]
        counts = torch.bincount(y, minlength=len(labels))
        weights[task] = (len(y)/(len(labels)*counts.float())).to(device)
        assert (counts > 0).all()
        assert weights[task].cpu().tolist() == source['ce_class_weights'][task]
        assert counts.tolist() == source['calibration_class_counts'][task]
        ids = schedule['local_indices'][schedule['task'] == tasks.index(task)]
        assert ids.min() >= 0 and ids.max() < len(y)
        task_mask = schedule['task'] == tasks.index(task)
        assert int(mixture[task_mask].sum()) == mixture_audit['datasets'][task]['old_pool_exposures'] == 6400
        probe_ids = np.concatenate([np.flatnonzero(vy==c)[:8] for c in range(len(labels))])
        probes[task] = torch.from_numpy(np.asarray(vx[probe_ids], dtype=np.float32)), torch.from_numpy(vy[probe_ids].astype(np.int64))
        if args.smoke:
            vx, vy = vx[probe_ids], vy[probe_ids]
        validation[task] = make_loader(vx, vy, config, len(labels), plan['batch_size'], False)
        dataset_audits[task] = {'calibration_size': n, 'validation_size': len(vy),
                               'training_pool': plan['training_pool'], 'training_pool_size': 2*n-overlap,
                               'replacement_pool_size': n, 'old_pool_size': n,
                               'class_counts': counts.tolist(), 'class_weights': weights[task].cpu().tolist(),
                               'class_counts_scope': 'Either512 source pool; both have identical class counts and each batch retains its original labels.',
                               'eeg_hash': array_hash(calibration_data[task][0]),
                               'training_pool_eeg_hash': array_hash(x),
                               'overlap_with_merger_calibration': n,
                               'replacement_overlap_with_merger_calibration': overlap,
                               'new_to_merger_calibration': n-overlap,
                               'old_pool_exposures': mixture_audit['datasets'][task]['old_pool_exposures'],
                               'replacement_pool_exposures': mixture_audit['datasets'][task]['replacement_pool_exposures'],
                               'actual_new_row_exposures': mixture_audit['datasets'][task]['actual_new_row_exposures'],
                               'batch_label_sequence_matches_original': True,
                               'teacher_hash': array_hash(teachers[task]), 'teacher_expert_sha256': meta['expert_sha256']}
        del tx, ty, vx, vy, cache, cache_x, cache_y
    wrappers = {task: PrototypeClassifier(leaf, *readouts[task]).to(device).eval() for task in tasks}
    steps = 32 if args.smoke else plan['steps']
    eval_every = 16 if args.smoke else plan['eval_every']
    average_steps = [16, 32] if args.smoke else plan['average_steps']
    history, snapshots = [], []
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
        if step == 0 and not args.smoke:
            assert all(abs(metrics[t]['BAcc']-initial_run['folded_val'][t]['BAcc']) < 1e-12 for t in tasks)
        history.append({'step': step, 'validation': metrics, 'macro_bacc': bacc, 'macro_loss': loss})
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
        batch = x[ids].clone()
        use_old = torch.from_numpy(mixture[step])
        old_ids = torch.from_numpy(ids)[use_old]
        batch[use_old] = calibration_data[task][0][old_ids]
        logits = wrappers[task](batch.to(device))
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
    calibration_metrics, training_subset_metrics = {}, {}
    with torch.no_grad():
        for task in tasks:
            x, y = calibration_data[task]
            logits = torch.cat([wrappers[task](x[i:i+plan['batch_size']].to(device)).cpu()
                                for i in range(0, len(x), plan['batch_size'])])
            calibration_metrics[task] = classification_metrics(logits, y)
            calibration_metrics[task]['kl_from_own_teacher'] = float(divergence(logits, teachers[task]).mean())
            x, y = train[task]
            logits = torch.cat([wrappers[task](x[i:i+plan['batch_size']].to(device)).cpu()
                                for i in range(0, len(x), plan['batch_size'])])
            training_subset_metrics[task] = classification_metrics(logits, y)
    result = {'condition': args.condition, 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'datasets': tasks, 'engineering_smoke': args.smoke,
              'updates': steps, 'training_examples': steps*plan['batch_size'], 'schedule_sha256': plan['schedule_sha256'],
              'trainable_parameters': sum(v.numel() for v in leaf.parameters() if v.requires_grad),
              'component_gradient_max': gradient_max, 'fixed_tensors_exact': True,
              'initial_validation': history[0], 'selected_step': best['step'], 'selected_validation': best['validation'],
              'final_validation': history[-1]['validation'], 'average_steps': average_steps,
              'average_snapshots': snapshots, 'average_validation': averaged_metrics,
              'average_validation_macro_bacc': selection_key(averaged_metrics)[0],
              'average_calibration': calibration_metrics, 'dataset_audits': dataset_audits,
              'average_training_subset': training_subset_metrics,
              'export_checks': checks, 'checkpoint_sha256': sha256(args.outdir/'merged.ckpt'),
              'seconds': {'setup': setup_seconds, 'fit_including_validation': fit_seconds,
                          'averaging_export_evaluation': time.time()-average_started, 'total': time.time()-started},
              'training_pool': plan['training_pool'],
              'training_indices_sha256': plan['training_indices_sha256'],
              'fresh_data_audit_sha256': plan['fresh_data_audit_sha256'],
              'mixture_mask_sha256': plan['mixture_mask_sha256'],
              'mixture_audit_sha256': plan['mixture_audit_sha256'],
              'training_subset_evaluation_scope': 'Replacement512 rows only; original512 reference metrics are reported separately.',
              'calibration_evaluation_scope': 'Always the same original512 reference examples per task.',
              'test_data_read': False, 'test_inference_performed': False}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', args.condition, 'fixed average BAcc', result['average_validation_macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
