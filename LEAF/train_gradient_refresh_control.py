"""Continue a shared step800 anchor with a fixed or refreshed basis and matched optimizer reset."""
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
from coefficient_control_support import ResidualTensorMerger
from gradient_basis_support import AugmentedResidualTensorMerger


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
    for path, expected in plan['input_sha256'].items():
        assert sha256(Path(path)) == expected, path
    assert sha256(Path(plan['initial_checkpoint'])) == plan['initial_checkpoint_sha256']
    assert sha256(Path(plan['calibration'])) == plan['calibration_sha256']
    assert sha256(Path(plan['schedule'])) == plan['schedule_sha256']
    assert sha256(Path(plan['teacher_cache'])/'run.json') == plan['teacher_cache_sha256']
    assert args.condition == 'balanced_ce'
    assert plan['training_pool'] in ['original512', 'full_training']
    assert sha256(Path(plan['coverage_audit'])) == plan['coverage_audit_sha256']
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    source = json.loads((Path(plan['source_fit'])/'run.json').read_text())
    start_step = plan['start_step']
    assert start_step == 800
    anchor_history = json.loads(Path(plan['anchor_history']).read_text())
    anchor_reference = next(row for row in anchor_history if row['step'] == start_step)
    tasks = source['datasets']
    indices, schedule = np.load(plan['calibration']), np.load(plan['schedule'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    assert plan['basis_kind'] in ['original_only', 'gradient', 'random']
    if plan['basis_kind'] == 'original_only':
        merger = ResidualTensorMerger(leaf, source)
    else:
        assert sha256(Path(plan['basis_run'])) == plan['basis_run_sha256']
        basis = json.loads(Path(plan['basis_run']).read_text())
        assert sha256(Path(basis['gradient_cache'])) == basis['gradient_cache_sha256']
        gradient_cache = json.loads(Path(basis['gradient_cache']).read_text())
        cached_initial = gradient_cache['plan'].get('checkpoint', gradient_cache['plan'].get('initial_checkpoint'))
        assert gradient_cache['plan']['input_sha256'][cached_initial] == plan['basis_gradient_checkpoint_sha256']
        output = basis['outputs'][plan['basis_kind']]
        merger = AugmentedResidualTensorMerger(leaf, source, output['path'], output['sha256'], basis['geometry'])
    active_names = set(merger.active)
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
        if plan['training_pool'] == 'full_training':
            x, y = torch.from_numpy(tx), torch.from_numpy(ty)
        train[task] = x, y
        teachers[task] = cache['own_expert_logits'][:n]
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
                               'teacher_hash': array_hash(teachers[task]), 'teacher_expert_sha256': meta['expert_sha256']}
        del tx, ty, vx, vy, cache, cache_x, cache_y
    wrappers = {task: PrototypeClassifier(leaf, *readouts[task]).to(device).eval() for task in tasks}
    initial_probe_checks = {}
    with torch.no_grad():
        initial_params = merger.compose()
        assert all(torch.equal(initial_params[n], merger.initial[n]) for n in initial_params)
        for task in tasks:
            x, y = probes[task]
            direct = wrappers[task](x.to(device)).cpu()
            composed = merger.logits(wrappers[task], x.to(device), initial_params).cpu()
            initial_probe_checks[task] = compare(direct, composed, y)
            assert torch.equal(direct, composed), task
    del initial_params
    steps = 32 if args.smoke else plan['steps']-start_step
    eval_every = 16 if args.smoke else plan['eval_every']
    average_steps = [start_step+16, start_step+32] if args.smoke else plan['average_steps']
    history, snapshots = [], []
    best = {'key': (-float('inf'), -float('inf'))}
    gradient_max = {'tower': 0.0, 'qformer': 0.0}

    def cpu_state():
        return merger.materialize()

    @torch.no_grad()
    def evaluate(functional=True):
        result = {}
        params = merger.compose() if functional else None
        for task in tasks:
            logits, truth = [], []
            for x, y in validation[task]:
                value = merger.logits(wrappers[task], x.to(device), params) if functional else wrappers[task](x.to(device))
                logits.append(value.cpu())
                truth.append(y)
            result[task] = classification_metrics(torch.cat(logits), torch.cat(truth))
        return result

    def record(step):
        metrics = evaluate()
        bacc, loss, key = selection_key(metrics)
        if step == start_step and not args.smoke:
            assert all(abs(metrics[t]['BAcc']-anchor_reference['validation'][t]['BAcc']) < 1e-12 for t in tasks)
        history.append({'step': step, 'validation': metrics, 'macro_bacc': bacc, 'macro_loss': loss})
        if key > best['key']:
            best.update(key=key, step=step, macro_bacc=bacc, validation=metrics)
            torch.save(cpu_state(), args.outdir/'selected.ckpt')
            torch.save(merger.beta.detach().cpu(), args.outdir/'selected_beta.pt')
        if step in average_steps:
            snapshot = args.outdir/f'step_{step}.ckpt'
            torch.save(cpu_state(), snapshot)
            beta_path = args.outdir/f'beta_{step}.pt'
            torch.save(merger.beta.detach().cpu(), beta_path)
            snapshots.append({'step': step, 'path': str(snapshot), 'sha256': sha256(snapshot),
                              'beta_path':str(beta_path), 'beta_sha256':sha256(beta_path)})
        (args.outdir/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print('step', step, 'validation macro BAcc', bacc, flush=True)

    setup_seconds = time.time()-started
    record(start_step)
    optimizer = torch.optim.Adam([merger.beta], lr=plan['lr'], weight_decay=0)
    assert len(optimizer.state) == 0
    fit_started = time.time()
    for step in range(start_step, start_step+steps):
        task = tasks[int(schedule['task'][step])]
        ids = schedule['local_indices'][step]
        x, y = train[task]
        logits = merger.logits(wrappers[task], x[ids].to(device))
        truth = y[ids].to(device)
        teacher = None  # CE does not use teacher predictions; IDs now index the specified pool.
        loss = objective(logits, truth, weights[task], teacher, args.condition)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([merger.beta], float('inf'), error_if_nonfinite=True)
        assert torch.isfinite(loss)
        if step == start_step or (step+1) % 50 == 0:
            component_norms = {}
            for component in gradient_max:
                columns = merger.component_columns[component]
                value = float(merger.beta.grad[:,columns].norm())
                gradient_max[component] = max(gradient_max[component], value)
                component_norms[component] = value
            with (args.outdir/'dynamics.jsonl').open('a') as f:
                f.write(json.dumps({'step': step+1, 'task': task, 'loss': float(loss), 'gradient_norm': float(norm),
                                    'component_gradient_norms': component_norms})+'\n')
        optimizer.step()
        if (step+1) % eval_every == 0 or step+1 == start_step+steps:
            record(step+1)
    fit_seconds = time.time()-fit_started
    assert all(value>0 for value in gradient_max.values())
    torch.save(cpu_state(), args.outdir/'final.ckpt')
    torch.save(merger.beta.detach().cpu(), args.outdir/'final_beta.pt')
    del optimizer
    average_started = time.time()
    assert [row['step'] for row in snapshots] == average_steps
    states = [torch.load(row['path'], map_location='cpu', weights_only=True) for row in snapshots]
    averaged = {name: (torch.stack([state[name].double() for state in states]).mean(0).float()
                       if name in active_names else value.clone()) for name, value in states[0].items()}
    assert all(torch.equal(averaged[name], value) for name, value in fixed.items())
    torch.save(averaged, args.outdir/'merged.ckpt')
    beta_states = [torch.load(row['beta_path'], map_location='cpu', weights_only=True) for row in snapshots]
    mean_beta = torch.stack([value.double() for value in beta_states]).mean(0).float()
    torch.save(mean_beta, args.outdir/'mean_beta.pt')
    coefficient_fold = merger.materialize(mean_beta.to(device))
    coefficient_mean_state_error = max(float((coefficient_fold[n]-averaged[n]).abs().max()) for n in active_names)
    assert coefficient_mean_state_error <= 1e-5
    del coefficient_fold, beta_states
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
    averaged_metrics = evaluate(functional=False)
    calibration_metrics = {}
    with torch.no_grad():
        for task in tasks:
            x, y = calibration_data[task]
            logits = torch.cat([wrappers[task](x[i:i+plan['batch_size']].to(device)).cpu()
                                for i in range(0, len(x), plan['batch_size'])])
            calibration_metrics[task] = classification_metrics(logits, y)
            calibration_metrics[task]['kl_from_own_teacher'] = float(divergence(logits, teachers[task]).mean())
    result = {'condition': args.condition, 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'datasets': tasks, 'engineering_smoke': args.smoke,
              'start_step': start_step, 'end_step': start_step+steps, 'optimizer_reset_at_anchor': True,
              'prefix_run': plan['prefix_run'], 'branch': plan['branch'],
              'updates': steps, 'training_examples': steps*plan['batch_size'], 'schedule_sha256': plan['schedule_sha256'],
              'trainable_parameters': merger.beta.numel(),
              'updated_model_parameter_elements':sum(merger.initial[n].numel() for n in active_names),
              'method':'one_refresh_intervention_'+plan['branch'], 'coefficient_names':merger.active,
              'basis_kind':plan['basis_kind'],
              'basis_artifact':merger.basis_info if plan['basis_kind'] != 'original_only' else None,
              'effective_trainable_parameters':merger.effective_coefficient_count if plan['basis_kind'] != 'original_only' else merger.beta.numel(),
              'coefficient_shape':list(merger.beta.shape), 'expert_sources':merger.sources,
              'initial_parameter_and_probe_replay_exact':True, 'initial_probe_checks':initial_probe_checks,
              'mean_coefficient_vs_mean_state_max_abs':coefficient_mean_state_error,
              'mean_coefficient_sha256':sha256(args.outdir/'mean_beta.pt'),
              'component_gradient_units':'Norms in tensor-coefficient coordinates; not full-weight gradient norms.',
              'component_gradient_max': gradient_max, 'fixed_tensors_exact': True,
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
              'test_data_read': False, 'test_inference_performed': False}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', args.condition, 'fixed average BAcc', result['average_validation_macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
