"""Measure training-only centered-logit input Jacobians of frozen models."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


def contrast_basis(k):
    q = torch.zeros(k-1, k, dtype=torch.float64)
    for j in range(1, k):
        q[j-1, :j] = 1/(j*(j+1))**.5
        q[j-1, j] = -j/(j*(j+1))**.5
    assert torch.allclose(q@q.T, torch.eye(k-1, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(q.T@q, torch.eye(k, dtype=torch.float64)-1/k, atol=1e-12, rtol=0)
    return q


def input_jacobian(wrapper, x, basis):
    with torch.no_grad():
        native = wrapper(x).cpu()
    variable = x.detach().requires_grad_(True)
    logits = wrapper(variable)
    projected = logits@basis.T
    gradients = [torch.autograd.grad(projected[:, j].sum(), variable,
                                     retain_graph=j < len(basis)-1)[0].detach().cpu()
                 for j in range(len(basis))]
    return native, logits.detach().cpu(), torch.stack(gradients).permute(1, 0, 2, 3).contiguous()


def energies(teacher, student, operator):
    # Shapes B,(K-1),C,T. Left multiplication acts on physical input channels.
    a, b = teacher.double(), student.double()
    if operator is not None:
        a, b = operator@a, operator@b
    dims = (1, 2, 3)
    return torch.stack([a.square().sum(dims), b.square().sum(dims), (a*b).sum(dims),
                        (a-b).square().sum(dims)], dim=1)


def summarize_energy(rows, weights):
    totals = (rows*weights[:, None]).mean(0)
    a, b, dot, error = totals.tolist()
    assert a > 0 and b > 0 and error >= 0
    return {'teacher_energy': a, 'student_energy': b, 'inner_product': dot,
            'mismatch_energy': error, 'relative_mismatch_norm': (error/a)**.5,
            'student_teacher_norm_ratio': (b/a)**.5, 'cosine': dot/(a*b)**.5}


def engineering_checks(wrapper, x, basis, batch_gradient, projector):
    # Compare batching with independent examples; then check directional derivatives.
    one = []
    for sample in x[:2]:
        variable = sample[None].detach().requires_grad_(True)
        output = (wrapper(variable)@basis.T)[0, 0]
        one.append(torch.autograd.grad(output, variable)[0].detach().cpu()[0])
    separate = torch.stack(one)
    batched = batch_gradient[:2, 0]
    error = float((separate-batched).norm()/batched.norm())
    assert error < 5e-4, error
    v = projector@batched.double()
    scale = x[:2].detach().cpu().double().flatten(1).norm(dim=1)
    v *= (scale/v.flatten(1).norm(dim=1))[:, None, None]
    reference = (batched.double()*v).sum((1, 2))
    direction = v.float().to(x.device)
    rows = []
    with torch.no_grad():
        for step in [1e-4, 5e-5]:
            positive = (wrapper(x[:2]+step*direction)@basis.T)[:, 0].double().cpu()
            negative = (wrapper(x[:2]-step*direction)@basis.T)[:, 0].double().cpu()
            finite = (positive-negative)/(2*step)
            relative = float((finite-reference).norm()/reference.norm())
            assert relative < .1, (step, finite.tolist(), reference.tolist(), relative)
            rows.append({'relative_input_step': step, 'autograd': reference.tolist(),
                         'central_difference': finite.tolist(), 'relative_l2_error': relative})
    return {'batch_independence_relative_error': error, 'directional_derivative_checks': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    source = json.loads(Path(plan['source_fit']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    fresh_run = json.loads(Path(plan['fresh_reference']).read_text())
    head = json.loads(Path(plan['head_cache']).read_text())
    old_ids, fresh_ids = np.load(plan['calibration']), np.load(plan['fresh_indices'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    models, snapshots = {}, {}
    for label, spec in plan['models'].items():
        assert sha256(Path(spec['checkpoint'])) == spec['sha256']
        model = load_leaf(config, Path(spec['checkpoint'])).to(device).eval()
        model.requires_grad_(False)
        models[label] = model
        snapshots[label] = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    teacher = load_leaf(config, Path(plan['models']['initial']['checkpoint'])).to(device).eval()
    teacher.requires_grad_(False)
    tasks = ['MI_OpenBMI', 'EMO_FACED'] if args.smoke else plan['tasks']
    records, forward_examples, backward_example_contrasts = {}, 0, 0
    for task in tasks:
        task_started = time.time()
        index = source['datasets'].index(task)
        labels, instruction_text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and instruction_text == source['instruction'][task]
        old, fresh = old_ids[task], fresh_ids[task]
        assert len(old) == len(fresh) == 512
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            all_rows, full_truth = row_map(h, task, 'calibration')
            full_truth = np.asarray(full_truth, dtype=np.int64)
            assert array_hash(full_truth) == coverage['audits'][task]['train_labels_hash']
            selected_rows = [all_rows[int(i)] for i in np.concatenate([old, fresh])]
            memory = {key: h[key][:] for key in dict.fromkeys(row[0] for row in selected_rows)}
        # Each512 block has unique indices; ADHD may overlap between blocks.
        old_x = read_rows(memory, selected_rows[:512])
        fresh_x = read_rows(memory, selected_rows[512:])
        del memory, all_rows
        assert array_hash(old_x) == coverage['audits'][task]['calibration_eeg_hash']
        assert array_hash(fresh_x) == fresh_run['dataset_audits'][task]['training_pool_eeg_hash']
        truth = full_truth[old]
        assert np.array_equal(truth, full_truth[fresh])
        rng = np.random.default_rng([plan['seed'], index, 20261116])
        count = plan['samples_per_pool']
        positions = np.concatenate([rng.choice(np.flatnonzero(truth == k),
                                               (2 if args.smoke else count//len(labels)+(k < count%len(labels))),
                                               replace=False) for k in range(len(labels))])
        n = len(positions)
        x = torch.from_numpy(np.concatenate([old_x[positions], fresh_x[positions]]))
        y = torch.from_numpy(np.tile(truth[positions], 2))
        del old_x, fresh_x, full_truth
        cov_data = np.load(plan['sensor_covariances'][task]['path'])
        covariance = torch.from_numpy(cov_data['covariance']).double()
        covariance = (covariance+covariance.T)/2
        eig, vectors = torch.linalg.eigh(covariance)
        assert float(eig.min()) >= -1e-10*float(eig.max())
        covariance_root = (vectors*eig.clamp_min(0).sqrt())@vectors.T
        covariance_root /= (torch.trace(covariance)/covariance.shape[0]).sqrt()
        projector = torch.eye(65, dtype=torch.float64)-torch.from_numpy(cov_data['null_projector'])
        assert (projector@projector-projector).norm() < 1e-8
        head_spec = head['records'][task]
        assert sha256(Path(head_spec['path'])) == head_spec['sha256']
        old_reference = torch.load(head_spec['path'], map_location='cpu', weights_only=True, mmap=True)
        expert = plan['experts'][task]
        assert sha256(Path(expert['checkpoint'])) == expert['sha256'] == head_spec['expert_sha256']
        expert_state = torch.load(expert['checkpoint'], map_location='cpu', weights_only=True)
        teacher.load_state_dict(expert_state, strict=True)
        wrappers = {label: PrototypeClassifier(model, instruction, prototypes).to(device).eval()
                    for label, model in {'teacher': teacher, **models}.items()}
        for wrapper in wrappers.values():
            assert array_hash(wrapper.instruction.cpu()) == array_hash(old_reference['instruction'])
            assert array_hash(wrapper.prototypes.cpu()) == array_hash(old_reference['prototypes'])
        basis = contrast_basis(len(labels)).float().to(device)
        predictions = {label: [] for label in wrappers}
        statistics = {label: {kind: [] for kind in ['raw', 'support', 'covariance']} for label in models}
        execution_checks = {label: [] for label in wrappers}
        gate = {}
        for start in range(0, len(x), plan['batch_size']):
            batch = x[start:start+plan['batch_size']].to(device)
            teacher_jacobian = None
            for label, wrapper in wrappers.items():
                native, differentiated, gradient = input_jacobian(wrapper, batch, basis)
                assert torch.isfinite(gradient).all() and gradient.square().sum() > 0
                check = compare(native, differentiated, y[start:start+len(batch)])
                assert check['max_abs_logit_error'] < 2e-5
                assert all(row['margin_within_twice_error'] for row in check['changed_rows'])
                execution_checks[label].append(check)
                predictions[label].append(native)
                if args.smoke and start == 0 and label in ['teacher', 'initial']:
                    gate[label] = engineering_checks(wrapper, batch, basis, gradient, projector)
                if label == 'teacher':
                    teacher_jacobian = gradient
                else:
                    for kind, operator in [('raw', None), ('support', projector), ('covariance', covariance_root)]:
                        statistics[label][kind].append(energies(teacher_jacobian, gradient, operator))
                forward_examples += 2*len(batch)
                backward_example_contrasts += len(batch)*len(basis)
            del teacher_jacobian, gradient
        predictions = {label: torch.cat(rows) for label, rows in predictions.items()}
        statistics = {label: {kind: torch.cat(rows) for kind, rows in values.items()} for label, values in statistics.items()}
        replay = {}
        for label, key in [('initial', 'initial_logits'), ('teacher', 'teacher_logits_calibration')]:
            replay[label] = compare(old_reference[key][positions], predictions[label][:n], y[:n])
            assert replay[label]['max_abs_logit_error'] < 2e-5
            assert all(row['margin_within_twice_error'] for row in replay[label]['changed_rows'])
        split_results = {}
        for pool, select in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            target = y[select]
            counts = torch.bincount(target, minlength=len(labels))
            weights = (n/(len(labels)*counts.double()))[target]
            tlogits = predictions['teacher'][select].double()
            values = {}
            for label in models:
                logits = predictions[label][select].double()
                residual = logits-tlogits
                centered = residual-residual.mean(-1, keepdim=True)
                agree = logits.argmax(-1) == tlogits.argmax(-1)
                values[label] = {
                    'metrics': classification_metrics(logits.float(), target),
                    'balanced_centered_logit_mse': float((weights*centered.square().mean(-1)).mean()),
                    'balanced_teacher_kl': float((weights*(tlogits.softmax(-1)*(tlogits.log_softmax(-1)-logits.log_softmax(-1))).sum(-1)).mean()),
                    'balanced_teacher_top1_agreement': float((weights*agree).mean()),
                    'jacobians': {kind: summarize_energy(rows[select], weights) for kind, rows in statistics[label].items()},
                    'support_jacobian_on_agreeing_examples': summarize_energy(statistics[label]['support'][select][agree], weights[agree]) if agree.any() else None,
                }
            split_results[pool] = {'examples': n, 'class_counts': counts.tolist(),
                                   'teacher_metrics': classification_metrics(tlogits.float(), target), 'models': values}
        assert all(torch.equal(value.detach().cpu(), expert_state[name]) for name, value in teacher.state_dict().items())
        assert all(p.grad is None for wrapper in wrappers.values() for p in wrapper.parameters())
        save_path = args.outdir/f'{task}.pt'
        torch.save({'positions_in512': torch.from_numpy(positions), 'original_training_ids': torch.from_numpy(old[positions]),
                    'replacement_training_ids': torch.from_numpy(fresh[positions]), 'truth': y, 'logits': predictions,
                    'jacobian_energy_columns': ['teacher_energy', 'student_energy', 'inner_product', 'mismatch_energy'],
                    'jacobian_energies': statistics, 'n_original': n}, save_path)
        records[task] = {'path': str(save_path), 'sha256': sha256(save_path), 'splits': split_results,
                         'old512_native_replay': replay, 'native_vs_gradient_execution': execution_checks,
                         'engineering_checks': gate, 'subset_eeg_hash': array_hash(x), 'subset_labels_hash': array_hash(y),
                         'selected_pool_overlap': len(set(old[positions].tolist())&set(fresh[positions].tolist())),
                         'sensor_support_rank': int(round(float(torch.trace(projector)))),
                         'teacher_state_unchanged': True, 'seconds': time.time()-task_started}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'seconds', records[task]['seconds'], 'replacement support mismatch',
              {label: split_results['replacement']['models'][label]['jacobians']['support']['relative_mismatch_norm'] for label in models}, flush=True)
        del old_reference, expert_state, x, y, wrappers
    assert all(torch.equal(value.detach().cpu(), snapshots[label][name])
               for label, model in models.items() for name, value in model.state_dict().items())
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': tasks, 'engineering_smoke': args.smoke, 'records': records,
              'main_forward_examples': forward_examples, 'backward_example_contrasts': backward_example_contrasts,
              'engineering_additional_forwards_and_backwards': args.smoke,
              'seconds': time.time()-started, 'optimizer_updates': 0, 'all_model_states_unchanged': True,
              'validation_inference_performed': False, 'test_data_read': False, 'test_inference_performed': False,
              'scope': plan['scope']}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed input-response diagnostic', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
