"""Cache fixed distillation targets on the exact full-pool repair sample stream."""
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
from coverage_support import training_and_validation
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, prototype_logits, sha256
from load_config import build_model_config, load_yaml


def statistics(features, raw, teacher, truth, class_weights, require_unit_mean=True):
    x = features.double()
    x = torch.cat([x, torch.ones((len(x), 1), dtype=torch.float64)], dim=1)
    raw, teacher = raw.double(), teacher.double()
    target = raw.norm(dim=1, keepdim=True)*F.normalize(teacher, dim=-1)-raw
    weights = class_weights[truth].double()
    if require_unit_mean:
        assert abs(float(weights.mean())-1) < 1e-6
    return {'S': x.T@(weights[:, None]*x)/len(x),
            'F': target.T@(weights[:, None]*x)/len(x),
            'R': target.T@(weights[:, None]*target)/len(x)}


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
    old_cache = json.loads(Path(plan['old_feature_cache']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    raw_reference = json.loads(Path(plan['raw_statistics_cache']).read_text())
    assert raw_reference['datasets'] == source['datasets']
    assert old_cache['datasets'] == source['datasets']
    indices = np.load(plan['calibration'])
    schedule = np.load(plan['schedule'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    initial = {n: v.detach().cpu().clone() for n, v in leaf.state_dict().items()}
    head = leaf.qformer.output_embed[2]
    assert isinstance(head, torch.nn.Linear) and head.out_features == 768
    records = {}
    for task in source['datasets']:
        task_started = time.time()
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        x, y, vx, vy = training_and_validation(task)
        audit = coverage['audits'][task]
        assert array_hash(x) == audit['train_eeg_hash'] and array_hash(y) == audit['train_labels_hash']
        # The existing partition loader reads validation arrays; they are not evaluated or fitted.
        assert array_hash(vx) == audit['validation_eeg_hash'] and array_hash(vy) == audit['validation_labels_hash']
        del vx, vy
        truth = torch.from_numpy(y)
        counts = torch.bincount(truth, minlength=len(labels))
        assert counts.tolist() == audit['train_class_counts']
        class_weights = len(y)/(len(labels)*counts.float())
        stream_ids = schedule['local_indices'][schedule['task'] == source['datasets'].index(task)].reshape(-1).astype(np.int64)
        assert len(stream_ids) == 12800 and stream_ids.min() >= 0 and stream_ids.max() < len(y)
        unique_fit_ids = np.unique(stream_ids)
        # Extra old-calibration rows are forwarded only for replay and have zero fitting multiplicity.
        forward_ids = np.union1d(unique_fit_ids, indices[task].astype(np.int64))
        x, y = x[forward_ids], y[forward_ids]
        truth = torch.from_numpy(y)
        fit_positions = torch.from_numpy(np.searchsorted(forward_ids, stream_ids))
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        captured = {}

        def hook(module, inputs, output):
            captured['features'] = inputs[0].detach()
            captured['raw'] = output.detach()

        handle = head.register_forward_hook(hook)
        features, raw, logits, native_errors = [], [], [], []
        try:
            leaf.load_state_dict(initial, strict=True)
            for xx, _ in loader:
                prediction = wrapper(xx.to(device))
                h, z = captured['features'], captured['raw']
                rebuilt = F.linear(h, head.weight, head.bias)
                assert torch.equal(z, rebuilt)
                rebuilt_logits = prototype_logits(F.normalize(rebuilt, dim=-1), wrapper.prototypes)
                error = float((prediction-rebuilt_logits).abs().max())
                assert error <= 1e-6
                assert torch.equal(prediction.argmax(-1), rebuilt_logits.argmax(-1))
                native_errors.append(error)
                features.append(h.cpu()); raw.append(z.cpu()); logits.append(prediction.cpu())
            features, raw, logits = torch.cat(features), torch.cat(raw), torch.cat(logits)
            expert_path = Path(source['experts'][task])
            assert sha256(expert_path) == plan['expert_sha256'][task]
            leaf.load_state_dict(torch.load(expert_path, map_location='cpu', weights_only=True), strict=True)
            teacher, teacher_logits = [], []
            for xx, _ in loader:
                teacher_logits.append(wrapper(xx.to(device)).cpu())
                teacher.append(captured['raw'].cpu())
            teacher, teacher_logits = torch.cat(teacher), torch.cat(teacher_logits)
        finally:
            handle.remove()
        old_record = old_cache['records'][task]
        assert sha256(Path(old_record['path'])) == old_record['sha256']
        old = torch.load(old_record['path'], map_location='cpu', weights_only=True)
        ids = torch.from_numpy(np.searchsorted(forward_ids, indices[task].astype(np.int64)))
        n = old['n_calibration']
        assert len(ids) == n == 512 and torch.equal(truth[ids], old['truth'][:n])
        initial_replay = compare(logits[ids], old['initial_logits'][:n], truth[ids])
        teacher_replay = compare(teacher_logits[ids], old['teacher_logits_calibration'], truth[ids])
        for replay in [initial_replay, teacher_replay]:
            assert replay['max_abs_logit_error'] <= 1e-5
            assert all(row['margin_within_twice_error'] for row in replay['changed_rows'])
        old_recomputed = statistics(features[ids], raw[ids], teacher[ids], truth[ids], old['class_weights'])
        old_reference = statistics(old['features'][:n], old['raw_output'][:n], old['teacher_raw_calibration'],
                                   old['truth'][:n], old['class_weights'])
        moment_errors = {k: float((old_recomputed[k]-old_reference[k]).norm()/old_reference[k].norm())
                         for k in old_reference}
        assert max(moment_errors.values()) < 5e-5, (task, moment_errors)
        st = statistics(features[fit_positions], raw[fit_positions], teacher[fit_positions],
                        truth[fit_positions], class_weights, require_unit_mean=False)
        assert all(torch.isfinite(v).all() for v in st.values())
        raw_record = raw_reference['records'][task]
        assert sha256(Path(raw_record['path'])) == raw_record['sha256']
        reference = torch.load(raw_record['path'], map_location='cpu', weights_only=True)
        full_replay = {k: float((st[k]-reference[k]).norm()/reference[k].norm()) for k in st}
        assert max(full_replay.values()) < 5e-5, (task, full_replay)
        target_path = args.outdir/f'{task}_targets.pt'
        target = {'pool_indices': torch.from_numpy(forward_ids), 'truth': truth,
                  'initial_raw': raw, 'initial_logits': logits, 'initial_norm': raw.norm(dim=-1, keepdim=True),
                  'teacher_unit': F.normalize(teacher, dim=-1), 'teacher_logits': teacher_logits,
                  'prototypes': wrapper.prototypes.detach().cpu()}
        assert target['initial_norm'].min() > 1e-12
        torch.save(target, target_path)
        del target, reference
        path = args.outdir/f'{task}_statistics.pt'
        torch.save(st, path)
        records[task] = {
            'path': str(path), 'sha256': sha256(path), 'target_path': str(target_path),
            'target_sha256': sha256(target_path), 'full_stream_moment_relative_errors': full_replay,
            'fit_examples': len(stream_ids),
            'unique_fit_examples': len(unique_fit_ids), 'forward_examples': len(forward_ids),
            'full_pool_size': audit['train_size'], 'stream_indices_hash': array_hash(stream_ids),
            'unique_fit_indices_hash': array_hash(unique_fit_ids),
            'fitting_stream_class_counts': torch.bincount(truth[fit_positions], minlength=len(labels)).tolist(),
            'fitting_stream_mean_class_weight': float(class_weights[truth[fit_positions]].mean()),
            'input_dimension': st['S'].shape[0], 'target_squared_norm_mean': float(st['R'].trace()),
            'class_counts': counts.tolist(), 'class_weights': class_weights.tolist(),
            'train_eeg_hash': audit['train_eeg_hash'], 'train_labels_hash': audit['train_labels_hash'],
            'instruction_hash': array_hash(instruction), 'prototypes_hash': array_hash(prototypes),
            'expert_sha256': plan['expert_sha256'][task],
            'initial_fitting_stream_metrics': classification_metrics(logits[fit_positions], truth[fit_positions]),
            'teacher_fitting_stream_metrics': classification_metrics(teacher_logits[fit_positions], truth[fit_positions]),
            'native_head_reconstruction_max_abs_logit_error': max(native_errors),
            'old512_initial_replay': initial_replay, 'old512_teacher_replay': teacher_replay,
            'old512_moment_relative_errors': moment_errors, 'seconds': time.time()-task_started}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'fit stream', len(stream_ids), 'unique', len(unique_fit_ids), 'old512 moment errors', moment_errors, flush=True)
        del x, y, loader, features, raw, logits, teacher, teacher_logits, old, old_recomputed, old_reference, st
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'datasets': source['datasets'], 'records': records, 'seconds': time.time()-started,
        'unique_training_examples': sum(v['unique_fit_examples'] for v in records.values()),
        'fitting_stream_examples': sum(v['fit_examples'] for v in records.values()),
        'student_forward_examples': sum(v['forward_examples'] for v in records.values()),
        'teacher_forward_examples': sum(v['forward_examples'] for v in records.values()),
        'optimizer_updates': 0, 'validation_used_in_statistics': False, 'target_format': 'fixed_initial_norm_teacher_direction_and_native_logits',
        'validation_arrays_read_by_partition_loader': True, 'validation_inference_performed': False,
        'test_data_read': False, 'test_inference_performed': False,
        'scope': 'Cached fixed initial raw outputs/norms and own-expert normalized features/native logits for full-model distillation controls. Raw sufficient statistics replay the prior full-stream artifact. '
                 'The existing1600-update schedule determines fitting rows/multiplicities and original full-pool class weights are retained. '
                 'Both frozen merger and own expert forward each unique fitting row plus original512 replay rows; construction cost is explicit. '
                 'Original512 outputs and sufficient statistics replay from the expanded sample order. '
                 'No validation fitting, checkpoint choice or candidate evaluation; larger data access and changed empirical weights are explicit.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed full-training statistics', result['unique_training_examples'], result['seconds'], flush=True)


if __name__ == '__main__':
    main()
