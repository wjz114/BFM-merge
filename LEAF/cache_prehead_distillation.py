"""Cache native prehead targets on the existing full fitting stream, training EEG only."""
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
from cache_preprojection_representations import selected_eeg
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, prototype_logits, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from prehead_distillation_support import make_target_controls


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    source = json.loads(Path(plan['source_fit']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    reference = json.loads(Path(plan['reference_targets']).read_text())
    schedule = np.load(plan['schedule'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    initial = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    snapshot = {n:v.detach().cpu().clone() for n,v in initial.state_dict().items()}
    teacher = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    records, examples = {}, 0
    for task in plan['tasks']:
        task_started = time.time()
        ti = source['datasets'].index(task)
        ref = reference['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        old = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        ids = old['pool_indices'].numpy()
        stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1)
        positions = torch.from_numpy(np.searchsorted(ids, stream))
        assert np.array_equal(ids[positions], stream)
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            rows, labels_all = row_map(handle, task, 'calibration')
            assert array_hash(np.asarray(labels_all, dtype=np.int64)) == coverage['audits'][task]['train_labels_hash']
            x = torch.from_numpy(selected_eeg(handle, [rows[int(i)] for i in ids]))
            truth = torch.from_numpy(np.asarray(labels_all[ids], dtype=np.int64))
        assert torch.equal(truth, old['truth'])
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        assert array_hash(instruction) == ref['instruction_hash'] and array_hash(prototypes) == ref['prototypes_hash']
        expert_path = Path(source['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        expert = torch.load(expert_path, map_location='cpu', weights_only=True)
        teacher.load_state_dict(expert, strict=True)
        saved, replay = {}, {}
        for name, model in [('initial', initial), ('teacher', teacher)]:
            wrapper = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
            head = model.qformer.output_embed[2]
            assert isinstance(head, torch.nn.Linear) and head.in_features == 2048
            captured = {}
            def hook(module, arguments, output):
                captured['h'], captured['raw'] = arguments[0].detach(), output.detach()
            handle = head.register_forward_hook(hook)
            features, raw, logits = [], [], []
            try:
                for start in range(0, len(x), plan['batch_size']):
                    prediction = wrapper(x[start:start+plan['batch_size']].to(device))
                    h, z = captured['h'], captured['raw']
                    assert torch.equal(z, F.linear(h, head.weight, head.bias))
                    assert torch.equal(prediction, prototype_logits(F.normalize(z, dim=-1), wrapper.prototypes))
                    assert torch.isfinite(h).all() and torch.isfinite(z).all()
                    features.append(h.cpu()); raw.append(z.cpu()); logits.append(prediction.cpu())
                    examples += len(prediction)
            finally:
                handle.remove()
            saved[name] = {'prehead': torch.cat(features), 'raw': torch.cat(raw), 'logits': torch.cat(logits)}
            replay[name] = compare(old[name+'_logits'], saved[name]['logits'], truth)
            assert replay[name]['max_abs_logit_error'] < 2e-5
            assert all(r['margin_within_twice_error'] for r in replay[name]['changed_rows'])
        means, shuffled, shuffle_audit = make_target_controls(saved['teacher']['prehead'], truth, positions,
                                                               len(labels), [0, ti, 20261118])
        assert all(torch.equal(v.detach().cpu(), expert[n]) for n,v in teacher.state_dict().items())
        assert all(p.grad is None for model in [initial,teacher] for p in model.parameters())
        target = {'pool_indices': old['pool_indices'], 'truth': truth,
                  'initial_prehead': saved['initial']['prehead'], 'initial_raw': saved['initial']['raw'],
                  'initial_logits': saved['initial']['logits'],
                  'initial_weight': initial.qformer.output_embed[2].weight.detach().cpu().clone(),
                  'initial_bias': initial.qformer.output_embed[2].bias.detach().cpu().clone(),
                  'teacher_prehead': saved['teacher']['prehead'], 'teacher_logits': saved['teacher']['logits'],
                  'teacher_class_means': means, 'shuffled_pool_positions': shuffled,
                  'prototypes': old['prototypes']}
        assert torch.equal(wrapper.prototypes.cpu(), old['prototypes'])
        path = args.outdir/f'{task}.pt'
        torch.save(target, path)
        records[task] = {'target_path': str(path), 'target_sha256': sha256(path),
            'selected_training_eeg_hash': array_hash(x), 'truth_hash': array_hash(truth),
            'train_eeg_hash': ref['train_eeg_hash'], 'train_labels_hash': ref['train_labels_hash'],
            'instruction_hash': ref['instruction_hash'], 'prototypes_hash': ref['prototypes_hash'],
            'class_weights': ref['class_weights'], 'class_counts': ref['class_counts'],
            'stream_indices_hash': array_hash(stream.astype(np.int64)),
            'fitting_exposures': len(stream), 'unique_fitting_rows': len(torch.unique(positions)),
            'forward_rows_per_model': len(x), 'native_logit_replay': replay,
            'native_head_reconstruction_exact': True, 'teacher_state_unchanged': True,
            'shuffle': shuffle_audit, 'seconds': time.time()-task_started}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'cached', len(x), 'rows x2; replay max', max(v['max_abs_logit_error'] for v in replay.values()),
              'shuffle unchanged exposures', shuffle_audit['unchanged_fitting_exposures'], flush=True)
        del x, old, target, saved, expert
    assert all(torch.equal(v.detach().cpu(), snapshot[n]) for n,v in initial.state_dict().items())
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'datasets': plan['tasks'], 'records': records, 'main_forward_examples': examples,
        'all_model_states_unchanged': True, 'optimizer_updates': 0, 'validation_used_in_statistics': False,
        'validation_eeg_read': False, 'validation_inference_performed': False, 'test_data_read': False,
        'seconds': time.time()-started, 'scope': plan['scope']}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
