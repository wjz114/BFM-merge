"""Cache native features immediately before/after the final projection on fixed training rows."""
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
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, prototype_logits, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


def selected_eeg(handle, rows):
    output = np.empty((len(rows), *handle[rows[0][0]].shape[1:]), dtype=np.float32)
    for key in dict.fromkeys(row[0] for row in rows):
        positions = np.asarray([i for i, row in enumerate(rows) if row[0] == key])
        indices = np.asarray([rows[i][1] for i in positions])
        unique, inverse = np.unique(indices, return_inverse=True)
        output[positions] = handle[key][unique][inverse]
    return output


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
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    models, snapshots = {}, {}
    for label, spec in plan['models'].items():
        assert sha256(Path(spec['checkpoint'])) == spec['sha256']
        model = load_leaf(config, Path(spec['checkpoint'])).to(device).eval().requires_grad_(False)
        models[label] = model
        snapshots[label] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    teacher = load_leaf(config, Path(plan['models']['initial']['checkpoint'])).to(device).eval().requires_grad_(False)
    records, forward_examples = {}, 0
    for task in plan['tasks']:
        task_started = time.time()
        ref = plan['reference_records'][task]
        assert sha256(Path(ref['path'])) == ref['sha256']
        cached = torch.load(ref['path'], map_location='cpu', weights_only=True)
        ids = torch.cat([cached['original_training_ids'], cached['replacement_training_ids']]).numpy()
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            rows, truth = row_map(handle, task, 'calibration')
            assert array_hash(np.asarray(truth, dtype=np.int64)) == coverage['audits'][task]['train_labels_hash']
            selected_rows = [rows[int(i)] for i in ids]
            x = torch.from_numpy(selected_eeg(handle, selected_rows))
            y = torch.from_numpy(np.asarray(truth[ids], dtype=np.int64))
        assert array_hash(x) == ref['subset_eeg_hash']
        assert array_hash(y) == ref['subset_labels_hash'] and torch.equal(y, cached['truth'])
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        expert = plan['experts'][task]
        assert sha256(Path(expert['checkpoint'])) == expert['sha256']
        expert_state = torch.load(expert['checkpoint'], map_location='cpu', weights_only=True)
        teacher.load_state_dict(expert_state, strict=True)
        features, logits, replay, head_states = {}, {}, {}, {}
        for label, model in {'teacher': teacher, **models}.items():
            wrapper = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
            head = model.qformer.output_embed[2]
            assert isinstance(head, torch.nn.Linear) and head.in_features == 2048 and head.out_features == 768
            captured = {}

            def hook(module, arguments, output):
                captured['prehead'] = arguments[0].detach()
                captured['raw'] = output.detach()

            handle = head.register_forward_hook(hook)
            saved = {name: [] for name in ['prehead', 'raw', 'unit', 'logits']}
            try:
                for start in range(0, len(x), plan['batch_size']):
                    prediction = wrapper(x[start:start+plan['batch_size']].to(device))
                    prehead, raw = captured['prehead'], captured['raw']
                    reproduced = F.linear(prehead, head.weight, head.bias)
                    assert torch.equal(raw, reproduced)
                    unit = F.normalize(raw, dim=-1)
                    rebuilt = prototype_logits(unit, wrapper.prototypes)
                    assert torch.equal(prediction, rebuilt)
                    for name, value in [('prehead', prehead), ('raw', raw), ('unit', unit), ('logits', prediction)]:
                        assert torch.isfinite(value).all()
                        saved[name].append(value.cpu())
                    forward_examples += len(prediction)
            finally:
                handle.remove()
            saved = {name: torch.cat(values) for name, values in saved.items()}
            logits[label] = saved.pop('logits')
            features[label] = saved
            replay[label] = compare(cached['logits'][label], logits[label], y)
            assert replay[label]['max_abs_logit_error'] < 2e-5
            assert all(row['margin_within_twice_error'] for row in replay[label]['changed_rows'])
            head_states[label] = {'weight': head.weight.detach().cpu().clone(), 'bias': head.bias.detach().cpu().clone()}
        assert all(torch.equal(v.detach().cpu(), expert_state[n]) for n,v in teacher.state_dict().items())
        assert all(p.grad is None for m in [teacher, *models.values()] for p in m.parameters())
        path = args.outdir/f'{task}.pt'
        torch.save({'truth': y, 'n_original': cached['n_original'], 'original_training_ids': cached['original_training_ids'],
                    'replacement_training_ids': cached['replacement_training_ids'], 'features': features,
                    'logits': logits, 'head_states': head_states, 'prototypes': wrapper.prototypes.cpu()}, path)
        records[task] = {'path': str(path), 'sha256': sha256(path), 'native_logit_replay': replay,
                         'subset_eeg_hash': array_hash(x), 'subset_labels_hash': array_hash(y),
                         'exact_head_reconstruction': True, 'teacher_state_unchanged': True,
                         'selected_pool_overlap': len(set(cached['original_training_ids'].tolist()) & set(cached['replacement_training_ids'].tolist())),
                         'seconds': time.time()-task_started}
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'cached', len(x), 'rows x4models', 'max logit replay error',
              max(row['max_abs_logit_error'] for row in replay.values()), flush=True)
        del expert_state, cached, x, y, features, logits, head_states
    assert all(torch.equal(value.detach().cpu(), snapshots[label][name])
               for label, model in models.items() for name,value in model.state_dict().items())
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': plan['tasks'], 'records': records, 'main_forward_examples': forward_examples,
              'optimizer_updates': 0, 'backward_passes': 0, 'all_model_states_unchanged': True,
              'validation_inference_performed': False, 'test_data_read': False, 'seconds': time.time()-started,
              'scope': plan['scope']}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed preprojection representation cache', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
