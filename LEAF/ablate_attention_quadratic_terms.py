"""Fixed training-only end-to-end ablation of attention's quadratic delta terms."""
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
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_attention_activation_effects import attend
from diagnose_attention_operator_geometry import factors, operators, quadratic
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    started = time.time()
    root = Path('runs/paper_merge_v2')
    out = root/'seed0/attention_quadratic_training_ablation'
    out.mkdir(exist_ok=False)
    device = configure(0, 0)
    gp = root/'seed0/local_gradient_cache/run.json'
    geometry_path = root/'analysis/attention_operator_geometry.json'
    observed_path = root/'seed0/attention_activation_diagnostic/run.json'
    gradient, geometry = [json.loads(p.read_text()) for p in [gp, geometry_path]]
    plan = gradient['plan']
    hashes = {**plan['source_sha256'], **plan['input_sha256'], **geometry['input_sha256']}
    for path, expected in hashes.items():
        assert sha256(Path(path)) == expected, path
    hashes.update({str(p): sha256(p) for p in [gp, geometry_path, observed_path, Path(__file__),
                                             Path('diagnose_attention_activation_effects.py'), Path('diagnose_attention_operator_geometry.py')]})
    source = json.loads(Path(plan['source_fit']).read_text())
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    names = geometry['modules']
    variants = {}
    for name in names:
        bf, mf = factors(base, name), factors(initial, name)
        op = operators(mf)
        quad = quadratic({k: mf[k]-bf[k] for k in bf})
        by_mode = {'operator_replay': {**op, 'ob': mf['ob']}}
        for mode, remove in [('remove_qk', ['qk']), ('remove_vo', ['vo']), ('remove_both', ['qk', 'vo'])]:
            by_mode[mode] = {**{k: v-(quad[k] if k in remove else 0) for k, v in op.items()}, 'ob': mf['ob']}
        variants[name] = {mode: {k: v.float().to(device) for k, v in row.items()} for mode, row in by_mode.items()}
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    current_mode = 'native'

    def hook(name):
        def replace(module, args, kwargs, output):
            if current_mode == 'native':
                return None
            assert kwargs.get('key_padding_mask') is None and not kwargs.get('is_causal', False)
            score, probability, value = attend(*args, variants[name][current_mode], kwargs.get('attn_mask'))
            if not kwargs.get('need_weights', True):
                weights = None
            elif kwargs.get('average_attn_weights', True):
                weights = probability.mean(1)
            else:
                weights = probability
            return value, weights
        return replace

    handles = [dict(leaf.named_modules())[n].register_forward_hook(hook(n), with_kwargs=True) for n in names]
    modes = ['native', 'operator_replay', 'remove_qk', 'remove_vo', 'remove_both']
    design = {'fixed_unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'modes': modes, 'removal_strength': 1, 'modules': names, 'examples_per_task': 512,
              'optimizer_updates': 0, 'validation_inference_planned': False,
              'scope': 'End-to-end intervention on the original learned merger, all32 attention modules, with nonlinear attention softmax retained. QK/VO operators remove only the exact quadratic term in the factor delta about the released base. This is a fixed training-only diagnostic, not a native-rank checkpoint or new algorithm. Replay verifies the full composed implementation. No scale/layer/rank sweep is included. Lower calibration loss is only a possible reason for further native-rank testing, not evidence of generalization.'}
    (out/'design.json').write_text(json.dumps(design, indent=2)+'\n')
    indices = np.load(plan['calibration'])
    records = {}
    for task in source['datasets']:
        task_start = time.time()
        expected = gradient['records'][task]
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            rows, full_truth = row_map(h, task, 'calibration')
            selected = [rows[int(i)] for i in indices[task]]
            memory = {key: h[key][:] for key in dict.fromkeys(row[0] for row in selected)}
        x = torch.from_numpy(read_rows(memory, selected))
        truth = torch.from_numpy(np.asarray(full_truth, dtype=np.int64)[indices[task]])
        del memory, rows, full_truth
        assert len(truth) == 512 and array_hash(x) == expected['eeg_hash'] and array_hash(truth) == expected['labels_hash']
        labels, instruction_text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and instruction_text == source['instruction'][task]
        assert array_hash(instruction) == expected['instruction_hash'] and array_hash(prototypes) == expected['prototypes_hash']
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        weights = torch.tensor(source['ce_class_weights'][task])
        logits, evaluations = {}, {}
        for mode in modes:
            current_mode = mode
            prediction = torch.cat([wrapper(x[i:i+128].to(device)).cpu() for i in range(0, len(x), 128)])
            logits[mode] = prediction
            evaluations[mode] = {**classification_metrics(prediction, truth),
                                 'weighted_ce': float((F.cross_entropy(prediction, truth, reduction='none')*weights[truth]).mean()),
                                 'prediction_changes_from_native': int((prediction.argmax(-1) != logits['native'].argmax(-1)).sum())}
        assert evaluations['native']['BAcc'] == expected['calibration_metrics']['BAcc']
        assert abs(evaluations['native']['weighted_ce']-expected['weighted_ce']) < 1e-6
        check = compare(logits['native'], logits['operator_replay'], truth)
        assert check['max_abs_logit_error'] < 1e-5
        assert all(row['margin_within_twice_error'] for row in check['changed_rows'])
        records[task] = {'evaluations': evaluations, 'operator_replay_check': check,
                         'eeg_hash': expected['eeg_hash'], 'labels_hash': expected['labels_hash'],
                         'seconds': time.time()-task_start}
        (out/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, json.dumps({mode: {k: evaluations[mode][k] for k in ['BAcc', 'weighted_ce']} for mode in modes}), flush=True)
    for handle in handles:
        handle.remove()
    assert all(torch.equal(v.cpu(), initial[n]) for n, v in leaf.state_dict().items())
    aggregate = {mode: {key: float(np.mean([r['evaluations'][mode][key] for r in records.values()]))
                        for key in ['BAcc', 'weighted_ce']} for mode in modes}
    contrasts = {}
    for mode in modes[2:]:
        changes = {task: {k: r['evaluations'][mode][k]-r['evaluations']['native'][k] for k in ['BAcc', 'weighted_ce']} for task, r in records.items()}
        delta = np.array([r['BAcc'] for r in changes.values()])
        loss_delta = np.array([r['weighted_ce'] for r in changes.values()])
        contrasts[mode] = {'per_task': changes, 'macro_bacc_delta_pp': float(delta.mean()*100),
                           'macro_weighted_ce_delta': float(loss_delta.mean()),
                           'tasks_bacc_improved': int((delta > 0).sum()), 'tasks_bacc_worse': int((delta < 0).sum()),
                           'tasks_ce_improved': int((loss_delta < 0).sum()), 'tasks_ce_worse': int((loss_delta > 0).sum()),
                           'task_deletion_bacc_pp': [float((delta.sum()-d)/(len(delta)-1)*100) for d in delta]}
    result = {'design': design, 'design_sha256': sha256(out/'design.json'), 'records': records,
              'aggregate': aggregate, 'contrasts': contrasts, 'model_state_exactly_unchanged': True,
              'native_calibration_bacc_exactly_replayed': True, 'forward_examples': 8192*len(modes),
              'optimizer_updates': 0, 'validation_inference_performed': False, 'test_data_read': False,
              'checkpoint_exported': False, 'seconds': time.time()-started}
    with (out/'run.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'aggregate': aggregate, 'contrasts': {m: {k: v for k, v in row.items() if k not in ['per_task', 'task_deletion_bacc_pp']} for m, row in contrasts.items()}, 'seconds': result['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
