"""Observe composed-attention differences on unchanged merger training activations."""
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, prepare_text, sha256
from diagnose_attention_operator_geometry import factors, operators, quadratic
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


def attend(query, key, value, op, mask):
    qa = torch.cat([query, torch.ones_like(query[..., :1])], -1)
    va = torch.cat([value, torch.ones_like(value[..., :1])], -1)
    score = torch.einsum('bqd,hde,bke->bhqk', qa, op['qk'], key)
    if mask is None:
        valid = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device)
    else:
        assert mask.ndim == 2 and not mask.dtype == torch.bool
        valid = torch.isfinite(mask)
        assert torch.equal(mask[valid], torch.zeros_like(mask[valid]))
    centered = (score-(score*valid).sum(-1, keepdim=True)/valid.sum(-1, keepdim=True))*valid
    probability = score.masked_fill(~valid, -float('inf')).softmax(-1)
    output = torch.einsum('bhqk,bke,hde->bqd', probability, va, op['vo'])+op['ob']
    return centered, probability, output


def measurements(reference, candidate):
    score, prob, out = reference
    cs, cp, co = candidate
    tv = (cp-prob).abs().sum(-1)/2
    return {'score_error_sq': float((cs-score).double().square().sum()),
            'score_reference_sq': float(score.double().square().sum()),
            'output_error_sq': float((co-out).double().square().sum()),
            'output_reference_sq': float(out.double().square().sum()),
            'tv_sum': float(tv.double().sum()), 'tv_count': tv.numel(),
            'tv_max': float(tv.max())}


def combine(rows):
    keys = next(iter(rows)).keys()
    sums = {k: (max(r[k] for r in rows) if k.endswith('_max') else sum(r[k] for r in rows)) for k in keys}
    return {**sums,
            'centered_score_relative_l2': math.sqrt(sums['score_error_sq']/max(sums['score_reference_sq'], 1e-30)),
            'output_relative_l2': math.sqrt(sums['output_error_sq']/max(sums['output_reference_sq'], 1e-30)),
            'attention_tv_mean': sums['tv_sum']/sums['tv_count']}


@torch.no_grad()
def main():
    started = time.time()
    root = Path('runs/paper_merge_v2')
    out = root/'seed0/attention_activation_diagnostic'
    out.mkdir(exist_ok=False)
    device = configure(0, 0)
    gp = root/'seed0/local_gradient_cache/run.json'
    geometry_path = root/'analysis/attention_operator_geometry.json'
    gradient, geometry = [json.loads(p.read_text()) for p in [gp, geometry_path]]
    plan = gradient['plan']
    hashes = {**plan['source_sha256'], **plan['input_sha256'], **geometry['input_sha256']}
    for path, expected in hashes.items():
        assert sha256(Path(path)) == expected, path
    hashes.update({str(p): sha256(p) for p in [gp, geometry_path, Path(__file__), Path('diagnose_attention_operator_geometry.py')]})
    source = json.loads(Path(plan['source_fit']).read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    names = geometry['modules']
    mean_f = {n: {k: torch.zeros_like(v) for k, v in factors(base, n).items()} for n in names}
    mean_op = {n: {k: torch.zeros_like(v) for k, v in operators(factors(base, n)).items()} for n in names}
    for task in source['datasets']:
        state = torch.load(source['experts'][task], map_location='cpu', weights_only=True, mmap=True)
        for n in names:
            f = factors(state, n)
            for k, v in f.items():
                mean_f[n][k] += v/len(source['datasets'])
            for k, v in operators(f).items():
                mean_op[n][k] += v/len(source['datasets'])
    variants = {}
    for name in names:
        bf, mf = factors(base, name), factors(initial, name)
        mo = operators(mf)
        mq = quadratic({k: mf[k]-bf[k] for k in bf})
        variants[name] = {'uniform_operator_mean': {**mean_op[name], 'ob': mean_f[name]['ob']},
                          'uniform_factor_mean': {**operators(mean_f[name]), 'ob': mean_f[name]['ob']},
                          'merger_actual': {**mo, 'ob': mf['ob']},
                          'merger_without_quadratic': {**{k: mo[k]-mq[k] for k in mo}, 'ob': mf['ob']}}
        variants[name] = {v: {k: a.float().to(device) for k, a in op.items()} for v, op in variants[name].items()}
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval()
    task_records, current, reconstruction = {}, {}, {}

    def hook(name):
        def observe(module, args, kwargs, output):
            query, key, value = args
            assert kwargs.get('key_padding_mask') is None and not kwargs.get('is_causal', False)
            assert module.batch_first and not module.training
            result = {k: attend(query, key, value, op, kwargs.get('attn_mask')) for k, op in variants[name].items()}
            native_error = float((result['merger_actual'][2]-output[0]).abs().max())
            native_relative = float((result['merger_actual'][2]-output[0]).double().norm()/output[0].double().norm())
            assert native_error < 1e-4 and native_relative < 1e-4, (name, native_error, native_relative)
            reconstruction.setdefault(name, []).append({'max_abs': native_error, 'relative_l2': native_relative})
            current.setdefault(name, {'uniform_factors_vs_operator_mean': [], 'remove_merger_quadratic': []})
            current[name]['uniform_factors_vs_operator_mean'].append(measurements(result['uniform_operator_mean'], result['uniform_factor_mean']))
            current[name]['remove_merger_quadratic'].append(measurements(result['merger_actual'], result['merger_without_quadratic']))
            # Observational hook: do not replace the actual module output.
        return observe

    handles = [dict(leaf.named_modules())[n].register_forward_hook(hook(n), with_kwargs=True) for n in names]
    indices = np.load(plan['calibration'])
    for task in source['datasets']:
        current.clear()
        task_start = time.time()
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            rows, full_truth = row_map(h, task, 'calibration')
            selected = [rows[int(i)] for i in indices[task]]
            memory = {key: h[key][:] for key in dict.fromkeys(row[0] for row in selected)}
        x = torch.from_numpy(read_rows(memory, selected))
        truth = torch.from_numpy(np.asarray(full_truth, dtype=np.int64)[indices[task]])
        del memory, rows, full_truth
        expected = gradient['records'][task]
        assert len(truth) == 512 and array_hash(x) == expected['eeg_hash'] and array_hash(truth) == expected['labels_hash']
        labels, instruction_text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and instruction_text == source['instruction'][task]
        assert array_hash(instruction) == expected['instruction_hash'] and array_hash(prototypes) == expected['prototypes_hash']
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        logits = torch.cat([wrapper(x[i:i+128].to(device)).cpu() for i in range(0, len(x), 128)])
        metrics = classification_metrics(logits, truth)
        assert metrics['BAcc'] == expected['calibration_metrics']['BAcc'], (task, metrics, expected['calibration_metrics'])
        module_records = {n: {mode: combine(rows) for mode, rows in by_mode.items()} for n, by_mode in current.items()}
        modes = next(iter(module_records.values())).keys()
        # Aggregate raw sums, not already derived ratios.
        combined = {mode: combine([row for by_mode in current.values() for row in by_mode[mode]]) for mode in modes}
        task_records[task] = {'examples': len(truth), 'eeg_hash': expected['eeg_hash'], 'labels_hash': expected['labels_hash'],
                              'calibration_metrics': metrics, 'per_module': module_records, 'aggregate': combined,
                              'seconds': time.time()-task_start}
        (out/'progress.json').write_text(json.dumps(task_records, indent=2)+'\n')
        print(task, json.dumps({m: {k: v[k] for k in ['output_relative_l2', 'attention_tv_mean']} for m, v in combined.items()}), flush=True)
    for handle in handles:
        handle.remove()
    assert all(torch.equal(v.cpu(), initial[n]) for n, v in leaf.state_dict().items())
    raw_keys = ['score_error_sq', 'score_reference_sq', 'output_error_sq', 'output_reference_sq', 'tv_sum', 'tv_count', 'tv_max']
    aggregate = {mode: combine([{k: row['aggregate'][mode][k] for k in raw_keys} for row in task_records.values()]) for mode in modes}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': hashes,
              'datasets': source['datasets'], 'records': task_records, 'aggregate': aggregate,
              'native_reconstruction_max_abs': max(v['max_abs'] for values in reconstruction.values() for v in values),
              'native_reconstruction_max_relative_l2': max(v['relative_l2'] for values in reconstruction.values() for v in values),
              'model_state_exactly_unchanged': True, 'old512_bacc_exactly_replayed': True,
              'forward_examples': 8192, 'optimizer_updates': 0,
              'validation_inference_performed': False, 'test_data_read': False,
              'seconds': time.time()-started,
              'scope': 'Local attention maps on all original512 training rows per task at unchanged learned-merger prefixes. Uniform operator means are constructed diagnostics, not teacher-output ensembles or learned targets. Removing quadratic terms is a coordinate-dependent counterfactual, not error removal. Outputs include affine biases and native causal masks. FP32 operators with FP64 norm accumulation; reconstruction checked against every actual native attention output. No outputs are replaced in model execution, no candidate exported, and no generalization or accuracy effect inferred from local norm differences. Aggregate norms weight by activation element count, not equally by task; per-task and per-module values retained.'}
    with (out/'run.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'aggregate': aggregate, 'seconds': result['seconds'], 'native_max_abs': result['native_reconstruction_max_abs']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
