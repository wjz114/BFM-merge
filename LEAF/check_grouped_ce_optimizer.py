"""Verify parameter partition and Adam LR behavior using one cached gradient."""
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, load_leaf, sha256
from grouped_ce_optimizer import build_optimizer
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    checkpoint = root/'seed0/original_fp32_tensor_balanced_avg/merged.ckpt'
    cache_path = root/'seed0/local_gradient_cache/run.json'
    cache = json.loads(cache_path.read_text())
    rec = cache['records'][cache['datasets'][0]]
    gradient_path = Path(rec['path'])
    assert sha256(gradient_path) == rec['sha256']
    assert sha256(checkpoint) == 'abd0aaf09d606387f12395f6ebc7a9de54c3178699f951a80a3cf413c2c6ee01'
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    models, opts, audits = {}, {}, {}
    for label in ['legacy', 'ratio1', 'ratio10']:
        model = load_leaf(config, checkpoint).eval()
        for name, p in model.named_parameters():
            p.requires_grad_(not name.startswith(UNUSED_PREFIXES))
        models[label] = model
        if label == 'legacy':
            opts[label] = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0)
        else:
            opts[label], audits[label] = build_optimizer(model, 1e-5, 1 if label == 'ratio1' else 10)
    grad = torch.load(gradient_path, map_location='cpu', weights_only=True)
    assert set(grad) == set(cache['parameter_names'])
    initial = {n: p.detach().clone() for n, p in models['legacy'].named_parameters() if p.requires_grad}
    fixed = {n: p.clone() for n, p in models['legacy'].state_dict().items() if n not in initial}
    historical_fast = {id(p) for p in models['ratio10'].tower.tokenizer.parameters()}
    actual_fast = {id(p) for p in opts['ratio10'].param_groups[0]['params']}
    assert historical_fast == actual_fast
    records = []
    for step in range(3):
        for label, model in models.items():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.grad = grad[name].clone()
            opts[label].step()
        states = {label: model.state_dict() for label, model in models.items()}
        assert all(torch.equal(states['legacy'][n], states['ratio1'][n]) for n in states['legacy'])
        assert all(torch.equal(state[n], p) for state in states.values() for n, p in fixed.items())
        parameter_maps = {label: dict(model.named_parameters()) for label, model in models.items()}
        fast_errors = {}
        for name in initial:
            a, b = parameter_maps['legacy'][name], parameter_maps['ratio10'][name]
            sa, sb = opts['legacy'].state[a], opts['ratio10'].state[b]
            assert set(sa) == set(sb)
            assert all(torch.equal(sa[k], sb[k]) for k in sa)
            if name.startswith('tower.tokenizer.'):
                d1 = states['legacy'][name].double()-initial[name].double()
                d10 = states['ratio10'][name].double()-initial[name].double()
                error = float((d10-10*d1).abs().max())
                tolerance = 8*(step+1)*torch.finfo(torch.float32).eps*max(1, float(initial[name].abs().max()))
                assert error <= tolerance, (name, error, tolerance)
                fast_errors[name] = {'max_abs_10x_displacement_error': error,
                                     'rounding_tolerance': tolerance,
                                     'displacement_norm_ratio': float(d10.norm()/d1.norm())}
            else:
                assert torch.equal(states['legacy'][name], states['ratio10'][name])
        records.append({'updates': step+1, 'legacy_vs_ratio1_all_states_exact': True,
                        'legacy_vs_ratio10_slow_parameters_exact': True,
                        'adam_moments_exact_for_identical_gradients': True,
                        'fast_displacement_checks': fast_errors})
    result = {'unix': time.time(), 'source_sha256': {str(p): sha256(p) for p in [
                  Path(__file__), Path('grouped_ce_optimizer.py'), Path('train_grouped_ce_control.py'),
                  Path('train_base_ce_control.py'), checkpoint, cache_path, gradient_path,
                  Path('d_downstream_qformer_ce.py'), Path('LEAF.py'), Path('load_config.py'),
                  Path('configs/LEAF_mpnet.yaml'), Path('merge_poc.py')]},
              'optimizer_audits': audits, 'historical_tokenizer_partition_exact': True,
              'fixed_states_exact': True, 'steps': records, 'seconds': time.time()-started,
              'scope': 'CPU optimizer-only engineering test on disposable model copies and one existing cached training gradient, repeated unchanged three times. No model forward/backward, new labels, validation/test metrics or scientific candidate. The ratio1 split must reproduce legacy Adam bitwise; ratio10 preserves all slow parameters/moments and changes only fast LR within FP32 rounding. The verified training/data/export loop is reused unchanged.'}
    output = root/'analysis/grouped_ce_optimizer_gate.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'seconds': result['seconds'], 'passed': True}))


if __name__ == '__main__':
    main()
