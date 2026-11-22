"""Independent gradient-selection algebra and real EEG integration checks."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import load_leaf, sha256, PrototypeClassifier
from run_streaming_conditional_factor import save_tensor_file
from run_streaming_strict_history import strict_settings, write_json, train_stage
from streaming_bilateral_merge import StreamingMerger
from streaming_gradient_selection import (SelectionData, select_arriving, gradient_geometry,
                                           choose_signs, align_shared_rows)


def independent_objective_audit(payload):
    g = payload['gradients'].double()
    scale = (g.square().sum((0, 1)) / (g.shape[0] * g.shape[1])).sqrt() + 1e-8
    assert torch.allclose(scale, payload['scale'], atol=1e-15, rtol=1e-14)
    selected = torch.stack([g[0 if s > 0 else 1, i] for i, s in enumerate(payload['selected_signs'])])
    residual = (selected.mean(0) - g.mean((0, 1))) / scale
    direct = float(residual.square().sum())
    signs = payload['selected_signs']
    quadratic = float(signs @ payload['gram'] @ signs) / g.shape[1] ** 2
    assert abs(direct - quadratic) <= 1e-10 * max(direct, 1.)
    return {'direct_gradient_mean_squared_error': direct, 'gram_squared_error': quadratic,
            'absolute_error': abs(direct - quadratic)}


def geometry_check():
    generator = torch.Generator().manual_seed(5327)
    g = torch.randn(2, 8, 31, generator=generator)
    _, gram = gradient_geometry(g, 1e-8)
    initial = torch.ones(8, dtype=torch.float64)
    result, trace = choose_signs(gram, initial, 64)
    current = initial.clone()
    for k in trace['flips']:
        alternatives = current.repeat(8, 1)
        alternatives[torch.arange(8), torch.arange(8)] *= -1
        direct = torch.einsum('bi,ij,bj->b', alternatives, gram, alternatives)
        assert int(direct.argmin()) == k
        current = alternatives[k]
    assert torch.equal(current, result)
    assert trace['objectives'][-1] <= trace['objectives'][0]
    zero, audit = choose_signs(torch.zeros(8, 8, dtype=torch.float64), initial, 64)
    assert torch.equal(zero, initial) and not audit['flips']
    original = np.array([0, 1, 2, 3]); fresh = np.array([4, 0, 5, 2]); labels = np.array([0, 0, 1, 1, 0, 1])
    aligned = align_shared_rows(original, fresh, labels)
    for bits in range(16):
        mask = np.array([bool(bits & (1 << k)) for k in range(4)])
        assert len(np.unique(np.where(mask, aligned, original))) == 4
    return {'independent_exhaustive_flip_comparisons_pass': True,
            'zero_gradient_case_pass': True, 'overlap_alignment_all_masks_pass': True}


def main():
    root = Path('runs/streaming_gradient_selection_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_gradient_selection_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    geometry = geometry_check()
    plan = json.loads((root / 'engineering_plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    records, stage2_gradients = {}, None
    extra_probe_error = None
    for condition in ('gradient_match', 'random_blocks'):
        data = SelectionData(plan, source, device)
        leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        records[condition] = []
        for count, task in enumerate(tasks):
            data.arrive(task)
            assert set(data.train) == set(tasks[:count + 1])
            assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
            incoming = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
            if count == 0:
                previous = incoming
                leaf.load_state_dict(previous, strict=True)
                continue
            original_probe = data.train[task][0][:32].clone()
            before = {n: v.detach().cpu().clone() for n, v in leaf.state_dict().items()}
            payload, selection = select_arriving(leaf, base, previous, incoming, count, data,
                                                 task, plan, condition, device)
            assert all(torch.equal(v.detach().cpu(), before[n]) for n, v in leaf.state_dict().items())
            objective = independent_objective_audit(payload)
            assert selection['selected_labels_hash'] == data.records[task]['calibration_labels_hash']
            assert selection['selected_eeg_hash'] == data.records[task]['calibration_eeg_hash']
            assert selection['selected_objective'] <= selection['initial_objective'] + 1e-8
            if count == 1:
                if stage2_gradients is None:
                    stage2_gradients = payload['gradients'].clone()
                    merger = StreamingMerger(leaf, base, previous, incoming, {}, count, plan['active'], 'old_rows', {})
                    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
                    truth = data.train[task][1][:32].to(device)
                    logits = merger.logits(wrapper, original_probe.to(device))
                    loss = (F.cross_entropy(logits, truth, reduction='none') * data.weights[task][truth]).mean()
                    gradient, = torch.autograd.grad(loss, merger.coefficients)
                    expected = payload['gradients'][0, :2].mean(0)
                    extra_probe_error = float((gradient.cpu() - expected).abs().max())
                    assert torch.allclose(gradient.cpu(), expected, atol=2e-7, rtol=2e-4)
                    del merger, wrapper, logits, loss, gradient
                else:
                    assert torch.equal(stage2_gradients, payload['gradients'])
            save_tensor_file(payload, root / f'engineering_{condition}_stage{count + 1}_selection.pt', root, storage)
            state, coefficients, record = train_stage(leaf, base, previous, incoming, {}, count,
                data, tasks[:count + 1], plan, 'old_rows', device, {})
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            record.update(selection=selection, independent_objective=objective)
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(coefficients, root / f'engineering_{condition}_stage{count + 1}_coefficients.pt', root, storage)
            records[condition].append(record)
            previous = state
            del incoming, payload, state, coefficients, before, original_probe
            gc.collect()
            torch.cuda.empty_cache()
        assert not data.valid
        del leaf, data, previous
        gc.collect()
        torch.cuda.empty_cache()
    frozen = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    assert all([r['schedule_hash'] for r in rows] == [r['schedule_hash'] for r in frozen] for rows in records.values())
    updates = sum(r['updates'] for rows in records.values() for r in rows)
    assert updates == 36
    sources = ['streaming_gradient_selection.py', 'run_streaming_gradient_selection.py', 'check_streaming_gradient_selection.py']
    write_json(root / 'engineering.json', {'passed': True, 'geometry': geometry,
        'coefficient_optimizer_updates': updates, 'training_CE_examples': updates * 4,
        'selection_CE_examples': 6144, 'selection_backward_probes': 384,
        'independent_batch_linearity_examples': 32, 'independent_batch_linearity_backwards': 1,
        'independent_batch_linearity_max_error': extra_probe_error,
        'same_model_scoring_exact_across_arms': True, 'scoring_preserves_native_state': True,
        'retained_rows_unique_and_labels_exact': True, 'paired_schedules_with_frozen_engineering': True,
        'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
        'development_validation_read': False, 'official_test_read': False,
        'records': records, 'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: sha256(Path(name)) for name in sources}})


if __name__ == '__main__':
    main()
