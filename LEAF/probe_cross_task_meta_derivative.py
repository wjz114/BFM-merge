"""Diagnose the failed native finite difference without changing the operator."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, PrototypeClassifier, sha256
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_bilateral_merge import StreamingMerger
from streaming_cross_task_meta import configure_attention, normalized_step, objective_gradient
from streaming_cross_task_meta_training import weighted_loss


def main():
    root = Path('runs/streaming_cross_task_meta_v1')
    assert not (root / 'derivative_probe.json').exists()
    plan = json.loads((root / 'engineering_plan.json').read_text())
    strict_settings()
    device = configure(0, 0)
    configure_attention()
    torch.cuda.set_per_process_memory_fraction(.2, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = ['MI_OpenBMI', 'EMO_FACED']
    for t in tasks:
        data.arrive(t)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    models = [torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks]
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    merger = StreamingMerger(leaf, base, *models, None, 1, plan['active'], 'old_rows', None)
    wrappers = [PrototypeClassifier(leaf, *data.readouts[t]).eval() for t in tasks]
    batches = [(data.train[t][0][:4].to(device), data.train[t][1][:4].to(device)) for t in tasks]
    support = lambda c: weighted_loss(merger, wrappers[0], batches[0], data.weights[tasks[0]], c)
    query = lambda c: weighted_loss(merger, wrappers[1], batches[1], data.weights[tasks[1]], c)
    capture = []
    def pool_hook(module, args, output):
        with torch.no_grad():
            _, ids = F.max_pool2d(args[0], module.kernel_size, module.stride,
                                 module.padding, module.dilation, module.ceil_mode, return_indices=True)
            capture.append(ids.cpu())
    handles = [m.register_forward_hook(pool_hook) for m in leaf.modules() if isinstance(m, torch.nn.MaxPool2d)]
    x = merger.coefficients.detach().clone()
    boundaries = (0, 766, 174216, 347666)
    c = x.clone().requires_grad_(True)
    full, _ = objective_gradient(c, support(c), query, 'cross_task_meta', boundaries, .01, 1e-8)
    nominal = capture[:]
    capture.clear()
    mask = torch.ones_like(full)
    n = len(plan['active'])
    excluded = []
    for j, name in enumerate(plan['active']):
        if 'tokenizer' in name:
            excluded.append(name)
            mask[j] = mask[n + j] = 0
            mask[merger.row_slices[name]] = 0
            mask[merger.retention_slices[name]] = 0
    assert excluded
    records = {}
    for label, vec in [('all_coefficients', full), ('tokenizer_fixed', full * mask)]:
        direction = vec / vec.norm()
        dot = float(full @ direction)
        records[label] = {'analytic': dot, 'steps': []}
        for delta in [.01, .005, .002, .001, .0005, .0002, .0001]:
            values, changes = [], []
            for sign in [-1, 1]:
                c = (x + sign * delta * direction).detach().requires_grad_(True)
                loss = support(c)
                grad = torch.autograd.grad(loss, c)[0]
                value = .5 * (loss.detach() + query(c - normalized_step(grad, boundaries, .01, 1e-8)))
                values.append(float(value.detach()))
                changes.append([int((a != b).sum()) for a, b in zip(capture, nominal)])
                capture.clear()
            finite = (values[1] - values[0]) / (2 * delta)
            records[label]['steps'].append({'delta': delta, 'finite': finite, 'absolute_error': abs(finite - dot),
                'within_original_tolerance': abs(finite - dot) <= 1e-4 + .02 * abs(dot),
                'pool_argmax_changes_support_query': changes})
    for h in handles:
        h.remove()
    write_json(root / 'derivative_probe.json', {'unix': time.time(), 'source_sha256': sha256(Path(__file__)),
        'records': records, 'tokenizer_names': excluded, 'performance_data_read': False,
        'operator_unchanged': True})
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
