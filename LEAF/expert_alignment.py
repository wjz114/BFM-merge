"""EEG adaptation helpers for Expert Merging's functional-alignment objective.

Reference: ICLR 2026 Expert Merging; official code commit
0670bb957d8a353c74dbb73d61b2e99d43112a39. This adaptation uses LEAF's
28 transformer-layer outputs, fixed class prototypes, and existing groups.
It does not implement Expert Merging++ importance-guided chunking.
"""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256


def hidden_layer_names(leaf):
    return [f'{prefix}.layers.{i}'
            for prefix, module in [('tower.mask_transformer', leaf.tower.mask_transformer),
                                   ('tower.next_transformer', leaf.tower.next_transformer),
                                   ('qformer', leaf.qformer)]
            for i in range(len(module.layers))]


@contextmanager
def capture_hidden(leaf, names):
    """Capture ordinary layer outputs without changing values or gradient flow."""
    modules = dict(leaf.named_modules())
    captured, handles = {}, []
    try:
        for name in names:
            def hook(module, inputs, output, name=name):
                if not torch.is_tensor(output):
                    raise TypeError(f'{name}: expected a tensor hidden state')
                captured[name] = output
            handles.append(modules[name].register_forward_hook(hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def expert_alignment_terms(student, teacher, student_hidden, teacher_hidden, prototypes, coefficients):
    """Unweighted KL(T=1), mean layer MSE, and mean absolute uniform-anchor loss.

    No ground-truth label enters this objective. The caller sets the three
    weights and records all terms. KL differs from soft-target CE only by the
    teacher entropy, which has zero student gradient.
    """
    if set(student_hidden) != set(teacher_hidden) or not student_hidden:
        raise ValueError('Student and teacher hidden-layer sets must match and be nonempty')
    with torch.autocast(device_type=student.device.type, enabled=False):
        student_logits = F.normalize(student.float(), dim=-1) @ prototypes.float().T
        teacher_logits = F.normalize(teacher.detach().float(), dim=-1) @ prototypes.float().T
        kl = F.kl_div(student_logits.log_softmax(-1), teacher_logits.softmax(-1), reduction='batchmean')
        layer_mse = {name: F.mse_loss(value.float(), teacher_hidden[name].detach().float())
                     for name, value in student_hidden.items()}
        hidden = torch.stack(list(layer_mse.values())).mean()
        anchor = (coefficients.float() - 1.0 / coefficients.shape[0]).abs().mean()
    return kl, hidden, anchor, layer_mse


@torch.no_grad()
def cached_hidden_teacher(template, expert_path, dataset, x, y, sample_indices,
                          instruction, prototypes, base_hash, cache_dir, batch_size, device):
    """Cache all token/query states on the exact existing train calibration IDs."""
    expert_path = Path(expert_path)
    run = json.loads(expert_path.with_name('run.json').read_text())
    if 'cpsc' in run:
        raise ValueError('This adapted baseline uses original, unrotated experts')
    names = hidden_layer_names(template)
    metadata = {
        'dataset': dataset, 'sample_id': [f'{dataset}:train:{int(i)}' for i in sample_indices],
        'expert_hash': sha256(expert_path), 'base_hash': base_hash,
        'eeg_hash': array_hash(x), 'labels_hash': array_hash(y),
        'instruction_hash': array_hash(instruction), 'prototypes_hash': array_hash(prototypes),
        'preprocessing_hash': sha256(Path(__file__).with_name('load_datasets.py')),
        'forward_code_hash': sha256(Path(__file__).with_name('LEAF.py')),
        'cache_code_hash': sha256(Path(__file__)), 'layers': names,
        'model_config': template.config.to_dict(),
        'precision': 'fp32_no_tf32_eval', 'coordinate_frame': 'original_expert',
    }
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'expert_alignment_{dataset}_{fingerprint}.pt'
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if cached['metadata'] != metadata:
            raise ValueError(f'Hidden teacher cache metadata mismatch: {path}')
    else:
        # Deepcopy avoids random model initialization; slicing avoids DataLoader RNG draws.
        teacher = copy.deepcopy(template).eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(expert_path, map_location=device, weights_only=True), strict=True)
        hidden_batches = {name: [] for name in names}
        embedding_batches = []
        with capture_hidden(teacher, names) as hidden:
            with torch.autocast(device_type=device.type, enabled=False):
                for start in range(0, len(x), batch_size):
                    hidden.clear()
                    batch = x[start:start + batch_size].to(device)
                    embedding, _ = teacher(batch, instruction[None].expand(len(batch), -1))
                    if set(hidden) != set(names):
                        raise RuntimeError('Not all requested teacher layers executed')
                    embedding_batches.append(F.normalize(embedding.float(), dim=-1).cpu())
                    for name in names:
                        hidden_batches[name].append(hidden[name].float().cpu())
        cached = {'metadata': metadata, 'embedding': torch.cat(embedding_batches),
                  'hidden': {name: torch.cat(batches) for name, batches in hidden_batches.items()}}
        torch.save(cached, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        del teacher
    tensors = (cached['embedding'], *(cached['hidden'][name] for name in names))
    if any(len(tensor) != len(x) or not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError('Hidden teacher cache has invalid sample counts or values')
    provenance = {'path': str(path.resolve()), 'fingerprint': fingerprint, 'layers': names,
                  'coordinate_frame': 'original_expert', 'storage_bytes': path.stat().st_size,
                  'shapes': {name: list(cached['hidden'][name].shape) for name in names}}
    return tensors, provenance
