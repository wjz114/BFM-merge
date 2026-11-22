"""Teacher cache and losses shared by the aligned-merging experiments."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256


def array_hash(value):
    value = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()


def distillation_loss(embedding, teacher, prototypes, kind):
    """FP32, per-example mean; teachers cannot receive gradients."""
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        student = F.normalize(embedding.float(), dim=-1)
        target = F.normalize(teacher.detach().float(), dim=-1)
        if kind == 'semkd':
            return (1 - (student * target).sum(-1)).mean()
        if kind == 'logitkd':
            return F.kl_div((student @ prototypes.T).log_softmax(-1),
                            (target @ prototypes.T).softmax(-1), reduction='batchmean')
        return student.new_zeros(())


@torch.no_grad()
def cached_teacher(template, expert_path, ds, x, y, sample_indices, instruction,
                   prototypes, base_hash, cache_dir, batch_size, device):
    """Store original coordinates; apply CPSC only when constructing the target.

    IDs are train-split ordinals, bound to exact EEG/label payload and loader
    source hashes. Cache mismatches create distinct content-addressed files.
    No DataLoader iterator or model initialization consumes training RNG here.
    """
    expert_path = Path(expert_path)
    run = json.loads(expert_path.with_name('run.json').read_text())
    original_path = Path(run.get('cpsc', {}).get('source_expert', expert_path))
    metadata = {
        'dataset': ds, 'sample_id': [f'{ds}:train:{i}' for i in sample_indices],
        'expert_hash': sha256(original_path), 'base_hash': base_hash,
        'instruction_hash': array_hash(instruction), 'prototypes_hash': array_hash(prototypes),
        'eeg_hash': array_hash(x), 'labels_hash': array_hash(y),
        'preprocessing_hash': sha256(Path(__file__).with_name('load_datasets.py')),
        'forward_code_hash': sha256(Path(__file__).with_name('LEAF.py')),
        'cache_code_hash': sha256(Path(__file__)),
        'precision': 'fp32_no_tf32_eval', 'coordinate_frame': 'original_expert',
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{ds}_{key}.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['metadata'] != metadata:
            raise ValueError(f'Teacher cache metadata mismatch: {path}')
        original = cache['original_embedding']
    else:
        teacher = copy.deepcopy(template).eval()
        teacher.load_state_dict(torch.load(original_path, map_location=device, weights_only=True), strict=True)
        outputs = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            z, _ = teacher(batch, instruction[None].expand(len(batch), -1))
            outputs.append(F.normalize(z.float(), dim=-1).cpu())
        original = torch.cat(outputs)
        torch.save({'metadata': metadata, 'original_embedding': original}, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        del teacher
    target = original
    provenance = {'path': str(path.resolve()), 'fingerprint': key, 'coordinate_frame': 'original_expert'}
    if 'cpsc' in run:
        rotation_path = expert_path.parent.parent / 'rotations.npz'
        rotation = torch.from_numpy(np.load(rotation_path)[ds]).float()
        target = original @ rotation.T
        provenance.update(coordinate_frame='cpsc_expert', rotation_hash=array_hash(rotation),
                          eta=run['cpsc']['eta'], folded_expert_hash=sha256(expert_path))
    return target.detach(), provenance
