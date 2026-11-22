"""Check saved CPSC experts on calibration/validation under both instruction protocols."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from cpsc_fold import BIAS, WEIGHT, contrast_basis, raw_output, unit
from c_direct_inference import build_instruction_embedding, load_text_embeddings, select_instructions
from d_downstream_qformer_ce import ROOT, load_leaf, prepare_text, sha256
from load_config import build_model_config, load_instructions, load_targets, load_yaml
from load_datasets import load_dataset


def compare(original, folded, rotation, prototypes, labels):
    old, new = torch.from_numpy(unit(original) @ prototypes.T), torch.from_numpy(unit(folded) @ prototypes.T)
    centered = lambda z: z - z.mean(1, keepdim=True)
    margins = lambda z: z.topk(2, dim=-1).values.diff(dim=-1).abs().flatten()
    delta = {
        'centered_logits_max_abs': (centered(old) - centered(new)).abs().max().item(),
        'softmax_max_abs': (old.softmax(-1) - new.softmax(-1)).abs().max().item(),
        'ce_abs': abs(F.cross_entropy(old, torch.as_tensor(labels).long()).item()
                      - F.cross_entropy(new, torch.as_tensor(labels).long()).item()),
        'margin_max_abs': (margins(old) - margins(new)).abs().max().item(),
        'prediction_changes': int((old.argmax(-1) != new.argmax(-1)).sum()),
        'min_original_margin': margins(old).min().item(),
        'fold_vs_rotation_max_abs': float(np.abs(unit(folded) - unit(original) @ rotation.T).max()),
        'raw_norm_relative_max_abs': float(np.max(np.abs(np.linalg.norm(folded, axis=1)
                                          / np.linalg.norm(original, axis=1) - 1))),
        'min_raw_norm': float(np.linalg.norm(original, axis=1).min()),
        'samples': len(labels),
    }
    if max(delta[k] for k in ['centered_logits_max_abs', 'softmax_max_abs', 'ce_abs',
                             'margin_max_abs', 'fold_vs_rotation_max_abs']) > 1e-4:
        raise AssertionError(delta)
    return delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aligned', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--datasets', help='Comma-separated subset; default: all saved rotations')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    base_path = ROOT / 'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'
    base = load_leaf(config, base_path).to(device).eval()
    expert = load_leaf(config, base_path).to(device).eval()
    rotations, indices = np.load(args.aligned / 'rotations.npz'), np.load(args.calibration)
    names = args.datasets.split(',') if args.datasets else rotations.files
    text_cache = load_text_embeddings(config)
    official_texts = select_instructions(load_targets(), load_instructions(), 42)
    results, before_vectors, after_vectors = {}, [], []
    base_final = np.concatenate([base.state_dict()[n].cpu().numpy().reshape(-1) for n in [WEIGHT, BIAS]])
    for ds in names:
        run = json.loads((args.aligned / ds / 'run.json').read_text())
        original_state = torch.load(run['cpsc']['source_expert'], map_location='cpu', weights_only=True)
        folded_state = torch.load(args.aligned / ds / 'best.ckpt', map_location='cpu', weights_only=True)
        rotation = rotations[ds]
        labels, training_text, instruction, prototypes = prepare_text(config, ds)
        official = build_instruction_embedding(official_texts[ds, 2], labels, 2, text_cache, device)
        prototypes = prototypes.double().numpy()
        q, n, _ = contrast_basis(prototypes)
        _, x, y, vx, vy, tx, ty = load_dataset(ds)
        del tx, ty
        splits = {'calibration': (x[indices[ds]][:args.samples], y[indices[ds]][:args.samples]),
                  'validation': (vx[:args.samples], vy[:args.samples])}
        del x, y, vx, vy
        results[ds] = {'training_instruction': training_text, 'official_l2_instruction': official_texts[ds, 2],
                       'rank': q.shape[1], 'orthogonality_error': float(np.abs(rotation.T @ rotation - np.eye(len(rotation))).max()),
                       'protected_space_error': float(np.abs(rotation @ q - q).max()), 'checks': {}}
        for protocol, cond in [('training', instruction.to(device)), ('official_l2_seed42', official)]:
            for split, (x, y) in splits.items():
                expert.load_state_dict(original_state, strict=True)
                original = raw_output(expert, x, cond, 32, device)
                base_z = unit(raw_output(base, x, cond, 32, device))
                expert.load_state_dict(folded_state, strict=True)
                folded = raw_output(expert, x, cond, 32, device)
                check = compare(original, folded, rotation, prototypes, y)
                check['complement_residual_before'] = float(np.square((unit(original) - base_z) @ n).sum(1).mean())
                check['complement_residual_after'] = float(np.square((unit(folded) - base_z) @ n).sum(1).mean())
                results[ds]['checks'][f'{protocol}/{split}'] = check
        for vectors, state in [(before_vectors, original_state), (after_vectors, folded_state)]:
            vectors.append(np.concatenate([state[k].numpy().reshape(-1) for k in [WEIGHT, BIAS]]) - base_final)
        print(f'{ds}: all four protocol/split checks passed', flush=True)
    def angles(vectors):
        vectors = np.stack(vectors).astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        return {'norms': norms.tolist(), 'cosine_matrix': ((vectors @ vectors.T) / np.outer(norms, norms)).tolist()}
    report = {'datasets': results, 'base_sha256': sha256(base_path), 'calibration_sha256': sha256(args.calibration),
              'rotation_sha256': sha256(args.aligned / 'rotations.npz'), 'seconds': time.time() - started,
              'final_layer_vectors_before': angles(before_vectors), 'final_layer_vectors_after': angles(after_vectors),
              'dataset_order': names, 'precision': 'FP32 model, FP64 geometry; TF32/autocast disabled',
              'test_data_used': False}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
