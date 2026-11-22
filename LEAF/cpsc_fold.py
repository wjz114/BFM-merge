"""CPSC: contrast-preserving semantic canonicalisation of LEAF experts.

For each expert k (dataset d = k) with fixed, normalized class prototypes P:

    V = span{p_c - mean(p)},  Q = orthonormal basis of V,  N = basis of its complement
    Y_k = N^T e_k(x),  Y_0 = N^T e_0(x)          on the expert's calibration subset
    R_perp = polar(Y_0 Y_k^T / n + mu I),  mu = eta * ||Y_0 Y_k^T / n||_F / sqrt(m)
    R = Q Q^T + N R_perp N^T

R is orthogonal and the identity on V, so replacing the final projection
(W, b) by (R W, R b) leaves every class-score difference of task k, hence its
softmax / CE / predictions, unchanged for any input and instruction. Only
qformer.output_embed.2.{weight,bias} change; everything else is copied.

Writes <outdir>/<dataset>/{best.ckpt,run.json} in the layout merge_poc.py reads,
plus alignment_diagnostics.json and rotations.npz. Sources are never modified.

Example, from LEAF/:
    python cpsc_fold.py --experts-dir runs/poc16/seed0/experts \
        --calibration runs/poc16/seed0/merge_block/calibration_indices.npz \
        --outdir runs/aligned_merge_v1/seed0/experts_cpsc_eta1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import ROOT, load_leaf, prepare_text, resolve_file, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset

WEIGHT, BIAS = 'qformer.output_embed.2.weight', 'qformer.output_embed.2.bias'


def contrast_basis(prototypes, rtol=1e-10, atol=1e-12):
    """Orthonormal bases (Q, N) of the prototype-contrast space and its complement (fp64)."""
    centered = prototypes - prototypes.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=True)
    rank = int((singular > max(rtol * singular[0], atol)).sum())
    return vt[:rank].T, vt[rank:].T, singular


def regularized_procrustes(y_expert, y_base, eta, threshold=1e-12):
    """argmin_R ||R y_expert - y_base||^2 / n + mu ||R - I||^2 over orthogonal R; rows are samples."""
    m = y_expert.shape[1]
    if m == 0:
        return np.eye(0), 0.0, 'empty complement'
    covariance = y_base.T @ y_expert / len(y_expert)
    scale = np.linalg.norm(covariance) / np.sqrt(m)
    if scale <= threshold:
        return np.eye(m), 0.0, 'zero covariance'
    mu = eta * scale
    u, _, vt = np.linalg.svd(covariance + mu * np.eye(m))
    return u @ vt, mu, None


@torch.inference_mode()
def raw_output(leaf, x, instruction, batch_size, device):
    """Unnormalized final projection in fp32 (no autocast), returned as fp64."""
    out = []
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.asarray(x[i:i + batch_size], dtype=np.float32)).to(device)
        tokens = leaf.tower(batch)
        out.append(leaf.qformer(tokens, instruction.unsqueeze(0).expand(len(batch), -1)).double().cpu())
    return torch.cat(out).numpy()


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--base', type=Path, default=Path('checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'))
    p.add_argument('--experts-dir', type=Path, required=True, help='Contains <dataset>/best.ckpt and run.json')
    p.add_argument('--calibration', type=Path, required=True,
                   help='calibration_indices.npz of the matching merge run (train-split indices per dataset)')
    p.add_argument('--datasets', help='Comma-separated subset; default: every dataset in --calibration')
    p.add_argument('--eta', type=float, default=1.0)
    p.add_argument('--check-samples', type=int, default=512, help='Validation trials for the invariance check')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--gpu', default='0')
    p.add_argument('--outdir', type=Path, required=True, help='New output directory')
    return p.parse_args()


def main():
    started = time.time()
    args = parse_args()
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f'Output directory already exists: {outdir}')
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = build_model_config(load_yaml(resolve_file(args.config, ROOT)))
    base_path = resolve_file(args.base, ROOT / 'checkpoints')
    base_sha = sha256(base_path)
    calibration = np.load(args.calibration)
    names = args.datasets.split(',') if args.datasets else list(calibration.files)

    base = load_leaf(config, base_path).to(device).eval()
    expert = load_leaf(config, base_path).to(device).eval()
    # Folding is exact only if the projection is the last operation before L2 normalization.
    if not isinstance(base.qformer.output_embed[-1], torch.nn.Linear) or WEIGHT not in base.state_dict():
        raise RuntimeError('Expected qformer.output_embed to end with the final nn.Linear')
    base_state = base.state_dict()
    w0 = base_state[WEIGHT].double().cpu().numpy()
    b0 = base_state[BIAS].double().cpu().numpy()

    outdir.mkdir(parents=True)
    diagnostics, rotations = {}, {}
    for ds in names:
        source = (args.experts_dir / ds).resolve()
        run = json.loads((source / 'run.json').read_text())
        if run['datasets'] != [ds] or run['source_checkpoint_sha256'] != base_sha:
            raise ValueError(f'{source}: expected a {ds} expert trained from {base_path}')
        state = torch.load(source / 'best.ckpt', map_location='cpu', weights_only=True)
        expert.load_state_dict(state, strict=True)
        _, _, instruction, prototypes = prepare_text(config, ds)
        instruction = instruction.to(device)
        prototypes = prototypes.double().numpy()
        _, train_x, _, val_x, _, test_x, test_y = load_dataset(ds)
        del test_x, test_y  # never used
        index = calibration[ds]
        cal_x, val_x = train_x[index], val_x[:args.check_samples]
        del train_x

        q, n, proto_singular = contrast_basis(prototypes)
        e_base = unit(raw_output(base, cal_x, instruction, args.bs, device))
        v_expert = raw_output(expert, cal_x, instruction, args.bs, device)
        e_expert = unit(v_expert)
        r_perp, mu, identity_reason = regularized_procrustes(e_expert @ n, e_base @ n, args.eta)
        rotation = q @ q.T + n @ r_perp @ n.T

        wk, bk = state[WEIGHT].double().numpy(), state[BIAS].double().numpy()
        folded = dict(state)
        folded[WEIGHT] = torch.from_numpy(rotation @ wk).to(state[WEIGHT].dtype)
        folded[BIAS] = torch.from_numpy(rotation @ bk).to(state[BIAS].dtype)

        # Invariance on held-out validation trials: original expert vs. the folded copy.
        v_original = raw_output(expert, val_x, instruction, args.bs, device)
        expert.load_state_dict(folded, strict=True)
        v_folded = raw_output(expert, val_x, instruction, args.bs, device)
        logits_o, logits_f = unit(v_original) @ prototypes.T, unit(v_folded) @ prototypes.T
        centered = lambda z: z - z.mean(axis=1, keepdims=True)
        residual = lambda a, b: float(((a - b) ** 2).sum(axis=1).mean())
        delta = lambda w, b: float(np.sqrt(((w - w0) ** 2).sum() + ((b - b0) ** 2).sum()))
        diagnostics[ds] = {
            'classes': len(prototypes), 'rank': q.shape[1], 'complement_dim': n.shape[1],
            'prototype_singular_values': proto_singular.tolist(),
            'calibration_samples': int(len(index)), 'eta': args.eta, 'mu': float(mu),
            'identity_reason': identity_reason,
            'orthogonality_max_abs': float(np.abs(rotation.T @ rotation - np.eye(len(rotation))).max()),
            'RQ_minus_Q_max_abs': float(np.abs(rotation @ q - q).max()),
            'R_perp_minus_I_fro': float(np.linalg.norm(r_perp - np.eye(len(r_perp)))),
            'min_raw_norm_calibration': float(np.linalg.norm(v_expert, axis=1).min()),
            'complement_residual_calibration': {'before': residual(e_expert @ n, e_base @ n),
                                                'after': residual(e_expert @ n @ r_perp.T, e_base @ n)},
            'final_layer_delta_l2': {'before': delta(wk, bk), 'after': delta(rotation @ wk, rotation @ bk)},
            'check_samples': int(len(val_x)),
            'fold_vs_rotated_output_max_abs': float(np.abs(unit(v_folded) - unit(v_original) @ rotation.T).max()),
            'centered_logit_max_abs_diff': float(np.abs(centered(logits_o) - centered(logits_f)).max()),
            'prediction_changes': int((logits_o.argmax(1) != logits_f.argmax(1)).sum()),
        }
        d = diagnostics[ds]
        if max(d['orthogonality_max_abs'], d['RQ_minus_Q_max_abs']) > 1e-8 or d['centered_logit_max_abs_diff'] > 1e-4:
            raise FloatingPointError(f'{ds}: CPSC invariance check failed: {d}')

        (outdir / ds).mkdir()
        torch.save(folded, outdir / ds / 'best.ckpt.tmp')
        (outdir / ds / 'best.ckpt.tmp').replace(outdir / ds / 'best.ckpt')
        run['cpsc'] = {'source_expert': str(source / 'best.ckpt'), 'source_expert_sha256': sha256(source / 'best.ckpt'),
                       'eta': args.eta, 'calibration': str(args.calibration.resolve()),
                       'folded_parameters': [WEIGHT, BIAS], 'command': ' '.join(sys.argv)}
        (outdir / ds / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        rotations[ds] = rotation
        print(f'{ds}: rank={d["rank"]} |R-I|={d["R_perp_minus_I_fro"]:.3f} '
              f'dW {d["final_layer_delta_l2"]["before"]:.2f}->{d["final_layer_delta_l2"]["after"]:.2f} '
              f'logit diff={d["centered_logit_max_abs_diff"]:.1e} pred changes={d["prediction_changes"]}', flush=True)
        del state, folded

    np.savez(outdir / 'rotations.npz', **rotations)
    (outdir / 'alignment_diagnostics.json').write_text(json.dumps(
        {'base_checkpoint_sha256': base_sha, 'eta': args.eta, 'seconds': time.time() - started,
         'datasets': diagnostics}, indent=2) + '\n')
    print(f'Saved {len(names)} folded experts to {outdir}')


if __name__ == '__main__':
    main()
