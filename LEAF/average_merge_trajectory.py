"""Export one fixed average of completed merge coefficient states; no fitting."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from check_merge_export_margins import compare
from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, prototype_logits, selection_key, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES, group_for_name


STEPS = [800, 960, 1120, 1280, 1440, 1600]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', help='Engineering smoke subset; default all source tasks')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    source = json.loads((args.source / 'run.json').read_text())
    source_metadata_defaults = {}
    if 'granularity' not in source:
        source['granularity'] = 'block'
        source_metadata_defaults['granularity'] = 'block (original legacy layout)'
    if 'structural_groups' not in source:
        assert source['granularity'] == 'block'
        source['structural_groups'] = source['groups']
        source_metadata_defaults['structural_groups'] = 'copied from original block groups'
    assert source['method'] == 'block' and source['loss'] == 'ce'
    assert source.get('task_vector_preprocessing') is None
    balanced = source.get('class_balanced_ce', False)
    assert source['alpha_updates_actual'] == source['steps_requested'] == 1600
    history = np.load(args.source / 'coefficient_history.npz')
    assert history['step'].tolist() == list(range(0, 1601, 160))
    rows = [history['step'].tolist().index(step) for step in STEPS]
    snapshots = torch.tensor(history['alpha'][rows], device=device)
    assert torch.isfinite(snapshots).all()
    # The rule is arithmetic averaging; FP64 reduction avoids avoidable sum error.
    alpha = snapshots.double().mean(0).float()
    expected = history['alpha'][rows].astype(np.float64).mean(0).astype(np.float32)
    assert np.array_equal(alpha.cpu().numpy(), expected)
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    base = {name: value.detach() for name, value in leaf.named_parameters()}
    buffers = dict(leaf.named_buffers())
    selected = [name for name in base if not name.startswith(UNUSED_PREFIXES)]
    if source_metadata_defaults:
        assert source['groups'] == list(dict.fromkeys(group_for_name(name) for name in selected))
    group_ids = {name: source['groups'].index(name if source['granularity'] == 'tensor'
                                           else group_for_name(name)) for name in selected}
    deltas = {name: [] for name in selected}
    for dataset in source['datasets']:
        state = torch.load(source['experts'][dataset], map_location=device, weights_only=True)
        for name in selected:
            deltas[name].append(state[name].float() - base[name].float())
    del state
    params, linear_error = dict(base), 0.0
    for name in selected:
        vectors = torch.stack(deltas.pop(name))
        column = group_ids[name]
        params[name] = base[name] + torch.tensordot(alpha[:, column], vectors, dims=1)
        mean_states = torch.stack([
            base[name] + torch.tensordot(snapshot[:, column], vectors, dims=1)
            for snapshot in snapshots]).double().mean(0)
        linear_error = max(linear_error, (params[name].double() - mean_states).abs().max().item())
    assert linear_error <= 1e-5, linear_error
    state = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
    for name in selected:
        state[name] = params[name].cpu()
    assert all(torch.isfinite(state[name]).all() for name in base)
    for name in [*buffers, *(name for name in base if name.startswith(UNUSED_PREFIXES))]:
        assert torch.equal(state[name], leaf.state_dict()[name].cpu())
    torch.save(state, args.outdir / 'merged.ckpt.tmp')
    (args.outdir / 'merged.ckpt.tmp').replace(args.outdir / 'merged.ckpt')
    np.save(args.outdir / 'best_alpha.npy', alpha.cpu().numpy())
    (args.outdir / 'calibration_indices.npz').write_bytes((args.source / 'calibration_indices.npz').read_bytes())
    check = load_leaf(config, args.outdir / 'merged.ckpt').to(device).eval()
    state_error = max((value - params[name]).abs().max().item() for name, value in check.named_parameters())
    assert state_error == 0, state_error
    setup_seconds = time.time() - started
    names = args.datasets.split(',') if args.datasets else source['datasets']
    assert set(names) <= set(source['datasets']) and len(names) == len(set(names))
    validation, folded, checks = {}, {}, {}
    for dataset in names:
        labels, text, instruction, prototypes = prepare_text(config, dataset)
        assert labels == source['labels'][dataset] and text == source['instruction'][dataset]
        instruction, prototypes = instruction.to(device), prototypes.to(device)
        wrapper = PrototypeClassifier(check, instruction, prototypes).to(device).eval()
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(dataset)
        del train_x, train_y, test_x, test_y
        loader = make_loader(val_x, val_y, config, len(labels), source['batch_size'], False)
        before, after, truth = [], [], []
        for x, y in loader:
            x = x.to(device)
            embedding, _ = functional_call(leaf, (params, buffers),
                                           (x, instruction[None].expand(len(x), -1)))
            before.append(prototype_logits(embedding, prototypes).cpu())
            after.append(wrapper(x).cpu())
            truth.append(y)
        before, after, truth = torch.cat(before), torch.cat(after), torch.cat(truth)
        validation[dataset] = classification_metrics(before, truth)
        folded[dataset] = classification_metrics(after, truth)
        checks[dataset] = compare(before, after, truth)
        assert checks[dataset]['max_abs_logit_error'] <= 1e-5, checks[dataset]
        assert all(row['margin_within_twice_error'] for row in checks[dataset]['changed_rows'])
        print(f'{dataset}: val BAcc={validation[dataset]["BAcc"]:.6f}, '
              f'export error={checks[dataset]["max_abs_logit_error"]:.3g}', flush=True)
    bacc, loss, _ = selection_key(validation)
    folded_bacc, folded_loss, _ = selection_key(folded)
    elapsed = time.time() - started
    metadata = {key: source[key] for key in [
        'base_checkpoint', 'base_checkpoint_sha256', 'experts', 'labels', 'instruction',
        'K', 'groups', 'num_coefficients', 'granularity', 'structural_groups',
        'calibration_input', 'calibration_sizes', 'calibration_per_dataset',
        'merged_tensors', 'unused_tensors_kept_at_base', 'seed', 'batch_size']}
    metadata.update(
        method='trajectory_average',
        loss='no additional fitting; source ' + ('class-balanced CE' if balanced else 'ordinary CE'),
        class_balanced_ce=balanced,
        calibration_class_counts=source.get('calibration_class_counts'),
        ce_class_weights=source.get('ce_class_weights', {}),
        ce_reduction=source.get('ce_reduction', 'ordinary minibatch mean CE'),
        datasets=names, source_run=str(args.source.resolve()),
        source_metadata_defaults=source_metadata_defaults,
        source_run_sha256=sha256(args.source / 'run.json'),
        source_history_sha256=sha256(args.source / 'coefficient_history.npz'),
        source_optimizer_updates=source['alpha_updates_actual'],
        source_optimizer=source['optimizer'], source_lr=source['lr'],
        averaging_steps=STEPS, averaging_weights=[1 / len(STEPS)] * len(STEPS),
        best_alpha=alpha.cpu().tolist(), best_step=None,
        selection='Single fixed arithmetic mean of states at800/960/1120/1280/1440/1600; no selection among averages',
        alpha_updates_actual=0, steps_requested=0, optimizer=None, lr=None,
        best_val_macro_BAcc=bacc, best_val_macro_Loss=loss, best_val=validation,
        folded_val_macro_BAcc=folded_bacc, folded_val_macro_Loss=folded_loss, folded_val=folded,
        export_logit_max_abs={name: row['max_abs_logit_error'] for name, row in checks.items()},
        export_prediction_checks=checks, coefficient_mean_exact=True,
        composed_mean_vs_mean_states_max_abs=linear_error, reloaded_state_max_abs=state_error,
        fixed_base_buffers_and_inactive_exact=True,
        precision='FP32 composition/forward/readout, FP64 arithmetic mean; no autocast or TF32',
        seconds={'source_training': source['seconds']['total'], 'postprocess_setup': setup_seconds,
                 'postprocess_validation': elapsed - setup_seconds, 'postprocess_total': elapsed,
                 'total': source['seconds']['total'] + elapsed},
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
        code_sha256={name: sha256(ROOT / name) for name in [
            'average_merge_trajectory.py', 'check_merge_export_margins.py',
            'merge_poc.py', 'd_downstream_qformer_ce.py', 'LEAF.py']},
        engineering_subset=args.datasets is not None,
    )
    (args.outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Fixed average val macro BAcc={bacc:.8f}; additional seconds={elapsed:.2f}', flush=True)


if __name__ == '__main__':
    main()
