"""Fixed FeatCal adaptation to ordinary LEAF blocks using the pinned official solver.

The final all-block checkpoint is primary. Smoke mode is an engineering check
on two datasets and one block per branch, never a performance candidate.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from coverage_support import training_and_validation
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from featcal_support import calibrate_state, paired_statistics
from load_config import build_model_config, load_yaml
from merge_poc import stratified_indices
from regmean_support import advance_layer, capture_layer_inputs


PREFIXES = ['tower.mask_transformer', 'tower.next_transformer', 'qformer']


def layer_state(state, name):
    return {key[len(name)+1:]: value for key, value in state.items() if key.startswith(name+'.')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['source_sha256'].items():
        assert sha256(Path(path)) == expected, path
    for name in ['initial_checkpoint', 'base_checkpoint', 'calibration', 'source_fit', 'coverage_audit']:
        assert sha256(Path(plan[name])) == plan[name+'_sha256'], name
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    initial = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
    base = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=True)
    source = json.loads(Path(plan['source_fit']).read_text())
    reference = json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    indices = np.load(plan['calibration'])
    tasks = plan['smoke_tasks'] if args.smoke else source['datasets']
    train, validation, wrappers, states, audits, expert_sources = {}, {}, {}, {}, {}, {}
    used_ids = {}
    for task in tasks:
        path = Path(source['experts'][task])
        meta = json.loads(path.with_name('run.json').read_text())
        assert meta['datasets'] == [task]
        assert meta['source_checkpoint_sha256'] == plan['base_checkpoint_sha256']
        states[task] = torch.load(path, map_location='cpu', weights_only=True)
        expert_sources[task] = {'path': str(path), 'sha256': sha256(path)}
        labels, text, instruction, prototypes = prepare_text(config, task)
        wrappers[task] = PrototypeClassifier(leaf, instruction.to(device), prototypes.to(device)).to(device).eval()
        x, y, vx, vy = training_and_validation(task)
        data_audit = coverage['audits'][task]
        assert array_hash(x) == data_audit['train_eeg_hash']
        assert array_hash(y) == data_audit['train_labels_hash']
        assert array_hash(vx) == data_audit['validation_eeg_hash']
        assert array_hash(vy) == data_audit['validation_labels_hash']
        ids = indices[task]
        if args.smoke:
            local = stratified_indices(y[ids], plan['smoke_calibration_size'], np.random.default_rng(plan['seed']))
            ids = ids[local]
        assert len(ids) == len(np.unique(ids)) and ids.min() >= 0 and ids.max() < len(y)
        assert set(np.unique(y[ids])) == set(range(len(labels)))
        used_ids[task] = ids
        train[task] = torch.from_numpy(x[ids].copy()), torch.from_numpy(y[ids].copy())
        if args.smoke:
            probe = np.concatenate([np.flatnonzero(vy == c)[:8] for c in range(len(labels))])
            vx, vy = vx[probe].copy(), vy[probe].copy()
        validation[task] = make_loader(vx, vy, config, len(labels), plan['evaluation_batch_size'], False)
        audits[task] = {'calibration_size': len(ids), 'validation_size': len(vy),
                       'calibration_eeg_sha256': array_hash(train[task][0]),
                       'calibration_labels_sha256': array_hash(train[task][1]),
                       'labels': labels, 'instruction': text,
                       'actual_instruction_sha256': array_hash(wrappers[task].instruction),
                       'actual_prototypes_sha256': array_hash(wrappers[task].prototypes)}
        print(task, 'calibration', len(ids), 'validation', len(vy), flush=True)
        del x, y, vx, vy
    np.savez(args.outdir/'calibration_indices.npz', **used_ids)

    def evaluate(calibration=False):
        metrics = {}
        for task in tasks:
            if calibration:
                x, y = train[task]
                loader = [(x[i:i+plan['evaluation_batch_size']], y[i:i+plan['evaluation_batch_size']])
                          for i in range(0, len(x), plan['evaluation_batch_size'])]
            else:
                loader = validation[task]
            logits, truth = [], []
            for x, y in loader:
                logits.append(wrappers[task](x.to(device)).cpu())
                truth.append(y)
            metrics[task] = classification_metrics(torch.cat(logits), torch.cat(truth))
        return metrics

    initial_validation = evaluate()
    if not args.smoke:
        assert all(initial_validation[t]['BAcc'] == reference['folded_val'][t]['BAcc'] for t in tasks)
    initial_calibration = evaluate(True)
    print('initial validation', np.mean([v['BAcc'] for v in initial_validation.values()]), flush=True)
    setup_seconds = time.time()-started

    def first_inputs(names, original_expert):
        result = {name: {} for name in names}
        probe = copy.deepcopy(leaf) if original_expert else leaf
        for task in tasks:
            if original_expert:
                probe.load_state_dict(states[task], strict=True)
            wrapper = wrappers[task]
            x, _ = train[task]
            for name in names:
                result[name][task] = []
            with capture_layer_inputs(probe, names) as captured:
                for start in range(0, len(x), plan['batch_size']):
                    captured.clear()
                    batch = x[start:start+plan['batch_size']].to(device)
                    probe(batch, wrapper.instruction[None].expand(len(batch), -1))
                    assert set(captured) == set(names)
                    for name in names:
                        result[name][task].append(captured[name])
        if original_expert:
            del probe
        return result

    fit_started = time.time()
    first_names = [p+'.layers.0' for p in PREFIXES]
    expert_inputs = first_inputs(first_names, True)
    student_inputs = first_inputs(first_names[:2], False)
    layer_audits, changed_layers = [], []
    for prefix in PREFIXES:
        if prefix == 'qformer':
            student_inputs.update(first_inputs([first_names[2]], False))
        sb = student_inputs.pop(prefix+'.layers.0')
        eb = expert_inputs.pop(prefix+'.layers.0')
        layers = dict(leaf.named_modules())[prefix].layers
        for index, layer in enumerate(layers):
            if args.smoke and index >= plan['smoke_blocks_per_branch']:
                break
            name = f'{prefix}.layers.{index}'
            expert_layer = copy.deepcopy(layer)
            stats, own_states, next_eb, observations = [], [], {}, {}
            for task in tasks:
                own_state = layer_state(states[task], name)
                expert_layer.load_state_dict(own_state, strict=True)
                values, next_eb[task], observations[task] = paired_statistics(
                    layer, expert_layer, sb[task], eb[task], plan['teacher_interpolation_alpha'], device)
                stats.append(values)
                own_states.append(own_state)
            # Every map observes the same unmodified block before this solve/load.
            prior = layer_state(initial, name)
            assert all(torch.equal(value.cpu(), prior[key]) for key, value in layer.state_dict().items())
            calibrated = calibrate_state(prior, layer_state(base, name), own_states, stats,
                                         plan['ridge_lambda'], plan['anchor_rho'], plan['covariance_eps'])
            layer.load_state_dict(calibrated, strict=True)
            next_sb = {task: advance_layer(layer, sb[task], device) for task in tasks}
            # Carry original expert/current student KV and masks without substitution.
            for task in tasks:
                for before, after in zip(sb[task], next_sb[task]):
                    assert len(before[0]) == len(after[0])
                    assert before[1] is after[1]
                    assert all(a is b for a, b in zip(before[0][1:], after[0][1:]))
            audit = {'layer': name, 'maps': len(stats[0]), 'observations': observations,
                     'update_l2': float(sum((calibrated[k].double()-prior[k].double()).square().sum()
                                            for k in calibrated).sqrt()),
                     'elapsed_s': time.time()-fit_started}
            layer_audits.append(audit)
            changed_layers.append(name)
            with (args.outdir/'layers.jsonl').open('a') as handle:
                handle.write(json.dumps(audit)+'\n')
            print('calibrated', name, 'maps', audit['maps'], 'update_l2', audit['update_l2'], flush=True)
            sb, eb = next_sb, next_eb
            del expert_layer, stats, own_states
    fit_seconds = time.time()-fit_started
    assert len(changed_layers) == (3*plan['smoke_blocks_per_branch'] if args.smoke else 28)
    final = {key: value.detach().cpu().clone() for key, value in leaf.state_dict().items()}
    changed_keys = {key for key in final if any(key.startswith(name+'.') for name in changed_layers)}
    assert all(torch.equal(final[key], initial[key]) for key in final if key not in changed_keys)
    assert all(torch.isfinite(final[key]).all() for key in changed_keys)
    checkpoint = args.outdir/'merged.ckpt'
    torch.save(final, checkpoint)
    final_validation, final_calibration = evaluate(), evaluate(True)
    reloaded = load_leaf(config, checkpoint).to(device).eval().requires_grad_(False)
    export_checks, direct_wrapper_errors = {}, {}
    for task in tasks:
        x, y = train[task]
        x, y = x[:16].to(device), y[:16]
        before = wrappers[task](x).cpu()
        after = PrototypeClassifier(reloaded, *prepare_text(config, task)[2:]).to(device).eval()(x).cpu()
        export_checks[task] = compare(before, after, y)
        assert export_checks[task]['max_abs_logit_error'] <= 1e-5
        assert all(row['margin_within_twice_error'] for row in export_checks[task]['changed_rows'])
        # The feature capture uses the classifier's actually normalized instruction.
        wrapper = wrappers[task]
        embedding, _ = leaf(x, wrapper.instruction[None].expand(len(x), -1))
        direct = torch.nn.functional.normalize(embedding.float(), dim=-1) @ wrapper.prototypes.T
        direct_wrapper_errors[task] = float((direct.cpu()-before).abs().max())
        assert direct_wrapper_errors[task] == 0
    result = {'method': 'featcal_leaf_adaptation', 'plan': plan, 'plan_sha256': sha256(args.plan),
              'engineering_smoke': args.smoke, 'datasets': tasks, 'data_audits': audits,
              'experts': expert_sources, 'initial_validation': initial_validation,
              'initial_calibration': initial_calibration, 'final_validation': final_validation,
              'final_calibration': final_calibration,
              'final_validation_macro_bacc': float(np.mean([v['BAcc'] for v in final_validation.values()])),
              'updated_layers': changed_layers, 'updated_parameter_elements': sum(final[k].numel() for k in changed_keys),
              'outside_blocks_exact': True, 'export_checks': export_checks,
              'actual_instruction_wrapper_checks': direct_wrapper_errors,
              'checkpoint_sha256': sha256(checkpoint),
              'seconds': {'data_setup_initial_evaluation': setup_seconds, 'fit': fit_seconds, 'total': time.time()-started},
              'fitting_labels_used': False, 'optimizer_updates': 0,
              'test_data_read': False, 'test_inference_performed': False,
              'selection': 'One final checkpoint after the fixed block list, no validation selection.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Complete', result['final_validation_macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
