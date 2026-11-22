"""Evaluate fixed prototype-contrast head edits with full native EEG forwards."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, sha256,
)
from diagnose_sensor_projection import calibration_and_validation
from load_config import build_model_config, load_yaml


HEAD = ['qformer.output_embed.2.weight', 'qformer.output_embed.2.bias']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == expected, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    build = json.loads(Path(plan['build_run']).read_text())
    fit = json.loads(Path(plan['source_fit']).read_text())
    average = json.loads(Path(plan['source_average']).read_text())
    reference = json.loads(Path(plan['reference_control']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    tasks = fit['datasets']
    assert tasks == build['datasets'] == reference['datasets']
    paths = {'initial': plan['initial_checkpoint'], **{k: v['path'] for k, v in build['outputs'].items()}}
    assert list(paths) == ['initial', 'cross_contrast_restore', 'global_null_restore', 'base_head']
    states = {k: torch.load(v, map_location='cpu', weights_only=True) for k, v in paths.items()}
    for state in states.values():
        assert set(state) == set(states['initial'])
        assert all(torch.equal(state[k], states['initial'][k]) for k in state if k not in HEAD)
        assert all(torch.isfinite(state[k]).all() for k in HEAD)
    patches = torch.load(build['own_patches_path'], map_location='cpu', weights_only=True)
    indices = np.load(plan['calibration'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(paths['initial'])).to(device).eval().requires_grad_(False)
    evaluations = {c: {} for c in paths}
    own_checks, global_checks, dataset_audits = {}, {}, {}
    for task in tasks:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        data_hashes = {k: array_hash(v) for k, v in [('calibration_eeg', cx), ('calibration_labels', cy),
                                                     ('validation_eeg', vx), ('validation_labels', vy)]}
        assert data_hashes['calibration_eeg'] == reference['dataset_audits'][task]['eeg_hash']
        # The audit was frozen before coverage training and all validation evaluations.
        audit = coverage['audits'][task]
        assert data_hashes['calibration_labels'] == audit['calibration_labels_hash']
        assert data_hashes['validation_eeg'] == audit['validation_eeg_hash']
        assert data_hashes['validation_labels'] == audit['validation_labels_hash']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        dataset_audits[task] = {**data_hashes, 'calibration_size': n, 'validation_size': len(vy)}
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        slices = [('calibration', slice(0, n)), ('validation', slice(n, None))]

        def predict(state):
            leaf.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            assert torch.isfinite(values).all()
            return values

        logits = {}
        for condition, state in states.items():
            logits[condition] = predict(state)
            evaluations[condition][task] = {s: classification_metrics(logits[condition][ids], truth[ids])
                                            for s, ids in slices}
        assert evaluations['initial'][task]['validation']['BAcc'] == average['folded_val'][task]['BAcc']
        global_checks[task] = {s: compare(logits['initial'][ids], logits['global_null_restore'][ids], truth[ids])
                               for s, ids in slices}
        expert_path = Path(fit['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        expert = torch.load(expert_path, map_location='cpu', weights_only=True)
        before = predict(expert)
        assert set(patches[task]) == set(HEAD)
        expert.update(patches[task])
        after = predict(expert)
        own_checks[task] = {s: {'comparison': compare(before[ids], after[ids], truth[ids]),
                                'original': classification_metrics(before[ids], truth[ids]),
                                'restored': classification_metrics(after[ids], truth[ids])}
                            for s, ids in slices}
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits,
                    'original_expert_logits': before, 'restored_expert_logits': after},
                   args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(
            {'evaluations': evaluations, 'own_checks': own_checks, 'global_checks': global_checks}, indent=2)+'\n')
        print(task, 'merged gain pp',
              {c: 100*(evaluations[c][task]['validation']['BAcc']-evaluations['initial'][task]['validation']['BAcc'])
               for c in paths if c != 'initial'},
              'own changed', own_checks[task]['validation']['comparison']['changed_predictions'],
              'global changed', global_checks[task]['validation']['changed_predictions'], flush=True)
    summaries, contrasts = {}, {}
    for split in ['calibration', 'validation']:
        summaries[split] = {c: float(np.mean([evaluations[c][t][split]['BAcc'] for t in tasks])) for c in paths}
        contrasts[split] = {}
        for treatment, control in [(c, 'initial') for c in paths if c != 'initial']+[('cross_contrast_restore', 'base_head')]:
            gains = {t: 100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in tasks}
            contrasts[split][treatment+'_vs_'+control] = {
                'macro_gain_pp': float(np.mean(list(gains.values()))), 'per_task_gain_pp': gains,
                'improved': sum(v > 0 for v in gains.values()), 'declined': sum(v < 0 for v in gains.values()),
                'leave_one_task_out_pp': {t: float(np.mean([v for k, v in gains.items() if k != t])) for t in tasks}}
    result = {
        'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
        'datasets': tasks, 'evaluations': evaluations, 'summaries': summaries, 'contrasts': contrasts,
        'own_checks': own_checks, 'global_checks': global_checks, 'dataset_audits': dataset_audits,
        'exact_initial_validation_bacc_replays': True, 'all_other_merged_tensors_exact': True,
        'own_changed_predictions': {s: sum(own_checks[t][s]['comparison']['changed_predictions'] for t in tasks)
                                    for s in ['calibration', 'validation']},
        'global_changed_predictions': {s: sum(global_checks[t][s]['changed_predictions'] for t in tasks)
                                       for s in ['calibration', 'validation']},
        'exported_paths': paths, 'checkpoint_sha256': {c: sha256(Path(p)) for c, p in paths.items()},
        'seconds': time.time()-started, 'optimizer_updates': 0, 'test_data_read': False,
        'test_inference_performed': False,
        'scope': 'Fixed bank0 candidate; all16 exposed full validation tasks. Every merged condition and both own expert states use native full EEG forwards, without feature caching or surrogate logits. Ordinary exported checkpoints are loaded strictly. Own/global argmax invariance is checked empirically; CE/probability preservation is not expected. No model selection, scale fitting, novelty claim or independent confirmation.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', summaries['validation'], 'own changes', result['own_changed_predictions'],
          'global changes', result['global_changed_predictions'], flush=True)


if __name__ == '__main__':
    main()
