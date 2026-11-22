"""Cached first-Adam head proposals and source-decision-null counterfactuals."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_contrast_normalizer_gradients import geometry


def balanced(value, truth, multiplicity, classes):
    return float(torch.stack([
        (value[truth == c] * multiplicity[truth == c]).sum()
        / multiplicity[truth == c].sum() for c in range(classes)
    ]).mean())


def margin(scores, truth):
    other = scores.clone()
    other[torch.arange(len(truth)), truth] = -torch.inf
    return scores[torch.arange(len(truth)), truth] - other.max(-1).values


def stream_counts(cache, schedule, ti, ref):
    ids = cache['pool_indices'].numpy()
    stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
    assert array_hash(stream) == ref['stream_indices_hash']
    positions = np.searchsorted(ids, stream)
    assert np.array_equal(ids[positions], stream)
    return torch.bincount(torch.from_numpy(positions), minlength=len(ids)).double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    output = Path(plan['output'])
    assert not output.exists()
    started = time.time()
    torch.set_num_threads(4)
    targets = json.loads(Path(plan['target_manifest']).read_text())
    schedule = np.load(plan['schedule'])
    proposals, records = [], {}
    for ti, task in enumerate(plan['datasets']):
        ref = targets['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        p = cache['prototypes'].double()
        centered = p-p.mean(0)
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[singular > singular[0]*plan['rank_relative_cutoff']]
        counts = stream_counts(cache, schedule, ti, ref)
        weights = torch.tensor(ref['class_weights'], dtype=torch.float64)
        head = torch.zeros(768, 2049, dtype=torch.float64)
        for start in range(0, len(counts), plan['batch_size']):
            sl = slice(start, start+plan['batch_size'])
            y = cache['truth'][sl]
            g, _, _, _, _ = geometry(cache['initial_raw'][sl].double(), p, y, basis)
            weighted = g * (counts[sl]*weights[y])[:, None]
            head[:, :-1] += weighted.T@cache['initial_prehead'][sl].double()
            head[:, -1] += weighted.sum(0)
        head /= counts.sum()
        initial = torch.cat([cache['initial_weight'], cache['initial_bias'][:, None]], dim=1)
        assert initial.dtype == torch.float32
        # Real torch Adam on disposable head tensors; no model checkpoint changes.
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([parameter], **plan['adam'])
        parameter.grad = head.float()
        optimizer.step()
        proposal = parameter.detach().double()-initial.double()
        projected = basis.T@(basis@proposal)
        materialized = (initial.double()+projected).float().double()-initial.double()
        relative = float((centered@(projected-proposal)).norm()
                         / (centered@proposal).norm())
        assert relative < plan['exact_contrast_relative_tolerance']
        records[task] = {
            'contrast_rank': len(basis),
            'head_gradient_norm': float(head.norm()),
            'adam_proposal_norm': float(proposal.norm()),
            'decision_null_update_energy_fraction': float((proposal-projected).square().sum()/proposal.square().sum()),
            'exact_source_contrast_relative_error': relative,
            'fp32_materialized_source_contrast_relative_error': float((centered@(materialized-proposal)).norm()/(centered@proposal).norm()),
            'fp32_materialization_update_relative_error': float((materialized-projected).norm()/projected.norm()),
            'unique_fitting_rows': int((counts > 0).sum()),
            'fitting_exposures': int(counts.sum()),
            'target_sha256': ref['target_sha256'],
        }
        proposals.append({'adam': proposal, 'contrast_exact': projected, 'contrast_fp32': materialized})
        print('proposal', task, records[task]['decision_null_update_energy_fraction'], flush=True)
    pair_records, initial_records = {}, {}
    for ri, recipient in enumerate(plan['datasets']):
        ref = targets['records'][recipient]
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        counts = stream_counts(cache, schedule, ri, ref)
        mask = counts > 0
        counts, y = counts[mask], cache['truth'][mask]
        h = cache['initial_prehead'][mask].double()
        raw = cache['initial_raw'][mask].double()
        p = cache['prototypes'].double()
        scores = raw@p.T
        radius = raw.norm(dim=1)
        prediction = scores.argmax(-1)
        base_margin = margin(scores, y)
        base_accuracy = balanced((prediction == y).double(), y, counts, len(p))
        base_unique_accuracy = balanced((prediction == y).double(), y, torch.ones_like(counts), len(p))
        initial_records[recipient] = {
            'exposure_balanced_accuracy': base_accuracy,
            'unique_balanced_accuracy': base_unique_accuracy,
            'raw_scores_vs_cached_logits_argmax_disagreements': int((prediction != cache['initial_logits'][mask].argmax(-1)).sum()),
        }
        for si, source in enumerate(plan['datasets']):
            pair = {}
            score_changes = {}
            for name, update in proposals[si].items():
                effective = p@update
                change = h@effective[:, :-1].T+effective[:, -1]
                new_scores = scores+change
                new_prediction = new_scores.argmax(-1)
                improvement = (margin(new_scores, y)-base_margin)/radius
                pair[name] = {
                    'exposure_balanced_accuracy_delta': balanced((new_prediction == y).double(), y, counts, len(p))-base_accuracy,
                    'unique_balanced_accuracy_delta': balanced((new_prediction == y).double(), y, torch.ones_like(counts), len(p))-base_unique_accuracy,
                    'balanced_margin_delta_in_initial_radius_units': balanced(improvement, y, counts, len(p)),
                    'changed_unique_predictions': int((new_prediction != prediction).sum()),
                }
                if si == ri:
                    score_changes[name] = change
            if si == ri:
                for name in ['contrast_exact', 'contrast_fp32']:
                    difference = score_changes[name]-score_changes['adam']
                    contrast_error = difference-difference.mean(-1, keepdim=True)
                    full_scores = scores+score_changes['adam']
                    projected_scores = scores+score_changes[name]
                    sorted_scores = full_scores.topk(2, dim=-1).values
                    margin_bound = 2*contrast_error.abs().max(-1).values
                    changed = full_scores.argmax(-1) != projected_scores.argmax(-1)
                    safe = sorted_scores[:, 0]-sorted_scores[:, 1] > margin_bound
                    assert not bool((changed & safe).any())
                    pair[name]['source_vs_adam_prediction_disagreements'] = int(changed.sum())
                    pair[name]['source_vs_adam_max_contrast_score_error'] = float(contrast_error.abs().max())
                    pair[name]['source_examples_not_certified_by_rounding_margin_bound'] = int((~safe).sum())
            pair_records[f'{source} -> {recipient}'] = pair
        print('recipient', recipient, flush=True)
    metrics = ['exposure_balanced_accuracy_delta', 'unique_balanced_accuracy_delta',
               'balanced_margin_delta_in_initial_radius_units']
    macro = {}
    for source in plan['datasets']:
        macro[source] = {}
        for name in proposals[0]:
            macro[source][name] = {
                metric: float(np.mean([pair_records[f'{source} -> {recipient}'][name][metric]
                                       for recipient in plan['datasets']])) for metric in metrics
            }
    off_pairs = [pair_records[f'{s} -> {r}'] for s in plan['datasets'] for r in plan['datasets'] if s != r]
    summary = {}
    for name in proposals[0]:
        summary[name] = {
            'off_diagonal_negative_margin_pairs': sum(pair[name][metrics[2]] < 0 for pair in off_pairs),
            'macro_source_recipient_mean': {metric: float(np.mean([macro[s][name][metric] for s in plan['datasets']])) for metric in metrics},
            'better_than_adam_macro_margin_source_tasks': sum(macro[s][name][metrics[2]] > macro[s]['adam'][metrics[2]] for s in plan['datasets']),
            'better_than_adam_macro_accuracy_source_tasks': sum(macro[s][name][metrics[0]] > macro[s]['adam'][metrics[0]] for s in plan['datasets']),
        }
    result = {'unix': time.time(), 'plan': plan, 'plan_sha256': sha256(args.plan),
              'code_sha256': sha256(Path(__file__)), 'proposals': records,
              'initial_records': initial_records, 'pairs': pair_records,
              'macro_by_source': macro, 'summary': summary, 'seconds': time.time()-started,
              'new_eeg_reads': 0, 'new_model_forward_examples': 0, 'model_optimizer_updates': 0,
              'disposable_first_adam_head_proposals': len(proposals), 'scope': plan['scope']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({'summary': summary, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
