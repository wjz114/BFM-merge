"""Separate accuracy threshold effects from nonadditive logits in fixed interventions."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from check_merge_export_margins import compare
from d_downstream_qformer_ce import classification_metrics, sha256


def main():
    torch.set_num_threads(4)
    started = time.time()
    root = Path('runs/paper_merge_v2')
    sources = {
        'original_only': root/'seed0/full_pool_span_diagnostic',
        'gradient': root/'seed0/full_pool_augmented_gradient_span_diagnostic',
        'random': root/'seed0/full_pool_augmented_random_span_diagnostic'}
    records, summaries, inputs = {}, {}, {}
    for kind, directory in sources.items():
        run = json.loads((directory/'run.json').read_text())
        tasks = list(run['evaluations']['initial'])
        inputs[str(directory/'run.json')] = sha256(directory/'run.json')
        records[kind] = {}
        for task in tasks:
            path = directory/f'{task}_logits.pt'
            data = torch.load(path, map_location='cpu', weights_only=True)
            inputs[str(path)] = sha256(path)
            logits = data['logits']
            y, n = data['truth'], data['n_calibration']
            additive = logits['parallel_only']+logits['orthogonal_only']-logits['initial']
            full_change = (logits['full_update']-logits['initial']).double()
            interaction = (logits['full_update']-additive).double()
            # Common class offsets cannot change softmax or argmax.
            centered_change = full_change-full_change.mean(1, keepdim=True)
            centered_interaction = interaction-interaction.mean(1, keepdim=True)
            records[kind][task] = {}
            for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
                metrics = {c: classification_metrics(v[ids], y[ids]) for c, v in logits.items()}
                assert all(metrics[c]['BAcc'] == run['evaluations'][c][task][split]['BAcc'] for c in logits)
                reconstructed = classification_metrics(additive[ids], y[ids])
                ci, cc = centered_interaction[ids], centered_change[ids]
                records[kind][task][split] = {
                    'native_metrics': metrics, 'additive_logit_metrics': reconstructed,
                    'actual_minus_additive_bacc_pp': 100*(metrics['full_update']['BAcc']-reconstructed['BAcc']),
                    'centered_interaction_rms': float(ci.square().mean().sqrt()),
                    'centered_full_change_rms': float(cc.square().mean().sqrt()),
                    'relative_centered_interaction_l2': float(ci.norm()/cc.norm().clamp_min(1e-30)),
                    'additive_vs_actual': compare(additive[ids], logits['full_update'][ids], y[ids])}
        summaries[kind] = {}
        for split in ['calibration', 'validation']:
            values = [records[kind][t][split] for t in tasks]
            native = {c: float(np.mean([v['native_metrics'][c]['BAcc'] for v in values])) for c in logits}
            additive_bacc = float(np.mean([v['additive_logit_metrics']['BAcc'] for v in values]))
            threshold_interaction = 100*(additive_bacc-native['parallel_only']-native['orthogonal_only']+native['initial'])
            nonadditivity = 100*(native['full_update']-additive_bacc)
            total = 100*(native['full_update']-native['parallel_only']-native['orthogonal_only']+native['initial'])
            assert abs(total-threshold_interaction-nonadditivity) < 1e-10
            summaries[kind][split] = {
                'native_macro_bacc': native, 'additive_logit_macro_bacc': additive_bacc,
                'accuracy_factorial_interaction_pp': total,
                'accuracy_interaction_under_additive_logits_pp': threshold_interaction,
                'actual_minus_additive_accuracy_pp': nonadditivity,
                'changed_predictions_additive_vs_actual': sum(v['additive_vs_actual']['changed_predictions'] for v in values),
                'mean_relative_centered_logit_interaction_l2': float(np.mean([v['relative_centered_interaction_l2'] for v in values]))}
    result = {'unix': time.time(), 'summaries': summaries, 'records': records,
              'input_sha256': inputs, 'code_sha256': sha256(Path(__file__)),
              'seconds': time.time()-started, 'new_EEG_read': False, 'model_inference_performed': False,
              'optimizer_updates': 0,
              'scope': 'Post-hoc descriptive decomposition of saved fixed-intervention logits on exposed data. Additive logits are an analytic surrogate, not a deployable model or new candidate. The accuracy factorial contrast splits algebraically into an additive-logit threshold term and actual-minus-additive accuracy; neither term uniquely identifies a training mechanism or measures curvature.'}
    output = root/'analysis/span_logit_interaction.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v['validation'] for k, v in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
