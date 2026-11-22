"""Report all frozen bank/order pairs without treating orders as independent seeds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_subspace_merge import write_json


def folder_for(bank, order, condition):
    if bank == 0:
        root = 'streaming_subspace_v1' if condition == 'incoming_rows' else 'streaming_history_sketch_v1'
        return Path('runs') / root / f'{condition}_{order}'
    return Path('runs/streaming_history_replication_v1') / f'seed{bank}' / f'{condition}_{order}'


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_history_replication_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    conditions = ['incoming_rows', 'history_sketch', 'random_sketch']
    tasks = json.loads(Path('runs/streaming_history_sketch_v1/plan.json').read_text())['tasks']
    runs, sources, pairs = {}, {}, {}
    for bank in [0, 1, 2]:
        for order in ['forward', 'reverse']:
            for condition in conditions:
                folder = folder_for(bank, order, condition)
                run = json.loads((folder / 'run.json').read_text())
                assert run['status'] == 'completed' and len(run['stages']) == 16
                plan = json.loads(Path(run['plan']).read_text())
                assert sha256(Path(run['plan'])) == run['plan_sha256']
                assert run['updates'] == plan['updates_per_arm'] == 1620
                assert run['training_examples'] == plan['training_examples_per_arm'] == 25920
                assert run['order'] == (tasks if order == 'forward' else list(reversed(tasks)))
                for stage in run['stages']:
                    stage_dir = folder / f"stage_{stage['stage']:02d}"
                    assert sha256(stage_dir / 'merged.ckpt') == stage['checkpoint_sha256']
                    if stage['stage'] > 1:
                        coefficients = torch.load(stage_dir / 'coefficients.pt', map_location='cpu', weights_only=True)
                        assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
                        assert coefficients['mean'].numel() == (174216 if condition == 'incoming_rows' else 694566)
                        assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                                   for v in stage['export_checks'].values())
                assert sum(s['input_split']['extra_forward_examples'] for s in run['stages'][1:]) == 1920
                sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
                runs[bank, order, condition] = run
            baseline = runs[bank, order, 'incoming_rows']
            for condition in conditions[1:]:
                run = runs[bank, order, condition]
                assert run['dataset_audits'] == baseline['dataset_audits']
                assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in baseline['stages']]
                assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in baseline['stages'][1:]]
            histories = [torch.load(folder_for(bank, order, c) / 'history.pt', weights_only=True, map_location='cpu')
                         for c in conditions[1:]]
            assert histories[0]['count'] == histories[1]['count'] == 16
            assert all(torch.equal(v, histories[1]['sketch'][n]) for n, v in histories[0]['sketch'].items())
            del histories
            validation = {c: runs[bank, order, c]['stages'][-1]['validation'] for c in conditions}
            pairs[f'bank{bank}_{order}'] = {
                'bank': bank, 'order': order,
                'macro_BAcc': {c: float(np.mean([validation[c][t]['BAcc'] for t in tasks])) for c in conditions},
                'history_minus_incoming': contrast(validation['history_sketch'], validation['incoming_rows'], tasks),
                'history_minus_random': contrast(validation['history_sketch'], validation['random_sketch'], tasks),
                'random_minus_incoming': contrast(validation['random_sketch'], validation['incoming_rows'], tasks)}
            print(f'checked bank{bank} {order}', flush=True)
    bank_means = {}
    for key in ['history_minus_incoming', 'history_minus_random', 'random_minus_incoming']:
        values = {str(bank): float(np.mean([pairs[f'bank{bank}_{order}'][key]['gain_pp']
                                           for order in ['forward', 'reverse']])) for bank in [0, 1, 2]}
        numbers = np.asarray(list(values.values()))
        bank_means[key] = {'per_bank_mean_over_orders_pp': values, 'mean_pp': float(numbers.mean()),
                           'sample_sd_across_three_bank_means_pp': float(numbers.std(ddof=1)),
                           'range_pp': [float(numbers.min()), float(numbers.max())]}
    result = {'sources': sources, 'pairs': pairs, 'bank_aggregates': bank_means,
              'matched_data_schedules_and_auxiliary_ids': True, 'native_exports_and_coefficient_means_pass': True,
              'paired_history_tensors_identical': True, 'new_replication_updates': 12 * 1620,
              'comparison_updates_including_reused_bank0': 18 * 1620,
              'scope': 'Three independently trained expert banks and bank-specific calibration/training randomness, two fixed arrival orders. Bank0 selected the candidate; banks1/2 are unchanged follow-ups. Orders within a bank are correlated, not independent seeds. Task-deletion ranges are sensitivity, not confidence intervals. The random orientation seed rule is unchanged across banks. Exposed development validation, no official test or new backbone. No bank0 Joint reference is silently reused for other banks.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.6]})
    names = list(pairs)
    x = np.arange(len(names))
    for j, (key, label, color) in enumerate([
            ('history_minus_incoming', 'History minus incoming rows', '#397da4'),
            ('history_minus_random', 'History minus random complement', '#b4772b')]):
        axes[0].bar(x + (j - .5) * .36, [pairs[n][key]['gain_pp'] for n in names], .36,
                    label=label, color=color)
    axes[0].axhline(0, color='grey', lw=.8)
    axes[0].set_xticks(x, [n.replace('_', '\n') for n in names], fontsize=9)
    axes[0].set_ylabel('Final macro gain (percentage points)')
    axes[0].legend(fontsize=8)
    heat = np.asarray([[pairs[n]['history_minus_incoming']['per_task_pp'][t] for n in names] for t in tasks])
    limit = max(abs(heat.min()), abs(heat.max()))
    im = axes[1].imshow(heat, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1].set_xticks(x, [n.replace('_', '\n') for n in names], fontsize=9)
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=9)
    axes[1].set_title('Per-task gain over incoming-row baseline')
    fig.colorbar(im, ax=axes[1], label='Percentage points')
    fig.suptitle('Unchanged historical-sketch merging across expert banks and arrival orders')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps(bank_means, indent=2))


if __name__ == '__main__':
    main()
