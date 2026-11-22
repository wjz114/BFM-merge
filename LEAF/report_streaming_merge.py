"""Report actual streaming outcomes and matched schedule/export checks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from run_streaming_residual_merge import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='label=directory; first is the baseline')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    runs, paths = {}, {}
    for spec in args.runs:
        label, directory = spec.split('=', 1)
        folder = Path(directory)
        run = json.loads((folder / 'run.json').read_text())
        assert run['status'] == 'completed' and len(run['stages']) == 16
        plan = json.loads(Path(run['plan']).read_text())
        assert sha256(Path(run['plan'])) == run['plan_sha256']
        assert run['updates'] == plan['updates_per_arm']
        assert run['training_examples'] == plan['training_examples_per_arm']
        for stage in run['stages']:
            path = folder / f"stage_{stage['stage']:02d}"
            assert sha256(path / 'merged.ckpt') == stage['checkpoint_sha256']
            if stage['stage'] > 1:
                coefficients = torch.load(path / 'coefficients.pt', map_location='cpu', weights_only=True)
                assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
                assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                           for c in stage['export_checks'].values())
        runs[label], paths[label] = run, str(folder)
    baseline_name, baseline = next(iter(runs.items()))
    contrasts = {}
    for label, run in runs.items():
        assert run['order'] == baseline['order']
        assert run['dataset_audits'] == baseline['dataset_audits']
        assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in baseline['stages']]
        if label == baseline_name:
            continue
        stages = {}
        for a, b in zip(baseline['stages'], run['stages']):
            if 'validation' not in a:
                continue
            gains = {t: 100 * (b['validation'][t]['BAcc'] - m['BAcc']) for t, m in a['validation'].items()}
            values = np.asarray(list(gains.values()))
            stages[str(a['stage'])] = {'macro_gain_pp': float(values.mean()), 'per_task_gain_pp': gains,
                                      'improved': int((values > 0).sum()), 'declined': int((values < 0).sum()),
                                      'tied': int((values == 0).sum())}
            if len(values) > 1:
                leave_one = (values.sum() - values) / (len(values) - 1)
                stages[str(a['stage'])]['task_deletion_range_pp'] = [float(leave_one.min()), float(leave_one.max())]
        contrasts[label] = stages
    context = json.loads(Path('runs/paper_merge_v2/analysis/contrast_head_training_summary.json').read_text())
    result = {'runs': paths, 'run_sha256': {k: sha256(Path(v) / 'run.json') for k, v in paths.items()},
              'baseline': baseline_name, 'order': baseline['order'], 'contrasts': contrasts,
              'final_macro_BAcc': {k: r['stages'][-1]['macro_BAcc'] for k, r in runs.items()},
              'stage_macro_BAcc': {k: {str(s['stage']): s['macro_BAcc'] for s in r['stages'] if 'macro_BAcc' in s}
                                   for k, r in runs.items()},
              'joint_reference_macro_BAcc': context['joint_macro_bacc'],
              'offline_M0_reference_macro_BAcc': context['initial_macro_bacc'],
              'matched_data_and_schedules': True, 'coefficient_averages_exact': True,
              'native_export_checks_pass': True,
              'scope': 'Development validation, one expert bank and one order in this comparison. Joint and offline M0 use different training/merge budgets and are context, not matched controls. Task-deletion ranges are sensitivity summaries, not confidence intervals.',
              'resource': {k: {'updates': r['updates'], 'training_examples': r['training_examples'],
                                'elapsed_seconds': r['elapsed_seconds'], 'peak_reserved_bytes': r['peak_reserved_bytes']}
                           for k, r in runs.items()}}
    write_json(args.output, result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    colors = ['#777777', '#397da4', '#b4772b', '#5d9854']
    for i, (label, run) in enumerate(runs.items()):
        points = [s for s in run['stages'] if s['stage'] > 1 and 'macro_BAcc' in s]
        axes[0].plot([s['stage'] for s in points], [100 * s['macro_BAcc'] for s in points],
                     'o-', label=label, color=colors[i % len(colors)])
    axes[0].set(xlabel='Number of seen tasks', ylabel='Seen-task macro balanced accuracy (%)')
    axes[0].legend()
    if contrasts:
        width = .8 / len(contrasts)
        y = np.arange(len(baseline['order']))
        for i, (label, stages) in enumerate(contrasts.items()):
            gains = stages['16']['per_task_gain_pp']
            axes[1].barh(y + (i - (len(contrasts) - 1) / 2) * width,
                         [gains[t] for t in baseline['order']], height=width, label=label,
                         color=colors[(i + 1) % len(colors)])
        axes[1].set_yticks(y, [t.replace('MI_', '').replace('EMO_', '') for t in baseline['order']], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].axvline(0, color='grey', lw=.8)
        axes[1].set_xlabel(f'Final task gain over {baseline_name} (percentage points)')
        axes[1].legend()
    fig.suptitle('Sequential expert merging: fixed calibration and update schedule')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(args.output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({'final_macro_BAcc': result['final_macro_BAcc'],
                      'final_contrasts': {k: v['16'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
