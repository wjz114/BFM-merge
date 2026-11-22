"""Report fixed averaging separately from its source fits and checkpoint selection."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from report_replication import official_scores, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/aligned_merge_v1'))
    args = parser.parse_args()
    out = args.root / 'analysis'
    out.mkdir(exist_ok=True)
    names = ['block_balanced_avg', 'tensor_balanced_avg']
    comparisons = [
        ('tensor_balanced_avg', 'block_balanced_avg'),
        ('tensor_balanced_avg', 'tensor_balanced_ce'),
        ('tensor_balanced_avg', 'block_balanced_ce'),
        ('block_balanced_avg', 'block_balanced_ce'),
        ('tensor_balanced_avg', 'block_ce'),
        ('tensor_balanced_avg', 'tensor_ce'),
        ('block_balanced_avg', 'block_ce'),
        ('tensor_balanced_avg', 'base'),
        ('tensor_balanced_avg', 'joint'),
        ('block_balanced_avg', 'base'),
        ('block_balanced_avg', 'joint'),
    ]
    rows, tasks, costs, audits, pending, score_rows = [], [], [], {}, [], []
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for seed in range(3):
        folder = args.root / f'seed{seed}'
        runs, scores = {}, {'validation': {}, 'official_l2': {}}
        for name in [*names, 'block_balanced_ce', 'tensor_balanced_ce', 'block_ce', 'tensor_ce']:
            path = folder / name / 'run.json'
            if not path.exists():
                if name in names:
                    pending.append(f'seed{seed}/{name}')
                continue
            run = runs[name] = json.loads(path.read_text())
            scores['validation'][name] = {ds: run['best_val'][ds]['BAcc'] for ds in run['datasets']}
            official = official_scores(folder, name, run['datasets'])
            if official is not None:
                scores['official_l2'][name] = official
            elif name in names:
                pending.append(f'seed{seed}/official_l2/{name}')
            if name not in names:
                continue
            source = folder / name.replace('_avg', '_ce')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            assert run['source_run_sha256'] == digest(source / 'run.json')
            assert run['source_history_sha256'] == digest(source / 'coefficient_history.npz')
            assert run['averaging_steps'] == [800, 960, 1120, 1280, 1440, 1600]
            assert run['alpha_updates_actual'] == 0 and run['source_optimizer_updates'] == 1600
            assert not run['engineering_subset'] and run['best_step'] is None
            a, b = np.load(folder / name / 'calibration_indices.npz'), np.load(source / 'calibration_indices.npz')
            assert a.files == b.files and all(np.array_equal(a[ds], b[ds]) for ds in a.files)
            history = np.load(source / 'coefficient_history.npz')
            expected = history['alpha'][5:].astype(np.float64).mean(0).astype(np.float32)
            assert np.array_equal(np.load(folder / name / 'best_alpha.npy'), expected)
            audits[f'seed{seed}/{name}'] = {'fixed_mean_exact': True, 'source_hashes_match': True,
                                           'calibration_ids_match': True, 'run_sha256': digest(path)}
            costs.append({'seed': seed, 'method': name, 'source_updates': 1600, 'new_updates': 0,
                          'source_fit_seconds': run['seconds']['source_training'],
                          'additional_seconds': run['seconds']['postprocess_total'],
                          'pipeline_seconds': run['seconds']['total']})
            trace = json.loads((source / 'history.json').read_text())
            color = '#586776' if name.startswith('block') else '#007f73'
            axes[seed].plot([h['step'] for h in trace], [100 * h['val_macro_BAcc'] for h in trace],
                            color=color, label=source.name)
            axes[seed].axhline(100 * run['best_val_macro_BAcc'], color=color, linestyle='--', label=name)
        datasets = runs['block_ce']['datasets']
        poc = args.root.parent / 'poc16' / f'seed{seed}' / 'direct_inference'
        for reference in ['base', 'joint']:
            meta = json.loads((poc / f'{reference}.json').read_text())
            assert meta['status'] == 'complete' and meta['seed'] == 42 and meta['batch_size'] == 256
            assert 2 in meta['levels'] and set(datasets) <= set(meta['completed_datasets'])
            scores['official_l2'][reference] = {
                row['dataset']: float(row['level2_bAcc'])
                for row in csv.DictReader((poc / f'{reference}.csv').open()) if row['dataset'] in datasets}
            for name in names:
                if name not in scores['official_l2']:
                    continue
                actual = json.loads((folder / 'official_l2' / f'{name}.json').read_text())
                assert all(actual['labels'][ds] == meta['labels'][ds] and
                           actual['instructions'][ds]['2'] == meta['instructions'][ds]['2'] for ds in datasets)
        for protocol, values in scores.items():
            score_rows.extend({'protocol': protocol, 'seed': seed, 'method': name,
                               'macro_bacc': statistics.mean(per_task.values())}
                              for name, per_task in values.items())
            for candidate, control in comparisons:
                if candidate not in values or control not in values:
                    continue
                delta = {ds: 100 * (value - values[control][ds]) for ds, value in values[candidate].items()}
                rows.append({'protocol': protocol, 'seed': seed, 'candidate': candidate, 'control': control,
                             'candidate_macro_bacc': statistics.mean(values[candidate].values()),
                             'control_macro_bacc': statistics.mean(values[control].values()),
                             'delta_macro_pp': statistics.mean(delta.values()),
                             'worst_task': min(delta, key=delta.get), 'worst_task_delta_pp': min(delta.values()),
                             'tasks_down': sum(v < 0 for v in delta.values())})
                tasks.extend({'protocol': protocol, 'seed': seed, 'candidate': candidate, 'control': control,
                              'dataset': ds, 'delta_pp': value} for ds, value in delta.items())
        axes[seed].axvspan(800, 1600, alpha=.07, color='black')
        axes[seed].set(title=f'Seed {seed}: source trajectories and evaluated averaged models', ylabel='Validation macro BAcc (%)')
        axes[seed].grid(alpha=.2)
        if any(name in runs for name in names):
            axes[seed].legend(fontsize=8)
        else:
            axes[seed].text(.5, .5, 'Averaged-model validation pending', ha='center',
                            transform=axes[seed].transAxes)
            axes[seed].set_yticks([])
    axes[-1].set_xlabel('Source optimizer updates; shaded interval contributes to the fixed average')
    fig.tight_layout()
    fig.savefig(out / 'trajectory_average_dynamics.png', dpi=160)
    plt.close(fig)
    aggregates = []
    for protocol in ['validation', 'official_l2']:
        for candidate, control in comparisons:
            selected = [r for r in rows if (r['protocol'], r['candidate'], r['control']) == (protocol, candidate, control)]
            if not selected:
                continue
            values = [r['delta_macro_pp'] for r in selected]
            aggregates.append({'protocol': protocol, 'candidate': candidate, 'control': control,
                               'n': len(values), 'mean_delta_pp': statistics.mean(values),
                               'sample_sd_pp': statistics.stdev(values) if len(values) > 1 else None})
    score_aggregates = []
    for protocol in ['validation', 'official_l2']:
        for name in [*names, 'block_balanced_ce', 'tensor_balanced_ce', 'block_ce', 'tensor_ce', 'base', 'joint']:
            values = [row['macro_bacc'] for row in score_rows if row['protocol'] == protocol and row['method'] == name]
            if values:
                score_aggregates.append({'protocol': protocol, 'method': name, 'n': len(values),
                                         'mean_macro_bacc': statistics.mean(values),
                                         'sample_sd': statistics.stdev(values) if len(values) > 1 else None})
    for suffix, data in [('pairs', rows), ('tasks', tasks), ('costs', costs), ('scores', score_rows)]:
        write_csv(out / f'trajectory_average_{suffix}.csv', data)
    summary = {'status': 'complete' if not pending else 'incomplete', 'pending': pending,
               'aggregates': aggregates, 'score_aggregates': score_aggregates, 'audits': audits}
    (out / 'trajectory_average_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Fixed trajectory averaging', '',
             'One arithmetic mean of coefficient states at800/960/1120/1280/1440/1600, evaluated as one ordinary model. '
             'Zero new optimizer updates; no selection among averaging windows. All earlier official tests were known during development.', '',
             '| Protocol | Method | Seeds | Macro BAcc mean | Sample SD |',
             '|---|---|---:|---:|---:|']
    for row in score_aggregates:
        sd = f'{row["sample_sd"]:.6f}' if row['sample_sd'] is not None else 'not estimated'
        lines.append(f'| {row["protocol"]} | {row["method"]} | {row["n"]}/3 | {row["mean_macro_bacc"]:.6f} | {sd} |')
    lines += ['',
             '| Protocol | Seed | Candidate | Control | Difference (pp) | Worst task | Worst difference (pp) | Tasks down |',
             '|---|---:|---|---|---:|---|---:|---:|']
    for row in rows:
        lines.append(f'| {row["protocol"]} | {row["seed"]} | {row["candidate"]} | {row["control"]} | '
                     f'{row["delta_macro_pp"]:+.3f} | {row["worst_task"]} | {row["worst_task_delta_pp"]:+.3f} | {row["tasks_down"]} |')
    lines += ['', '| Protocol | Candidate | Control | Pairs | Mean difference (pp) | Sample SD (pp) |',
              '|---|---|---|---:|---:|---:|']
    for row in aggregates:
        sd = f'{row["sample_sd_pp"]:.3f}' if row['sample_sd_pp'] is not None else 'not estimated'
        lines.append(f'| {row["protocol"]} | {row["candidate"]} | {row["control"]} | {row["n"]}/3 | {row["mean_delta_pp"]:+.3f} | {sd} |')
    lines += ['', 'Pending: ' + (', '.join(pending) if pending else 'none') + '.', '',
              'Seed pairs measure replication; task differences and checkpoints are not independent repetitions. '
              'The dashed lines below are measured performance of the averaged weights, not the arithmetic mean of checkpoint scores.', '',
              '![Source trajectories and averaged-model validation](trajectory_average_dynamics.png)', '',
              'Source fitting and additional export/validation costs are separated in `trajectory_average_costs.csv`. '
              'The source fits are reused, not rerun. Base/Joint references and expert-bank costs remain in '
              '[the class-balanced report](class_balance_replication_report.md).', '']
    if not pending:
        primary = [row for row in rows if row['protocol'] == 'official_l2'
                   and row['candidate'] == 'tensor_balanced_avg']
        original = [row for row in primary if row['control'] == 'block_ce']
        matched = [row for row in primary if row['control'] == 'block_balanced_avg']
        worst = min(original, key=lambda row: row['worst_task_delta_pp'])
        averaging = [row for row in primary if row['control'] == 'tensor_balanced_ce']
        lines += [
            f'The combined candidate improves official macro BAcc over original Block in '
            f'{sum(row["delta_macro_pp"] > 0 for row in original)}/3 seeds and over the matched averaged '
            f'Block control in {sum(row["delta_macro_pp"] > 0 for row in matched)}/3 seeds. '
            'This supports modest potential on the macro objective; it does not establish a uniform task benefit.', '',
            f'The worst original-Block task regression is {worst["worst_task"]} in seed{worst["seed"]}: '
            f'{worst["worst_task_delta_pp"]:+.3f} pp. Averaging alone beats the selected balanced tensor '
            f'source in {sum(row["delta_macro_pp"] > 0 for row in averaging)}/3 seeds. '
            'The per-task tables and Joint comparisons bound the claim.', '',
            'The arithmetic averaging rule is related to '
            '[Izmailov et al., SWA](https://arxiv.org/abs/1803.05407v3); here the saved states come from '
            'Adam fitting of merge coefficients. This is not a reproduction of the SGD experiments or a novelty claim. '
            'Earlier test results informed development, so these are not untouched test confirmations. '
            'Three training seeds do not establish strong statistical significance.', '',
            'Actual backbone/interface displacements are in `trajectory_average_parameter_geometry.json`. '
            'Export/state and frozen official-receipt audits are in '
            '`../inputs/trajectory_average_deployment_review.json` and '
            '`../inputs/trajectory_average_official_review.json`.', '',
        ]
    (out / 'trajectory_average_report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
