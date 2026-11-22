"""Paired three-seed Block/tensor replication, with partial results explicit."""
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


def write_csv(path, rows):
    if rows:
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def official_scores(folder, method, datasets):
    path = folder / 'official_l2' / f'{method}.csv'
    meta_path = path.with_suffix('.json')
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get('status') != 'complete':
        return None
    assert meta['seed'] == 42 and meta['batch_size'] == 256 and meta['levels'] == [2]
    assert meta['completed_datasets'] == datasets
    values = {row['dataset']: float(row['level2_bAcc']) for row in csv.DictReader(path.open())
              if row['dataset'] in datasets}
    assert set(values) == set(datasets)
    return values


def summarize_dynamics(root, output, methods, prefix):
    """Describe completed trajectories; checkpoints are not independent repetitions."""
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    block_method, tensor_method = methods
    colors = {block_method: '#586776', tensor_method: '#007f73'}
    if block_method == 'block_ce':
        colors['tensor_scale_ce'] = '#c07a24'
    for seed in [0, 1, 2]:
        folder = root / f'seed{seed}'
        runs, histories = {}, {}
        for method, color in colors.items():
            run_path = folder / method / 'run.json'
            if not run_path.exists():
                continue
            runs[method] = json.loads(run_path.read_text())
            histories[method] = json.loads((folder / method / 'history.json').read_text())
            for column, key in enumerate(['val_macro_BAcc', 'val_macro_Loss']):
                history = histories[method]
                scale = 100 if column == 0 else 1
                axes[seed, column].plot([item['step'] for item in history],
                    [scale * item[key] for item in history], label=method, color=color)
                selected = next(item for item in history if item['step'] == runs[method]['best_step'])
                axes[seed, column].scatter(selected['step'], scale * selected[key], color=color, s=35)
        for column, metric in enumerate(['Validation macro BAcc (%)', 'Validation macro CE']):
            axes[seed, column].set_title(f'Seed {seed}: {metric}')
            axes[seed, column].grid(alpha=.2)
            if runs:
                axes[seed, column].legend(fontsize=8)
            else:
                axes[seed, column].text(.5, .5, 'Runs pending', ha='center', transform=axes[seed, column].transAxes)
        if not all(method in runs for method in methods):
            continue
        block = {item['step']: item for item in histories[block_method]}
        tensor = {item['step']: item for item in histories[tensor_method]}
        assert block.keys() == tensor.keys()
        steps = sorted(step for step in block if step > 0)
        gains = [100 * (tensor[step]['val_macro_BAcc'] - block[step]['val_macro_BAcc']) for step in steps]
        selected = runs[tensor_method]['best_val_macro_BAcc']
        rows.append({'seed': seed, 'post_init_checkpoints': len(steps),
                     'checkpoints_tensor_above_same_step_block': sum(gain > 0 for gain in gains),
                     'median_same_step_gain_pp': float(np.median(gains)),
                     'minimum_same_step_gain_pp': min(gains),
                     'final_step_gain_pp': gains[-1],
                     'tensor_final_minus_selected_pp': 100 * (tensor[steps[-1]]['val_macro_BAcc'] - selected)})
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer updates')
    fig.suptitle('Completed runs only; dots mark validation-selected checkpoints')
    fig.tight_layout()
    fig.savefig(output / f'{prefix}_dynamics.png', dpi=170)
    plt.close(fig)
    write_csv(output / f'{prefix}_dynamics.csv', rows)
    return rows


def reference_and_cost_context(root, output, methods, prefix):
    """Reuse existing POC references and distinguish their sunk cost from merging."""
    references, comparisons, costs, task_rows = [], [], [], []
    for seed in [0, 1, 2]:
        folder = root / f'seed{seed}'
        poc = root.parent / 'poc16' / f'seed{seed}'
        audit = json.loads((root / 'inputs' / f'seed{seed}' / 'audit.json').read_text())
        datasets = list(audit['datasets'])
        scores = {}
        for name in ['base', 'joint']:
            path = poc / 'direct_inference' / name
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['status'] == 'complete' and meta['seed'] == 42 and meta['batch_size'] == 256
            assert 2 in meta['levels'] and set(datasets) <= set(meta['completed_datasets'])
            scores[name] = {row['dataset']: float(row['level2_bAcc'])
                            for row in csv.DictReader(path.with_suffix('.csv').open()) if row['dataset'] in datasets}
            assert set(scores[name]) == set(datasets)
        expert_runs = [json.loads((poc / 'experts' / ds / 'run.json').read_text()) for ds in datasets]
        joint = json.loads((poc / 'joint/run.json').read_text())
        references.append({'seed': seed, 'base_official_l2_macro': statistics.mean(scores['base'].values()),
                           'joint_official_l2_macro': statistics.mean(scores['joint'].values()),
                           'expert_training_updates_total': sum(run['actual_optimizer_steps'] for run in expert_runs),
                           'expert_training_wall_seconds_sum': sum(run['seconds']['total'] for run in expert_runs),
                           'joint_training_updates': joint['actual_optimizer_steps'],
                           'joint_training_wall_seconds': joint['seconds']['total']})
        context_methods = methods + (['tensor_scale_ce'] if methods[0] == 'block_ce' else ['block_ce', 'tensor_ce'])
        for method in context_methods:
            path = folder / method / 'run.json'
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            seconds = run['seconds']
            receipt_path = folder / 'official_l2' / f'{method}_receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            costs.append({'seed': seed, 'method': method, 'fitted_coefficients': run['num_coefficients'],
                          'updates': run['alpha_updates_actual'], 'setup_seconds': seconds['data_and_setup'],
                          'optimization_and_validation_seconds': seconds['optimization_incl_validation'],
                          'export_reload_and_reporting_seconds': seconds['total'] - seconds['data_and_setup'] - seconds['optimization_incl_validation'],
                          'merge_total_seconds': seconds['total'], 'peak_cuda_allocated_gib': run['peak_cuda_memory_bytes'] / 2**30,
                          'checkpoint_bytes': (folder / method / 'merged.ckpt').stat().st_size,
                          'official_test_seconds': receipt.get('elapsed_s') if receipt.get('status') == 'complete' else None})
            actual = official_scores(folder, method, run['datasets'])
            if actual is None:
                continue
            actual_meta = json.loads((folder / 'official_l2' / f'{method}.json').read_text())
            for reference in ['base', 'joint']:
                reference_meta = json.loads((poc / 'direct_inference' / f'{reference}.json').read_text())
                assert all(actual_meta['labels'][ds] == reference_meta['labels'][ds] and
                           actual_meta['instructions'][ds]['2'] == reference_meta['instructions'][ds]['2'] for ds in datasets)
                delta = {ds: 100 * (actual[ds] - scores[reference][ds]) for ds in datasets}
                comparisons.append({'seed': seed, 'candidate': method, 'reference': reference,
                                    'delta_macro_pp': statistics.mean(delta.values()),
                                    'worst_task_delta_pp': min(delta.values()),
                                    'worst_task': min(delta, key=delta.get),
                                    'tasks_down': sum(value < 0 for value in delta.values())})
                task_rows.extend({'seed': seed, 'candidate': method, 'reference': reference, 'dataset': ds,
                                  'candidate_bacc': actual[ds], 'reference_bacc': scores[reference][ds], 'delta_pp': value}
                                 for ds, value in delta.items())
    for name, rows in [('poc_references', references), ('reference_comparisons', comparisons),
                       ('reference_tasks', task_rows), ('compute', costs)]:
        write_csv(output / f'{prefix}_{name}.csv', rows)
    lines = ['', 'Existing POC references and compute context:', '',
             '| Seed | Base L2 | Joint L2 | Expert-bank training updates / summed seconds | Joint updates / seconds |',
             '|---|---:|---:|---:|---:|']
    for row in references:
        lines.append(f'| {row["seed"]} | {row["base_official_l2_macro"]:.6f} | {row["joint_official_l2_macro"]:.6f} | '
                     f'{row["expert_training_updates_total"]} / {row["expert_training_wall_seconds_sum"]:.1f} | '
                     f'{row["joint_training_updates"]} / {row["joint_training_wall_seconds"]:.1f} |')
    lines += ['', 'The expert bank is reused by both mergers. Its original training cost is included above, not charged again per merge. '
              'Summed process wall time is not elapsed calendar time or measured device utilization; concurrent jobs and hardware affect comparisons.', '',
              '| Seed | Method | Coefficients | Merge seconds, including validation/export | Peak allocated CUDA GiB |',
              '|---|---|---:|---:|---:|']
    for row in costs:
        lines.append(f'| {row["seed"]} | {row["method"]} | {row["fitted_coefficients"]} | '
                     f'{row["merge_total_seconds"]:.1f} | {row["peak_cuda_allocated_gib"]:.2f} |')
    lines += ['', 'Each deployed merger is one ordinary full checkpoint; expert tensors and fitted coefficients are needed only during merging. '
              f'The candidate uses no teacher cache or alignment/SVD stage. Detailed phase times, checkpoint sizes and test times are in `{prefix}_compute.csv`.', '',
              f'Official per-task changes versus Base and Joint are in `{prefix}_reference_tasks.csv`; '
              f'paired granularity changes are in `{prefix}_tasks.csv`. References predate this experiment.', '']
    interrupted = []
    for path in sorted(root.glob('seed*/interrupted/*/interruption_summary.json')):
        row = json.loads(path.read_text())
        interrupted.append({'seed': int(path.parts[-4].removeprefix('seed')), 'attempt': path.parent.name, **row})
        lines.append(f'Interrupted attempt `{path.relative_to(root)}`: last logged update '
                     f'{row["last_logged_optimizer_update"]}; optimization/validation time through its last checkpoint '
                     f'{row["last_validation_optimization_elapsed_seconds"]:.1f}s; elapsed time to observed termination '
                     f'at most {row["observed_duration_upper_bound_seconds"]:.1f}s. This is extra experiment cost, '
                     'excluded from completed-run performance and timing comparisons.')
    write_csv(output / f'{prefix}_interrupted_cost.csv', interrupted)
    return lines


def compare_class_balance_effect(root, output):
    """Compare each weighted run with its already-completed unweighted counterpart."""
    rows, tasks, aggregates = [], [], []
    for seed in range(3):
        folder = root / f'seed{seed}'
        for candidate, control in [('block_balanced_ce', 'block_ce'), ('tensor_balanced_ce', 'tensor_ce')]:
            paths = [folder / name / 'run.json' for name in [candidate, control]]
            if not all(path.exists() for path in paths):
                continue
            weighted, raw = [json.loads(path.read_text()) for path in paths]
            for key in ['datasets', 'experts', 'seed', 'base_checkpoint_sha256', 'lr', 'batch_size',
                        'alpha_updates_actual', 'eval_every', 'num_coefficients', 'precision']:
                assert weighted[key] == raw[key], (seed, candidate, key)
            calibration = [np.load(folder / name / 'calibration_indices.npz') for name in [candidate, control]]
            assert all(np.array_equal(calibration[0][ds], calibration[1][ds]) for ds in raw['datasets'])
            schedules = [np.load(folder / name / 'metrics.npz')['step_dataset'] for name in [candidate, control]]
            assert np.array_equal(*schedules)
            scores = {'validation': [{ds: run['best_val'][ds]['BAcc'] for ds in run['datasets']} for run in [weighted, raw]]}
            official = [official_scores(folder, name, raw['datasets']) for name in [candidate, control]]
            if all(value is not None for value in official):
                scores['official_l2'] = official
            for protocol, (w, u) in scores.items():
                delta = {ds: 100 * (w[ds] - u[ds]) for ds in raw['datasets']}
                rows.append({'seed': seed, 'protocol': protocol, 'candidate': candidate, 'control': control,
                             'weighted_macro_bacc': statistics.mean(w.values()),
                             'unweighted_macro_bacc': statistics.mean(u.values()),
                             'delta_macro_pp': statistics.mean(delta.values()), 'worst_task': min(delta, key=delta.get),
                             'worst_task_delta_pp': min(delta.values()), 'tasks_down': sum(value < 0 for value in delta.values())})
                tasks.extend({'seed': seed, 'protocol': protocol, 'candidate': candidate, 'control': control,
                              'dataset': ds, 'weighted_bacc': w[ds], 'unweighted_bacc': u[ds], 'delta_pp': value}
                             for ds, value in delta.items())
    for candidate in ['block_balanced_ce', 'tensor_balanced_ce']:
        for protocol in ['validation', 'official_l2']:
            selected = [row for row in rows if row['candidate'] == candidate and row['protocol'] == protocol]
            if selected:
                values = [row['delta_macro_pp'] for row in selected]
                aggregates.append({'candidate': candidate, 'protocol': protocol, 'n': len(values),
                                   'paired_delta_mean_pp': statistics.mean(values),
                                   'paired_delta_sample_sd_pp': statistics.stdev(values) if len(values) > 1 else None})
    write_csv(output / 'class_balance_effect_pairs.csv', rows)
    write_csv(output / 'class_balance_effect_tasks.csv', tasks)
    (output / 'class_balance_effect_summary.json').write_text(json.dumps(aggregates, indent=2) + '\n')
    lines = ['', 'Class-balanced CE versus the same coefficient layout with ordinary CE:', '',
             '| Protocol | Seed | Candidate | Control | Difference (pp) | Worst task | Worst difference (pp) |',
             '|---|---:|---|---|---:|---|---:|']
    for row in rows:
        lines.append(f'| {row["protocol"]} | {row["seed"]} | {row["candidate"]} | {row["control"]} | '
                     f'{row["delta_macro_pp"]:+.3f} | {row["worst_task"]} | {row["worst_task_delta_pp"]:+.3f} |')
    lines += ['', '| Protocol | Candidate | Completed pairs | Mean difference (pp) | Sample SD (pp) |',
              '|---|---|---:|---:|---:|']
    for row in aggregates:
        spread = f'{row["paired_delta_sample_sd_pp"]:.3f}' if row['paired_delta_sample_sd_pp'] is not None else 'not estimated'
        lines.append(f'| {row["protocol"]} | {row["candidate"]} | {row["n"]}/3 | '
                     f'{row["paired_delta_mean_pp"]:+.3f} | {spread} |')
    lines += ['', 'Only completed pairs are included. Exact calibration IDs, task-update order, experts and training settings are matched. '
              'Per-task weighting effects are saved in `class_balance_effect_tasks.csv`.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/aligned_merge_v1'))
    parser.add_argument('--family', choices=['unweighted', 'class_balanced'], default='unweighted')
    args = parser.parse_args()
    output = args.root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    balanced = args.family == 'class_balanced'
    methods = ['block_balanced_ce', 'tensor_balanced_ce'] if balanced else ['block_ce', 'tensor_ce']
    prefix = 'class_balance_replication' if balanced else 'tensor_replication'
    pairs, task_rows, audits, pending = [], [], {}, []
    for seed in [0, 1, 2]:
        folder = args.root / f'seed{seed}'
        runs = {}
        for method in methods:
            path = folder / method / 'run.json'
            if path.exists():
                runs[method] = json.loads(path.read_text())
            else:
                pending.append(f'seed{seed}/{method}')
        if len(runs) != 2:
            continue
        baseline, candidate = [runs[method] for method in methods]
        for key in ['datasets', 'experts', 'seed', 'base_checkpoint_sha256', 'loss', 'lr',
                    'steps_requested', 'alpha_updates_actual', 'eval_every', 'batch_size', 'precision']:
            assert baseline[key] == candidate[key], (seed, key)
        assert baseline['seed'] == seed and baseline['alpha_updates_actual'] == 1600
        assert baseline.get('granularity', 'block') == 'block' and candidate['granularity'] == 'tensor'
        assert baseline['num_coefficients'] == 528 and candidate['num_coefficients'] == 6128
        assert baseline.get('class_balanced_ce', False) == candidate.get('class_balanced_ce', False) == balanced
        if balanced:
            assert baseline['calibration_class_counts'] == candidate['calibration_class_counts']
            assert baseline['ce_class_weights'] == candidate['ce_class_weights']
        calibration = [np.load(folder / method / 'calibration_indices.npz') for method in methods]
        assert all(np.array_equal(calibration[0][ds], calibration[1][ds]) for ds in baseline['datasets'])
        traces = [np.load(folder / method / 'metrics.npz') for method in methods]
        assert np.array_equal(traces[0]['step_dataset'], traces[1]['step_dataset'])
        audits[str(seed)] = {'matching_settings_and_experts': True, 'matching_calibration_ids': True,
                            'matching_dataset_update_order': True,
                            'run_sha256': {method: hashlib.sha256((folder / method / 'run.json').read_bytes()).hexdigest()
                                           for method in methods}}
        scores_by_protocol = {'validation': [{ds: run['best_val'][ds]['BAcc'] for ds in run['datasets']}
                                             for run in [baseline, candidate]]}
        official = [official_scores(folder, method, baseline['datasets']) for method in methods]
        if all(values is not None for values in official):
            scores_by_protocol['official_l2'] = official
        else:
            pending.extend(f'seed{seed}/official_l2/{method}' for method, values in zip(methods, official)
                           if values is None)
        for protocol, (reference, merged) in scores_by_protocol.items():
            differences = {ds: 100 * (merged[ds] - reference[ds]) for ds in reference}
            macro_delta = statistics.mean(differences.values())
            pairs.append({'protocol': protocol, 'seed': seed,
                          'block_macro_bacc': statistics.mean(reference.values()),
                          'tensor_macro_bacc': statistics.mean(merged.values()),
                          'delta_macro_pp': macro_delta, 'worst_task_delta_pp': min(differences.values()),
                          'worst_task': min(differences, key=differences.get),
                          'tasks_down': sum(value < 0 for value in differences.values()),
                          'block_selected_step': baseline['best_step'], 'tensor_selected_step': candidate['best_step']})
            task_rows.extend({'protocol': protocol, 'seed': seed, 'dataset': ds,
                              'block_bacc': reference[ds], 'tensor_bacc': merged[ds], 'delta_pp': delta}
                             for ds, delta in differences.items())
    aggregates = {}
    for protocol in ['validation', 'official_l2']:
        rows = [row for row in pairs if row['protocol'] == protocol]
        if not rows:
            continue
        aggregates[protocol] = {'seeds': [row['seed'] for row in rows], 'n': len(rows)}
        for key in ['block_macro_bacc', 'tensor_macro_bacc', 'delta_macro_pp']:
            values = [row[key] for row in rows]
            aggregates[protocol][key] = {'mean': statistics.mean(values),
                                        'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
    dynamics = summarize_dynamics(args.root, output, methods, prefix)
    summary = {'status': 'complete' if not pending else 'incomplete', 'pending': pending,
               'family': args.family, 'methods': methods,
               'aggregates': aggregates, 'paired_audits': audits,
               'dynamics': dynamics,
               'interpretation': 'Training seeds are repetitions; heterogeneous tasks are not independent seeds. Seed0 was exploratory; seeds1/2 use the predeclared paired configuration.'}
    write_csv(output / f'{prefix}_pairs.csv', pairs)
    write_csv(output / f'{prefix}_tasks.csv', task_rows)
    (output / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Class-balanced CE replication' if balanced else '# Tensor-granularity replication', '',
              f'Methods: `{methods[0]}` and `{methods[1]}`.',
              'Original experts and exact existing calibration IDs; matched CE, optimizer, 1600 updates, and validation schedule.',
              'Block has 528 coefficients; the tensor variant has 6,128. Both export the same full LEAF architecture and fixed text readout.',
              'Checkpoint selection uses validation macro BAcc, then CE. Official L2 uses instruction seed42 and batch256.', '',
              '| Protocol | Seed | Block | Tensor | Difference (pp) | Worst task difference (pp) | Worst task | Tasks down |',
              '|---|---:|---:|---:|---:|---:|---|---:|']
    for row in pairs:
        report.append(f'| {row["protocol"]} | {row["seed"]} | {row["block_macro_bacc"]:.6f} | '
                      f'{row["tensor_macro_bacc"]:.6f} | {row["delta_macro_pp"]:+.3f} | '
                      f'{row["worst_task_delta_pp"]:+.3f} | {row["worst_task"]} | {row["tasks_down"]} |')
    report += ['', '| Protocol | Completed pairs | Block mean ± sample SD | Tensor mean ± sample SD | Paired difference, pp ± sample SD |',
               '|---|---:|---:|---:|---:|']
    for protocol, stats in aggregates.items():
        cells = []
        for key in ['block_macro_bacc', 'tensor_macro_bacc', 'delta_macro_pp']:
            values = stats[key]
            spread = f'{values["sample_sd"]:.6f}' if values['sample_sd'] is not None else 'not estimated'
            cells.append(f'{values["mean"]:.6f} ± {spread}')
        report.append(f'| {protocol} | {stats["n"]}/3 | ' + ' | '.join(cells) + ' |')
    if aggregates.get('official_l2', {}).get('n') == 3:
        worst = min((row for row in pairs if row['protocol'] == 'official_l2'), key=lambda row: row['worst_task_delta_pp'])
        report += ['', f'The completed official paired gain is {aggregates["official_l2"]["delta_macro_pp"]["mean"]:+.3f} pp, '
                   f'versus {aggregates["validation"]["delta_macro_pp"]["mean"]:+.3f} pp on validation. '
                   f'The worst official regression is {worst["worst_task"]} in seed{worst["seed"]}: '
                   f'{worst["worst_task_delta_pp"]:+.3f} pp.']
        if not balanced:
            report.append('The validation improvement is repeatable; a robust test improvement is not established.')
    report += ['', summary['interpretation'],
               ('Class weights are N_cal/(C*n_class), fixed from the original calibration labels; validation CE is unweighted. '
                'The earlier unweighted official results were known during development.' if balanced else
                'The seed0 tensor-scale control is reported separately; it has not been replicated across seeds.'),
               'These candidates do not establish a CPSC effect or a new semantic-alignment mechanism.', '',
               'Pending: ' + (', '.join(pending) if pending else 'none') + '.', '',
               '| Seed | Tensor above Block at same update | Median same-update gain (pp) | Final gain (pp) | Tensor final minus selected (pp) |',
               '|---|---:|---:|---:|---:|']
    for row in dynamics:
        report.append(f'| {row["seed"]} | {row["checkpoints_tensor_above_same_step_block"]}/{row["post_init_checkpoints"]} | '
                      f'{row["median_same_step_gain_pp"]:+.3f} | {row["final_step_gain_pp"]:+.3f} | '
                      f'{row["tensor_final_minus_selected_pp"]:+.3f} |')
    report += ['', 'These checkpoint comparisons describe within-run behavior, not independent statistical evidence.', '',
               f'![Validation trajectories]({prefix}_dynamics.png)', '']
    report += reference_and_cost_context(args.root, output, methods, prefix)
    if balanced:
        report += compare_class_balance_effect(args.root, output)
    (output / f'{prefix}_report.md').write_text('\n'.join(report))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
