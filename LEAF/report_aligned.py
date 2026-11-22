"""Summarize observed validation results and learning dynamics, without test selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize_hidden_alignment(root, completed, baseline, output):
    """Compare early/late fitting windows; these are not held-out errors."""
    summaries, task_rows, layer_rows = {}, [], []
    for name in ['expert_merging_adapted', 'block_hidden_ce']:
        if name not in completed:
            continue
        run = completed[name]
        data = [json.loads(line) for line in (root / name / 'dynamics.jsonl').read_text().splitlines()]
        assert len(data) == run['alpha_updates_actual']
        windows = {'first_160': data[:160], 'last_160': data[-160:]}
        terms = ['ce', 'kd', 'hidden', 'anchor', 'total', 'grad_norm']
        means = {window: {term: float(np.mean([row[term] for row in records])) for term in terms}
                 for window, records in windows.items()}
        for window, records in windows.items():
            for ds in run['datasets']:
                task = [row for row in records if row['dataset'] == ds]
                task_rows.append({'method': name, 'window': window, 'dataset': ds,
                                  'updates': len(task),
                                  **{term: float(np.mean([row[term] for row in task])) for term in terms}})
            total_hidden = sum(sum(row['hidden_by_layer'].values()) for row in records)
            for layer in run['hidden_alignment']['layers']:
                layer_rows.append({'method': name, 'window': window, 'layer': layer,
                    'mean_mse': float(np.mean([row['hidden_by_layer'][layer] for row in records])),
                    'share_of_hidden_loss': sum(row['hidden_by_layer'][layer] for row in records) / total_hidden})
        delta = {ds: 100 * (metrics['BAcc'] - baseline['best_val'][ds]['BAcc'])
                 for ds, metrics in run['best_val'].items()}
        macro_delta = 100 * (run['best_val_macro_BAcc'] - baseline['best_val_macro_BAcc'])
        schedule = np.load(root / name / 'metrics.npz')['step_dataset']
        baseline_schedule = np.load(root / 'block_ce/metrics.npz')['step_dataset']
        summaries[name] = {'updates': len(data), 'window_means': means,
            'validation_delta_block_pp': macro_delta, 'worst_task_delta_block_pp': min(delta.values()),
            'pilot_gate_passed': bool(macro_delta >= .5 and min(delta.values()) >= -3),
            'same_dataset_update_order_as_block': bool(np.array_equal(schedule, baseline_schedule)),
            'best_step': run['best_step'], 'source_sha256': run['code_sha256']}
    if summaries:
        (output / 'hidden_alignment_summary.json').write_text(json.dumps({
            'meaning': 'First/last 160 optimizer updates, 10 per task; calibration fitting loss, not held-out error. CE is diagnostic for expert_merging_adapted.',
            'runs': summaries}, indent=2) + '\n')
        for filename, rows in [('hidden_alignment_task_losses.csv', task_rows),
                               ('hidden_alignment_layer_losses.csv', layer_rows)]:
            with (output / filename).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return summaries


def summarize_class_balance(root, completed, histories, output):
    family = ['block_ce', 'tensor_ce', 'block_balanced_ce', 'tensor_balanced_ce']
    if not any(name in completed for name in family[2:]):
        return [], []
    comparisons, losses, task_losses = [], [], []
    for candidate, control in [('block_balanced_ce', 'block_ce'),
                               ('tensor_balanced_ce', 'tensor_ce'),
                               ('tensor_balanced_ce', 'block_balanced_ce'),
                               ('tensor_balanced_ce', 'block_ce')]:
        if candidate not in completed or control not in completed:
            continue
        run, ref = completed[candidate], completed[control]
        for key in ['datasets', 'experts', 'seed', 'lr', 'batch_size', 'alpha_updates_actual', 'eval_every']:
            assert run[key] == ref[key], (candidate, control, key)
        cal, original = [np.load(root / name / 'calibration_indices.npz') for name in [candidate, control]]
        assert all(np.array_equal(cal[ds], original[ds]) for ds in run['datasets'])
        schedules = [np.load(root / name / 'metrics.npz')['step_dataset'] for name in [candidate, control]]
        assert np.array_equal(*schedules)
        delta = {ds: 100 * (run['best_val'][ds]['BAcc'] - ref['best_val'][ds]['BAcc']) for ds in run['datasets']}
        comparisons.append({'candidate': candidate, 'control': control,
                            'delta_macro_pp': float(np.mean(list(delta.values()))),
                            'worst_task': min(delta, key=delta.get), 'worst_task_delta_pp': min(delta.values()),
                            'tasks_down': sum(value < 0 for value in delta.values()),
                            'settings_calibration_and_task_order_match': True})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in family:
        if name not in completed:
            continue
        data = [json.loads(line) for line in (root / name / 'dynamics.jsonl').read_text().splitlines()]
        for label, window in [('first160', data[:160]), ('last160', data[-160:])]:
            losses.append({'method': name, 'window': label,
                           'mean_fitting_ce': float(np.mean([row['ce'] for row in window])),
                           'mean_unweighted_ce': float(np.mean([row.get('unweighted_ce', row['ce']) for row in window]))})
            for ds in completed[name]['datasets']:
                task = [row for row in window if row['dataset'] == ds]
                task_losses.append({'method': name, 'window': label, 'dataset': ds, 'updates': len(task),
                                    'mean_fitting_ce': float(np.mean([row['ce'] for row in task])),
                                    'mean_unweighted_ce': float(np.mean([row.get('unweighted_ce', row['ce']) for row in task]))})
        history = histories[name]
        axes[0].plot([row['step'] for row in history], [100 * row['val_macro_BAcc'] for row in history], label=name)
        windows = [data[i:i+160] for i in range(0, len(data), 160)]
        axes[1].plot([window[-1]['step'] for window in windows],
                     [np.mean([row.get('unweighted_ce', row['ce']) for row in window]) for window in windows], label=name)
    for ax, title, ylabel in zip(axes, ['Completed-run validation trajectories', 'Calibration CE, 160-update means'],
                                 ['Macro BAcc (%)', 'Unweighted CE']):
        ax.set(title=title, xlabel='Optimizer updates', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'class_balance_dynamics.png', dpi=170)
    plt.close(fig)
    for filename, rows in [('class_balance_comparisons.csv', comparisons), ('class_balance_loss_windows.csv', losses),
                           ('class_balance_task_loss_windows.csv', task_losses)]:
        if rows:
            with (output / filename).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return comparisons, losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/aligned_merge_v1/seed0'))
    args = p.parse_args()
    output = args.root / 'analysis'
    output.mkdir(exist_ok=True)
    baseline_name = 'block_ce' if (args.root / 'block_ce/run.json').exists() else 'merge_block_mseed0'
    baseline = json.loads((args.root / baseline_name / 'run.json').read_text())
    names = ['merge_block_mseed0', 'merge_cpsc_block', 'block_ce', 'cpsc_block_ce', 'group_scale', 'block_semkd',
             'cpsc_block_semkd', 'block_logitkd', 'cpsc_uniform', 'eta01', 'eta10',
             'expert_merging_adapted', 'block_hidden_ce',
             'regmean_adapted', 'regmeanpp_adapted',
             'ties_fixed_adapted', 'ties_block_ce_adapted',
             'tensor_ce', 'tensor_scale_ce',
             'block_balanced_ce', 'tensor_balanced_ce',
             'block_balanced_avg', 'tensor_balanced_avg',
             'repair_base', 'repair_block', 'repair_cpsc', 'repair_block_fp32', 'repair_cpsc_fp32',
             'repair_base_full', 'repair_block_full', 'repair_cpsc_full',
             'proj_block_output', 'proj_block_contrast', 'proj_block_random',
             'proj_base_contrast', 'proj_cpsc_contrast']
    rows, all_tasks, histories, completed = [], [], {}, {}
    for name in names:
        folder = args.root / name
        if (folder / 'history.json').exists():
            histories[name] = json.loads((folder / 'history.json').read_text())
        if not (folder / 'run.json').exists():
            continue
        run = json.loads((folder / 'run.json').read_text())
        completed[name] = run
        delta = {ds: 100 * (m['BAcc'] - baseline['best_val'][ds]['BAcc']) for ds, m in run['best_val'].items()}
        rows.append({'method': name, 'validation_bacc': run['best_val_macro_BAcc'],
                     'delta_block_pp': 100 * (run['best_val_macro_BAcc'] - baseline['best_val_macro_BAcc']),
                     'worst_task_delta_pp': min(delta.values()), 'tasks_down': sum(d < 0 for d in delta.values()),
                     'best_step': run['best_step'], 'seconds': run['seconds']['total'],
                     'parameter_retraining': name.startswith(('repair_', 'proj_')),
                     'backbone_trainable': run.get('backbone_trainable', False),
                     'trainable_parameters': run.get('trainable_parameters')})
        all_tasks.extend({'method': name, 'dataset': ds, 'bacc': run['best_val'][ds]['BAcc'],
                          'delta_block_pp': d} for ds, d in delta.items())
    for filename, data in [('validation_summary.csv', rows), ('per_dataset_validation.csv', all_tasks)]:
        if data:
            with (output / filename).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    repair_comparisons = []
    if 'repair_base' in completed:
        base_repair = completed['repair_base']
        for name, run in completed.items():
            if not name.startswith('repair_') or name.endswith('_full'):
                continue
            diffs = [100 * (metrics['BAcc'] - base_repair['best_val'][ds]['BAcc'])
                     for ds, metrics in run['best_val'].items()]
            repair_comparisons.append({'method': name,
                'gain_from_own_step_zero_pp': 100 * (run['best_val_macro_BAcc'] - histories[name][0]['val_macro_BAcc']),
                'delta_base_repair_pp': 100 * (run['best_val_macro_BAcc'] - base_repair['best_val_macro_BAcc']),
                'worst_task_delta_base_repair_pp': min(diffs), 'tasks_down_vs_base_repair': sum(d < 0 for d in diffs)})
        with (output / 'matched_repair_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(repair_comparisons[0]))
            writer.writeheader()
            writer.writerows(repair_comparisons)
    projection_comparisons = []
    for candidate, control in [
        ('proj_block_contrast', 'block_ce'),
        ('proj_block_contrast', 'proj_block_output'),
        ('proj_block_contrast', 'proj_block_random'),
        ('proj_block_contrast', 'proj_base_contrast'),
        ('proj_cpsc_contrast', 'proj_block_contrast'),
    ]:
        if candidate not in completed or control not in completed:
            continue
        run, ref = completed[candidate], completed[control]
        diffs = {ds: 100 * (metrics['BAcc'] - ref['best_val'][ds]['BAcc'])
                 for ds, metrics in run['best_val'].items()}
        projection_comparisons.append({
            'candidate': candidate, 'control': control,
            'delta_macro_pp': 100 * (run['best_val_macro_BAcc'] - ref['best_val_macro_BAcc']),
            'worst_task_delta_pp': min(diffs.values()),
            'worst_task': min(diffs, key=diffs.get),
            'tasks_down': sum(d < 0 for d in diffs.values()),
        })
    if projection_comparisons:
        with (output / 'matched_projection_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(projection_comparisons[0]))
            writer.writeheader()
            writer.writerows(projection_comparisons)
    full_repair_comparisons = []
    for candidate, control in [
        ('repair_block_full', 'repair_base_full'),
        ('repair_cpsc_full', 'repair_block_full'),
        ('repair_block_full', 'block_ce'),
        ('repair_block_full', 'repair_block_fp32'),
        ('repair_cpsc_full', 'repair_cpsc_fp32'),
    ]:
        if candidate not in completed or control not in completed:
            continue
        run, ref = completed[candidate], completed[control]
        diffs = {ds: 100 * (metrics['BAcc'] - ref['best_val'][ds]['BAcc'])
                 for ds, metrics in run['best_val'].items()}
        full_repair_comparisons.append({
            'candidate': candidate, 'control': control,
            'delta_macro_pp': 100 * (run['best_val_macro_BAcc'] - ref['best_val_macro_BAcc']),
            'worst_task_delta_pp': min(diffs.values()), 'worst_task': min(diffs, key=diffs.get),
            'tasks_down': sum(d < 0 for d in diffs.values()),
        })
    if full_repair_comparisons:
        with (output / 'matched_full_repair_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(full_repair_comparisons[0]))
            writer.writeheader()
            writer.writerows(full_repair_comparisons)
    if rows:
        datasets = list(baseline['best_val'])
        values = {(r['method'], r['dataset']): r['delta_block_pp'] for r in all_tasks}
        matrix = np.array([[values[r['method'], ds] for ds in datasets] for r in rows])
        bound = max(float(np.abs(matrix).max()), .1)
        fig, ax = plt.subplots(figsize=(15, max(3.5, .45 * len(rows) + 2)))
        im = ax.imshow(matrix, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(len(datasets)), datasets, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(rows)), [r['method'] for r in rows], fontsize=8)
        for i in range(len(rows)):
            for j in range(len(datasets)):
                ax.text(j, i, f'{matrix[i, j]:+.1f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(matrix[i, j]) > .6 * bound else 'black')
        ax.set_title(f'Validation task changes relative to {baseline_name}; selected checkpoints')
        fig.colorbar(im, ax=ax, label='BAcc difference (percentage points)')
        fig.tight_layout()
        fig.savefig(output / 'per_task_changes.png', dpi=180)
        plt.close(fig)
    trajectory = {}
    for name, history in histories.items():
        updated = [row for row in history if row['step'] > 0]
        if len(updated) < 2:
            continue
        trajectory[name] = {
            ds: {'min_bacc': min(row['val'][ds]['BAcc'] for row in updated),
                 'max_bacc': max(row['val'][ds]['BAcc'] for row in updated),
                 'peak_to_peak_pp': 100 * np.ptp([row['val'][ds]['BAcc'] for row in updated])}
            for ds in updated[0]['val']}
    (output / 'validation_trajectory_ranges.json').write_text(json.dumps({
        'meaning': 'Variation across checkpoints within one run; not seed uncertainty',
        'runs': trajectory}, indent=2) + '\n')
    invariance_path = args.root.parent / 'inputs' / args.root.name / 'invariance_all_eta1_gpu.json'
    alignment_rows = []
    if invariance_path.exists():
        invariance = json.loads(invariance_path.read_text())
        for ds, data in invariance['datasets'].items():
            alignment_rows.extend({'dataset': ds, 'protocol_split': key, **check}
                                  for key, check in data['checks'].items())
        with (output / 'alignment_checks.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(alignment_rows[0]))
            writer.writeheader()
            writer.writerows(alignment_rows)
    fig, grid = plt.subplots(2, 2, figsize=(14, 9))
    axes = grid.ravel()
    for name, history in histories.items():
        if name.startswith('regmean'):
            continue  # Layer solves are not optimizer steps.
        axis = axes[3 if name.endswith('_full') else 2 if name.startswith('proj_') else 1 if name.startswith('repair_') else 0]
        axis.plot([r['step'] for r in history], [100 * r['val_macro_BAcc'] for r in history],
                  marker='.', linestyle='--' if name.startswith('merge_') else '-', label=name)
    if any(name.startswith('proj_') for name in histories):
        axes[2].axhline(100 * baseline['best_val_macro_BAcc'], color='black', linestyle=':', label=baseline_name)
    for ax, title in zip(axes, ['Coefficient optimization', 'Full shared-interface CE repair', 'Final-projection CE repair', 'Backbone + interface CE repair']):
        ax.set(xlabel='Optimizer updates', ylabel='Validation macro BAcc (%)', title=title)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'Awaiting first validation result', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / 'validation_dynamics.png', dpi=180)
    plt.close(fig)
    fig, grid = plt.subplots(2, 2, figsize=(14, 8))
    axes = grid.ravel()
    for name in names:
        path = args.root / name / 'dynamics.jsonl'
        if not path.exists():
            continue
        data = [json.loads(line) for line in path.read_text().splitlines()]
        rounds = [data[i:i+16] for i in range(0, len(data) - 15, 16)]
        x = [r[-1]['step'] for r in rounds]
        axes[0].plot(x, [np.mean([v['ce'] for v in r]) for r in rounds], label=name, alpha=.8)
        if 'kd' in data[0] and any(v['kd'] for v in data):
            axes[1].plot(x, [np.mean([v['kd'] for v in r]) for r in rounds], label=name)
        for axis, key in [(axes[2], 'hidden'), (axes[3], 'anchor')]:
            if key in data[0]:
                axis.plot(x, [np.mean([v[key] for v in r]) for r in rounds], label=name)
    for ax, title in zip(axes, ['Calibration CE (diagnostic for label-free fitting)', 'Semantic / logit KD (raw loss)',
                              'Hidden-state MSE (raw mean across layers)', 'Coefficient anchor (raw mean absolute change)']):
        ax.set(xlabel='Optimizer updates', ylabel='Loss', title=title)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No completed updates for this loss', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / 'loss_dynamics.png', dpi=180)
    plt.close(fig)
    hidden_summaries = summarize_hidden_alignment(args.root, completed, baseline, output)
    balance_comparisons, balance_losses = summarize_class_balance(args.root, completed, histories, output)
    report = [f'# {args.root.name} observed validation results', '',
              'The first tables use the existing validation/training readout. Official L2 test results are reported separately below.', '',
              f'Delta reference: `{baseline_name}`. Legacy `merge_*` runs used the previous numerical settings; current runs disable TF32.', '',
              '| Method | BAcc | Δ Block (pp) | Worst task Δ (pp) | Tasks down | Selected step / layer | Wall seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append(f'| {r["method"]} | {r["validation_bacc"]:.6f} | {r["delta_block_pp"]:+.3f} | '
                      f'{r["worst_task_delta_pp"]:+.3f} | {r["tasks_down"]} | {r["best_step"]} | {r["seconds"]:.1f} |')
    if repair_comparisons:
        report += ['', '| Repair initialization | Gain from own step 0 (pp) | Δ Base repair (pp) | Worst task vs Base repair (pp) |',
                   '|---|---:|---:|---:|']
        for row in repair_comparisons:
            report.append(f'| {row["method"]} | {row["gain_from_own_step_zero_pp"]:+.3f} | '
                          f'{row["delta_base_repair_pp"]:+.3f} | {row["worst_task_delta_base_repair_pp"]:+.3f} |')
    if projection_comparisons:
        report += ['', 'Preset projection comparisons (all projection variants use FP32 and the same 400-update budget):', '',
                   '| Candidate | Control | Macro difference (pp) | Worst task difference (pp) | Worst task | Tasks down |',
                   '|---|---|---:|---:|---|---:|']
        for row in projection_comparisons:
            report.append(f'| {row["candidate"]} | {row["control"]} | {row["delta_macro_pp"]:+.3f} | '
                          f'{row["worst_task_delta_pp"]:+.3f} | {row["worst_task"]} | {row["tasks_down"]} |')
        report += ['', 'Projection repair changes the existing shared output layer; full-output and rank-matched random updates are controls.',
                   'Contrast restriction is a learned repair hypothesis, not the prediction-preserving CPSC transform.']
    if full_repair_comparisons:
        report += ['', 'Matched full-model repair comparisons:', '',
                   '| Candidate | Control | Macro difference (pp) | Worst task difference (pp) | Worst task | Tasks down |',
                   '|---|---|---:|---:|---|---:|']
        for row in full_repair_comparisons:
            report.append(f'| {row["candidate"]} | {row["control"]} | {row["delta_macro_pp"]:+.3f} | '
                          f'{row["worst_task_delta_pp"]:+.3f} | {row["worst_task"]} | {row["tasks_down"]} |')
    if hidden_summaries:
        report += ['', 'Hidden-alignment fitting dynamics (means over the first/last 160 updates):', '',
                   '| Method | Hidden MSE, first → last | CE, first → last | Δ Block (pp) | Pilot gate |',
                   '|---|---:|---:|---:|---|']
        for name, summary in hidden_summaries.items():
            first, last = summary['window_means']['first_160'], summary['window_means']['last_160']
            report.append(f'| {name} | {first["hidden"]:.4f} → {last["hidden"]:.4f} | '
                          f'{first["ce"]:.4f} → {last["ce"]:.4f} | {summary["validation_delta_block_pp"]:+.3f} | '
                          f'{"pass" if summary["pilot_gate_passed"] else "fail"} |')
        report += ['', 'Each window contains 10 updates per task; these are calibration fitting losses, not held-out errors.',
                   'Per-task and per-layer values are saved in `hidden_alignment_task_losses.csv` and `hidden_alignment_layer_losses.csv`.']
    if balance_comparisons:
        report += ['', 'Fixed class-balancing comparisons of validation-selected checkpoints:', '',
                   '| Candidate | Control | Macro difference (pp) | Worst task | Worst difference (pp) | Tasks down |',
                   '|---|---|---:|---|---:|---:|']
        for row in balance_comparisons:
            report.append(f'| {row["candidate"]} | {row["control"]} | {row["delta_macro_pp"]:+.3f} | '
                          f'{row["worst_task"]} | {row["worst_task_delta_pp"]:+.3f} | {row["tasks_down"]} |')
        report += ['', 'Matched settings, calibration IDs and task-update order are checked before comparison. '
                   'Weighted fitting CE and diagnostic unweighted CE are saved in `class_balance_loss_windows.csv` '
                   'and per task in `class_balance_task_loss_windows.csv`; '
                   'their first/last160-update windows each contain10 updates per task. These fitting losses are not held-out errors.', '',
                   '![Class-balancing trajectories](class_balance_dynamics.png)', '']
    report += ['', 'Repair methods use 400 extra updates; coefficient methods use 1,600 updates.',
               'Names ending in `_full` train the backbone and shared interface; other repair/projection runs freeze the backbone.',
               '`expert_merging_adapted` fits hidden states/logits plus an anchor without hard-label CE; `block_hidden_ce` adds hard CE. Both use labeled validation selection.',
               'These EEG adaptations use the existing 33 groups and fixed budget; they do not implement Expert Merging++ chunk allocation.',
               'RegMean/RegMean++ use fixed closed-form layer solves with alpha=0.95 and zero optimizer updates; their final layer index is 28. They merge all attention/MLP maps and average other active parameters.',
               'TIES variants use global 20% retention, mass-sign election, and a scale-1 disjoint mean. The fixed variant has zero updates; the adapted Block-CE variant learns the existing 528 coefficients over transformed vectors.',
               'Tensor granularity uses the 383 active parameter tensors: tensor_ce fits 6,128 expert/tensor coefficients and tensor_scale_ce fits 383 uniform-delta scales, with the same CE and 1,600-update budget.',
               'Class-balanced CE variants weight each calibration example by N_cal/(C*n_class), with fixed counts from the original calibration pool. Validation selection remains macro BAcc then unweighted CE; training traces also retain unweighted CE.',
               'Parameter-tensor coefficients are an established adaptive-merging design; [AdaMerging reference](https://github.com/EnnengYang/AdaMerging/blob/cea1165ad900113b8a52ae5ff190626576635f9b/src/main_layer_wise_adamerging.py) uses tensor-level coefficients with clamping and an entropy objective. This supervised signed-CE variant is not a claim of granularity novelty.',
               'Compare repair methods with each other to isolate initialization/merging effects.', '',
               'Pilot threshold: at least +0.5 pp macro and no task worse than −3 pp relative to Block.',
               'A single expert seed cannot establish reproducibility or an independent CPSC contribution.', '',
               '![Validation dynamics](validation_dynamics.png)', '', '![Loss dynamics](loss_dynamics.png)', '',
               '![Per-task validation changes](per_task_changes.png)', '',
               'Not completed: ' + ', '.join(n for n in names if n not in {r['method'] for r in rows}) + '.', '']
    if alignment_rows:
        report += [f'Completed {len(alignment_rows)} CPSC numerical checks; '
                   f'{sum(r["prediction_changes"] for r in alignment_rows)} prediction changes; '
                   f'maximum centered-logit error {max(r["centered_logits_max_abs"] for r in alignment_rows):.3g}.', '',
                   'Alignment residual increased in the following cases (all checks are in `alignment_checks.csv`):', '',
                   '| Dataset | Instruction / split | Before | After |', '|---|---|---:|---:|']
        for row in alignment_rows:
            if row['complement_residual_after'] >= row['complement_residual_before']:
                report.append(f'| {row["dataset"]} | {row["protocol_split"]} | '
                              f'{row["complement_residual_before"]:.6f} | {row["complement_residual_after"]:.6f} |')
        report.append('')
    component_path = output / 'component_swaps.json'
    if component_path.exists():
        component = json.loads(component_path.read_text())
        base_bacc = np.mean([metrics['BAcc'] for metrics in component['base_control'].values()])
        report += ['', '## Validation component swaps', '',
                   'No fitting. B denotes backbone and A the shared interface; 0 is Base and M is the selected merge.', '',
                   '| Merge | B0/A0 | B0/AM | BM/A0 | BM/AM | Interaction (pp) |',
                   '|---|---:|---:|---:|---:|---:|']
        for name, values in component['runs'].items():
            report.append(f'| {name} | {base_bacc:.6f} | {values["B0_AM"]["macro_BAcc"]:.6f} | '
                          f'{values["BM_A0"]["macro_BAcc"]:.6f} | {values["BM_AM"]["macro_BAcc"]:.6f} | '
                          f'{values["interaction_pp"]:+.3f} |')
        report += ['', 'Interaction = BM/AM − BM/A0 − B0/AM + B0/A0. Non-additivity alone does not establish semantic drift.', '']
    gradient_path = output / 'tensor_gradient_diagnostic/summary.json'
    if gradient_path.exists():
        diagnostic = json.loads(gradient_path.read_text())
        gradient = diagnostic['gradients']['macro_mean']
        report += ['', '## Calibration gradient under coefficient tying', '',
                   f'The selected Block checkpoint is reproduced with {diagnostic["coefficients"]:,} '
                   f'expanded coefficients over {diagnostic["parameter_tensors"]} parameter tensors '
                   f'(maximum reconstruction error {diagnostic["state_reconstruction_max_abs"]:.3g}).',
                   f'For macro calibration CE, {100 * gradient["untied_fraction_squared_l2"]:.2f}% of '
                   'the squared Euclidean coefficient-gradient norm lies outside the original tied subspace.',
                   'This is a local measurement in coefficient coordinates. No fitting was performed; held-out gains remain an experimental question.',
                   'Full task/group measurements and raw arrays are in `tensor_gradient_diagnostic/`.', '']
    official_summary = args.root / 'official_l2/summary.csv'
    if official_summary.exists():
        official = list(csv.DictReader(official_summary.open()))
        references_path = output / 'poc_official_references.json'
        references = json.loads(references_path.read_text()) if references_path.exists() else []
        scores = {row['method']: float(row['source_macro_bacc']) for row in official}
        block_score = scores.get('block_ce')
        report += ['', '## Official L2 test results', '',
                   'Fixed checkpoints, instruction seed 42, batch 256; no checkpoint selection on test.',
                   'Macros below average the official four-decimal per-task CSV values.', '',
                   '| Method | Source macro BAcc | Difference from fresh Block (pp) |', '|---|---:|---:|']
        for row in references + official:
            score = float(row['source_macro_bacc'])
            label = ('POC ' if row in references else '') + row['method']
            delta = f'{100 * (score - block_score):+.3f}' if block_score is not None else 'pending'
            report.append(f'| {label} | {score:.6f} | {delta} |')
        report += ['', 'The POC references predate this experiment. New official receipts record selected steps and checkpoint hashes.', '']
        task_rows = list(csv.DictReader((args.root / 'official_l2/per_dataset.csv').open()))
        task_scores = {}
        for row in task_rows:
            task_scores.setdefault(row['method'], {})[row['dataset']] = float(row['bacc'])
        pairs = [
            ('repair_block_fp32', 'block_ce'),
            ('repair_block_fp32', 'repair_base'),
            ('repair_cpsc_fp32', 'repair_block_fp32'),
            ('proj_block_contrast', 'block_ce'),
            ('proj_block_contrast', 'proj_block_output'),
            ('proj_block_contrast', 'proj_block_random'),
            ('proj_cpsc_contrast', 'proj_block_contrast'),
            ('repair_base_full', 'repair_base'),
            ('repair_block_full', 'block_ce'),
            ('repair_cpsc_full', 'cpsc_block_ce'),
            ('expert_merging_adapted', 'block_ce'),
            ('block_hidden_ce', 'block_ce'),
            ('block_hidden_ce', 'expert_merging_adapted'),
            ('tensor_ce', 'block_ce'),
            ('tensor_scale_ce', 'block_ce'),
            ('tensor_ce', 'tensor_scale_ce'),
            ('block_balanced_ce', 'block_ce'),
            ('tensor_balanced_ce', 'tensor_ce'),
            ('tensor_balanced_ce', 'block_balanced_ce'),
        ]
        comparisons = []
        for candidate, control in pairs:
            if candidate not in task_scores or control not in task_scores:
                continue
            assert set(task_scores[candidate]) == set(task_scores[control])
            differences = {ds: 100 * (score - task_scores[control][ds])
                           for ds, score in task_scores[candidate].items()}
            comparisons.append({'candidate': candidate, 'control': control,
                'delta_macro_pp': float(np.mean(list(differences.values()))),
                'worst_task_delta_pp': min(differences.values()),
                'worst_task': min(differences, key=differences.get),
                'tasks_down': sum(value < 0 for value in differences.values())})
        if comparisons:
            with (output / 'official_matched_comparisons.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
            report += ['Matched official comparisons of validation-selected checkpoints:', '',
                       '| Candidate | Control | Macro difference (pp) | Worst task difference (pp) | Worst task | Tasks down |',
                       '|---|---|---:|---:|---|---:|']
            for row in comparisons:
                report.append(f'| {row["candidate"]} | {row["control"]} | {row["delta_macro_pp"]:+.3f} | '
                              f'{row["worst_task_delta_pp"]:+.3f} | {row["worst_task"]} | {row["tasks_down"]} |')
            report.append('')
    (output / 'report.md').write_text('\n'.join(report))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
