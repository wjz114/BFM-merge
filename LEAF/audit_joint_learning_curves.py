"""Reconcile recorded Joint learning curves without running new inference."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    replication_path = root/'analysis/bank_coverage_replication_summary.json'
    replication = json.loads(replication_path.read_text())
    sources = {str(replication_path): sha256(replication_path)}
    banks, curves, late, tasks = {}, [], [], None
    for replica in replication['replicas']:
        seed = replica['bank_seed']
        folder = Path(f'runs/poc16/seed{seed}/joint')
        run_path, history_path, metrics_path = [folder/p for p in ['run.json', 'history.json', 'metrics.npz']]
        run, history = [json.loads(p.read_text()) for p in [run_path, history_path]]
        sources.update({str(p): sha256(p) for p in [run_path, history_path, metrics_path]})
        metrics = np.load(metrics_path)
        if tasks is None:
            tasks = run['datasets']
        assert run['datasets'] == tasks and len(tasks) == 16
        assert run['actual_optimizer_steps'] == 48000 and run['warmup_steps'] == 2400
        assert run['precision'] == 'bf16-mixed'
        assert len(metrics['step_dataset']) == 48000
        steps = np.array([row['step'] for row in history])
        assert np.array_equal(steps, metrics['val_round']*16)
        macro = np.array([row['val_macro_BAcc'] for row in history])
        assert np.array_equal(macro, metrics['val_macro_BAcc'])
        per_task = np.array([[row['val'][task]['BAcc'] for task in tasks] for row in history])
        assert np.max(np.abs(per_task.mean(1)-macro)) < 1e-12
        for i, task in enumerate(tasks):
            assert np.array_equal(per_task[:, i], metrics['valBAcc__'+task])
        selected = int(np.flatnonzero(steps == run['best_step'])[0])
        assert history[selected]['val'] == run['best_val']
        assert abs(macro[selected]-run['best_val_macro_BAcc']) < 1e-12
        half = int(np.flatnonzero(steps == 24000)[0])
        late_gain = (per_task[-1]-per_task[half])*100
        late.append(late_gain)
        curves.append(macro)
        crossing = {}
        for label, threshold in [('initial_merger', replica['initial_macro_bacc']),
                                 ('full_pool_repair', replica['fixed_average_macro_bacc']['full'])]:
            ids = np.flatnonzero(macro >= threshold)
            crossing[label] = int(steps[ids[0]]) if len(ids) else None
        banks[str(seed)] = {
            'steps': steps.tolist(), 'macro_bacc': macro.tolist(),
            'per_task_bacc': {task: per_task[:, i].tolist() for i, task in enumerate(tasks)},
            'fixed_endpoint_bacc': float(macro[-1]), 'selected_step': run['best_step'],
            'historical_selected_bacc': run['best_val_macro_BAcc'],
            'selected_checkpoint_fp32_reference': replica['joint_macro_bacc'],
            'selected_fp32_minus_historical_pp': 100*(replica['joint_macro_bacc']-run['best_val_macro_BAcc']),
            'second_half_gain_pp': float(late_gain.mean()),
            'second_half_per_task_gain_pp': dict(zip(tasks, late_gain.tolist())),
            'first_logged_crossing_of_reference': crossing,
            'reference_scope': 'Historical BF16 curve versus separate FP32 merger/repair references; retrospective crossing only, no optimal stopping or speedup claim.',
            'recipe': {k: run[k] for k in ['warmup_steps', 'lr_tokenizer', 'lr_rest', 'lr_schedule', 'weight_decay', 'precision', 'batch_size', 'batchnorm']},
            'early_checkpoint_files_available': False,
        }
        assert {p.name for p in folder.glob('*.ckpt')} == {'best.ckpt'}
    curves, late = np.array(curves), np.array(late)
    curve_mean, curve_sd = curves.mean(0), curves.std(0, ddof=1)
    late_mean, late_sd = late.mean(0), late.std(0, ddof=1)
    table_steps = [0, 1600, 3200, 4800, 9600, 19200, 24000, 32000, 38400, 48000]
    fixed_table = {str(step): {'mean_bacc': float(curve_mean[steps.tolist().index(step)]),
                              'sample_sd_pp': float(100*curve_sd[steps.tolist().index(step)])}
                   for step in table_steps}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'input_sha256': sources,
              'datasets': tasks, 'banks': banks, 'fixed_step_summary': fixed_table,
              'mean_second_half_gain_pp': float(late.mean()),
              'sample_sd_second_half_macro_gain_pp': float(late.mean(1).std(ddof=1)),
              'per_task_second_half': {task: {'mean_gain_pp': float(late_mean[i]), 'sample_sd_pp': float(late_sd[i]),
                                              'positive_banks': int((late[:, i] > 0).sum())}
                                        for i, task in enumerate(tasks)},
              'new_model_inference': False, 'new_EEG_data_read': False,
              'scope': 'Historical BF16 validation curves,31recorded points per seed, independently reconciled against per-task history and NPZ. Fixed last endpoint and fixed24000-to48000 interval are distinct from selected checkpoints. Only selected checkpoints remain, so intermediate FP32 replay is unavailable. The first1600steps lie within2400-step warmup; truncating this48000-step recipe is not an optimized short-budget comparator. No candidate selection, speedup, causal budget effect or independent confirmation is inferred.'}
    output = root/'analysis/historical_joint_learning_curve_audit.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1.1, 1]})
    for i, curve in enumerate(curves):
        ax.plot(steps, 100*curve, lw=1, alpha=.45, label=f'Historical Joint seed{i}')
    ax.plot(steps, 100*curve_mean, color='#172554', lw=2.2, label='Three-seed mean')
    ax.fill_between(steps, 100*(curve_mean-curve_sd), 100*(curve_mean+curve_sd), color='#93c5fd', alpha=.25)
    ax.axvspan(0, 2400, color='#fde68a', alpha=.45, label='Warmup(2400steps)')
    ax.axvline(24000, color='#64748b', ls=':', lw=1)
    ax.set_xlabel('Historical Joint optimizer updates')
    ax.set_ylabel('Validation macro BAcc (%)')
    ax.set_title('A. Recorded full trajectories (BF16)', loc='left', weight='bold')
    ax.legend(fontsize=8, loc='lower right')
    y = np.arange(len(tasks))
    bx.barh(y, late_mean, color=['#2563eb' if v >= 0 else '#d97706' for v in late_mean])
    for bank in late:
        bx.scatter(bank, y, color='#334155', marker='|', s=65, alpha=.7)
    bx.axvline(0, color='#64748b', lw=.8)
    bx.set_yticks(y, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel('BAcc gain from step24000 to48000 (pp)')
    bx.set_title('B. Fixed second-half gains; ticks=seeds', loc='left', weight='bold')
    fig.suptitle('Training duration and the historical Joint reference')
    fig.tight_layout(rect=(0, 0, 1, .96))
    figure = root/'analysis/figures/historical_joint_learning_curves'
    for suffix in ['.png', '.pdf']:
        fig.savefig(figure.with_suffix(suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'fixed_step_summary': fixed_table,
                      'mean_second_half_gain_pp': result['mean_second_half_gain_pp'],
                      'sample_sd_second_half_macro_gain_pp': result['sample_sd_second_half_macro_gain_pp'],
                      'crossings': {k: v['first_logged_crossing_of_reference'] for k, v in banks.items()}}, indent=2))


if __name__ == '__main__':
    main()
