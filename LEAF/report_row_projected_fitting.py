"""Integrate the fixed training-stream diagnostic with the primary comparison."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256
from report_grouped_ce_controls import contrast


def main():
    root = Path('runs/paper_merge_v2')
    plan_path = root/'inputs/row_projected_fitting_plan_v2.json'
    plan = json.loads(plan_path.read_text())
    run_path = Path(plan['outdir'])/'run.json'
    run = json.loads(run_path.read_text())
    primary_path = Path(plan['primary_report'])
    primary = json.loads(primary_path.read_text())
    assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
    assert run['primary_report_sha256'] == sha256(primary_path)
    for name, digest in {**plan['source_sha256'], **plan['input_sha256'], **run['checkpoint_sources']}.items():
        assert sha256(Path(name)) == digest, name
    assert run['code_sha256'] == sha256(Path('diagnose_row_projected_fitting.py'))
    assert run['all_model_states_unchanged'] and run['optimizer_updates'] == 0
    assert not run['validation_eeg_read'] and not run['test_data_read']
    assert run['native_forward_examples'] == plan['budget']['native_forward_examples'] == 355890
    tasks = plan['datasets']
    assert set(run['records']) == set(tasks)
    metrics = ['weighted_ce', 'exposure_bacc', 'unique_fitting_bacc']
    arrays = {c: {m: np.array([run['records'][t]['models'][c][m] for t in tasks]) for m in metrics}
              for c in ['initial_cached', *plan['conditions']]}
    replay_max = max(v for row in run['records'].values() for v in row['initial_reference_absolute_errors'].values())
    comparisons = {}
    for a, b in [('row', 'full'), ('shuffled', 'full'), ('row', 'shuffled')]:
        loss = arrays[a]['weighted_ce']-arrays[b]['weighted_ce']
        comparisons[f'{a}_vs_{b}'] = {
            'weighted_ce_difference': float(loss.mean()),
            'ce_lower_tasks': int((loss < 0).sum()), 'ce_higher_tasks': int((loss > 0).sum()),
            'per_task_ce_difference': dict(zip(tasks, loss.tolist())),
            'exposure_bacc': contrast(arrays[a]['exposure_bacc'], arrays[b]['exposure_bacc'], tasks),
            'unique_fitting_bacc': contrast(arrays[a]['unique_fitting_bacc'], arrays[b]['unique_fitting_bacc'], tasks),
            'validation_bacc': primary['primary_contrasts'][f'{a}_vs_{b}']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
        'sources': {str(p): sha256(p) for p in [plan_path, run_path, primary_path]},
        'datasets': tasks, 'macro': run['macro'], 'comparisons': comparisons,
        'initial_reference_max_absolute_error': replay_max,
        'native_forward_examples': run['native_forward_examples'], 'seconds': run['seconds'],
        'optimizer_updates': 0, 'all_model_states_unchanged': True,
        'decision': 'The fixed norm-restored row-space recipe is not advanced as a paper method or expanded to new seeds. Row structure improves fitting and validation relative to shuffled structure but does not outperform full CE. Endpoint fitting does not separate constrained capacity from incomplete optimization; no optimizer or learning-rate rescue is selected by this report.',
        'scope': plan['scope']}
    path = root/'analysis/row_projected_fitting_summary.json'
    with path.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, color in [('row_vs_full', '#0072B2'), ('row_vs_shuffled', '#009E73')]:
        row = comparisons[key]
        x = [100*row['per_task_ce_difference'][t] for t in tasks]
        y = [row['validation_bacc']['per_task_gain_pp'][t] for t in tasks]
        axes[0].scatter(x, y, color=color, label=key.replace('_', ' '), s=25)
        train = [row['exposure_bacc']['per_task_gain_pp'][t] for t in tasks]
        axes[1].scatter(train, y, color=color, label=key.replace('_', ' '), s=25)
    for ax in axes:
        ax.axhline(0, color='#777777', lw=.7)
        ax.axvline(0, color='#777777', lw=.7)
        ax.set_ylabel('Validation BAcc difference (pp)')
        ax.legend(fontsize=8)
    axes[0].set_xlabel('Fitting CE difference × 100 (lower is better)')
    axes[1].set_xlabel('Fitting-stream BAcc difference (pp)')
    fig.suptitle('Fixed means: fitting differences and validation differences')
    fig.text(.5, .01, '16 tasks; one expert bank; endpoint diagnostics do not identify capacity or convergence limits.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for suffix in ['png', 'pdf']:
        fig.savefig(root/f'analysis/figures/row_projected_fitting.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'path': str(path), 'sha256': sha256(path), 'macro': run['macro'],
                      'initial_reference_max_absolute_error': replay_max}, indent=2))


if __name__ == '__main__':
    main()
