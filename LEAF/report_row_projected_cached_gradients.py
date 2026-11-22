"""Summarize every cached-gradient counterfactual without selecting a method."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'analysis/row_projected_cached_gradients.json'
    data = json.loads(source.read_text())
    assert data['new_leaf_forwards'] == data['new_backpropagations'] == data['new_optimizer_updates'] == 0
    assert sha(data['plan']['cache_manifest']) == data['input_sha256'][data['plan']['cache_manifest']]
    assert sha('diagnose_row_projected_cached_gradients.py') == data['worker_sha256']
    labels = data['labels']; tasks = labels[:-1]
    assert len(tasks) == 16 and labels[-1] == 'uniform_task_mean_gradient'
    metrics = ['projected_gradient_energy_fraction','update_direction_cosine',
        'update_only_gradient_cosine','project_both_gradient_cosine',
        'normal_contribution_norm_over_update_norm','tangent_normal_cosine',
        'update_only_restored_directional_derivative','project_both_restored_directional_derivative']
    results = {}
    for kind, rows in data['conditions'].items():
        assert list(rows) == labels
        summary = {}
        for metric in metrics:
            values = np.array([rows[t][metric] for t in tasks])
            assert np.isfinite(values).all()
            summary[metric] = dict(mean=float(values.mean()),median=float(np.median(values)),
                minimum=float(values.min()),maximum=float(values.max()))
        results[kind] = {'task_summaries':summary,
            'uniform_mean_gradient':rows[labels[-1]],
            'update_only_ascent_tasks':[t for t in tasks if rows[t]['update_only_restored_directional_derivative']>0],
            'project_both_ascent_tasks':[t for t in tasks if rows[t]['project_both_restored_directional_derivative']>0],
            'max_decomposition_error':max(v['decomposition_relative_l2_error'] for v in rows.values())}
    out = root/'analysis/row_projected_cached_gradients_summary.json'
    result = {'sources':{str(source):sha(source),str(Path(__file__)):sha(__file__)},
        'results':results,'all_tasks_and_uniform_mean_reported':True,
        'new_leaf_forwards':0,'new_backpropagations':0,'new_optimizer_updates':0,
        'scope':data['scope'],
        'interpretation_boundary':'A nonzero normal-gradient effect does not imply ascent or harm. A negative initial directional derivative does not establish finite-step or held-out improvement. No optimizer replacement or extra training arm is selected by this report.'}
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    fig,axes = plt.subplots(1,2,figsize=(13,7),sharey=True)
    ys = np.arange(len(labels))
    for kind,offset,color in [('row',-.13,'#0072B2'),('shuffled',.13,'#D55E00')]:
        rows = data['conditions'][kind]
        for ax,metric in zip(axes,['update_direction_cosine','projected_gradient_energy_fraction']):
            ax.scatter([rows[t][metric] for t in labels],ys+offset,s=25,color=color,label=kind)
    axes[0].set_yticks(ys,[*tasks,'Uniform task-mean gradient'],fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlim(-1.02,1.02); axes[1].set_xlim(-.02,1.02)
    axes[0].set_xlabel('Cosine: update-only vs gradient-and-update projection')
    axes[1].set_xlabel('Squared gradient norm retained by the expert space')
    axes[0].set_title('A. Dependence on the ambient gradient',loc='left')
    axes[1].set_title('B. Initial gradient capture',loc='left')
    for ax in axes:ax.axhline(15.5,color='#888888',ls=':',lw=.8);ax.legend(fontsize=9)
    fig.text(.5,.012,'Archived original512/task gradients at M0; first-step FP64 algebra; no new inference or training; all16 tasks retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['png','pdf']:fig.savefig(root/f'analysis/figures/row_projected_cached_gradients.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output':str(out),'sha256':sha(out),'results':results},indent=2))


if __name__=='__main__': main()
