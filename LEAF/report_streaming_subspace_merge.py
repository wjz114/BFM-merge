"""Paired order results and the measured-geometry versus random control."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from run_streaming_subspace_merge import write_json


def contrast(a, b, tasks):
    d = {t: 100 * (a[t]['BAcc'] - b[t]['BAcc']) for t in tasks}
    x = np.asarray(list(d.values()))
    leave_one = (x.sum() - x) / (len(x) - 1)
    return {'gain_pp': float(x.mean()), 'per_task_pp': d, 'improved': int((x > 0).sum()),
            'declined': int((x < 0).sum()), 'tied': int((x == 0).sum()),
            'task_deletion_range_pp': [float(leave_one.min()), float(leave_one.max())]}


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_subspace_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    plan = json.loads((root / 'plan.json').read_text())
    tasks = plan['tasks']
    runs, sources = {}, {}
    for order in ['forward', 'reverse']:
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and len(run['stages']) == 16
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['updates'] == plan['updates_per_arm'] and run['training_examples'] == plan['training_examples_per_arm']
            total_forward = 0
            for stage in run['stages'][1:]:
                d = folder / f"stage_{stage['stage']:02d}"
                assert sha256(d / 'merged.ckpt') == stage['checkpoint_sha256']
                c = torch.load(d / 'coefficients.pt', map_location='cpu', weights_only=True)
                assert torch.equal(c['snapshots'].double().mean(0).float(), c['mean'])
                assert c['mean'].numel() == plan['per_arm_coefficients'][condition]
                assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0
                           for v in stage['export_checks'].values())
                audit = stage['input_split']
                assert audit['old_tasks'] == run['order'][:stage['stage'] - 1]
                assert list(audit['layers']) == plan['split_layers']
                total_forward += audit['extra_forward_examples']
            assert total_forward == plan['old_response_forward_examples_per_arm']
            runs[order, condition] = run
            sources[str(folder / 'run.json')] = sha256(folder / 'run.json')
        base = runs[order, 'incoming_rows']
        for condition in plan['conditions']:
            run = runs[order, condition]
            assert run['dataset_audits'] == base['dataset_audits']
            assert [s.get('schedule_hash') for s in run['stages']] == [s.get('schedule_hash') for s in base['stages']]
            assert [s['input_split']['selected_ids_hash'] for s in run['stages'][1:]] == [s['input_split']['selected_ids_hash'] for s in base['stages'][1:]]
    macro, differences, retention = {}, {}, {}
    for order in ['forward', 'reverse']:
        values = {c: runs[order, c]['stages'][-1]['validation'] for c in plan['conditions']}
        macro[order] = {c: float(np.mean([v[t]['BAcc'] for t in tasks])) for c, v in values.items()}
        differences[order] = {f'{a}_minus_{b}': contrast(values[a], values[b], tasks)
                              for a, b in [('activation_split', 'incoming_rows'), ('random_split', 'incoming_rows'),
                                           ('activation_split', 'random_split')]}
        retention[order] = {}
        for c in plan['conditions']:
            run = runs[order, c]
            old, new = run['order'][:12], run['order'][12:]
            before, after = run['stages'][11]['validation'], run['stages'][-1]['validation']
            retention[order][c] = {
                'old12_final_BAcc': float(np.mean([after[t]['BAcc'] for t in old])),
                'old12_stage12_to16_change_pp': float(np.mean([100 * (after[t]['BAcc'] - before[t]['BAcc']) for t in old])),
                'new4_final_BAcc': float(np.mean([after[t]['BAcc'] for t in new]))}
    result = {'sources': sources, 'plan_sha256': sha256(root / 'plan.json'), 'macro_BAcc': macro,
              'contrasts': differences, 'last_block_retention': retention,
              'matched_training_and_probe_ids': True, 'native_export_and_coefficient_means_pass': True,
              'scientific_updates': sum(r['updates'] for r in runs.values()),
              'extra_activation_forward_examples': len(runs) * plan['old_response_forward_examples_per_arm'],
              'scope': 'One bank, two arrival orders, exposed development validation. Random norms/ranks match the real-activation construction within that evolving model; evolving models differ across arms. Activation energy is not a function-preservation guarantee. Task-deletion ranges are sensitivity, not confidence intervals.'}
    write_json(output, result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.6]})
    labels = plan['conditions']
    colors = ['#397da4', '#d18b36']
    for j, order in enumerate(['forward', 'reverse']):
        values = [100 * macro[order][c] for c in labels]
        axes[0].bar(np.arange(3) + (j - .5) * .36, values, width=.36, color=colors[j], label=order)
        for i, v in enumerate(values):
            axes[0].text(i + (j - .5) * .36, v + .04, f'{v:.3f}', ha='center', fontsize=8)
        d = differences[order]['activation_split_minus_random_split']['per_task_pp']
        axes[1].barh(np.arange(16) + (j - .5) * .36, [d[t] for t in tasks], height=.36, color=colors[j], label=order)
    all_values = [100 * v for m in macro.values() for v in m.values()]
    axes[0].set_ylim(min(all_values) - .5, max(all_values) + .5)
    axes[0].set_xticks(np.arange(3), ['Incoming rows', 'Activation split', 'Random split'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    axes[1].set_yticks(np.arange(16), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', lw=.8)
    axes[1].set_xlabel('Activation minus norm-matched random split (percentage points)')
    axes[1].legend()
    fig.suptitle('Incoming updates split by measured old-task input subspaces')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output.with_suffix('.' + ext), dpi=180)
    print(json.dumps({'macro_BAcc': macro, 'gain_pp': {o: {k: v['gain_pp'] for k, v in d.items()}
                                                     for o, d in differences.items()}}, indent=2))


if __name__ == '__main__':
    main()
