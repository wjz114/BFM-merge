"""Report fixed1600/4800 CE means from each uninterrupted native trajectory."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_grouped_ce_controls import contrast


def verify_average(product, window):
    assert product['average_steps'] == window
    assert [s['step'] for s in product['average_snapshots']] == window
    states = []
    for spec in product['average_snapshots']:
        assert sha256(Path(spec['path'])) == spec['sha256']
        states.append(torch.load(spec['path'], map_location='cpu', weights_only=True, mmap=True))
    assert sha256(Path(product['checkpoint'])) == product['checkpoint_sha256']
    saved = torch.load(product['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    assert all(s.keys() == saved.keys() for s in states)
    for name, value in saved.items():
        if value.is_floating_point():
            expected = torch.stack([s[name].double() for s in states]).mean(0).to(value.dtype)
        else:
            assert all(torch.equal(s[name], states[0][name]) for s in states)
            expected = states[0][name]
        assert torch.equal(value, expected), name
    for check in product['export_checks'].values():
        assert check['max_abs_logit_error'] <= 1e-5
        assert all(r['margin_within_twice_error'] for r in check['changed_rows'])
    return {'all_saved_tensors_reconstruct_exactly': True, 'tensor_count': len(saved),
            'native_export_checks_pass': True}


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/duration4800_ce_v2_design.json'
    design = json.loads(design_path.read_text())
    tasks = design['datasets']
    sources = {str(p): sha256(p) for p in [design_path, Path(__file__), Path('report_grouped_ce_controls.py')]}
    for name, digest in design['source_sha256'].items():
        assert sha256(Path(name)) == digest, name
    schedules = [np.load(design[k]) for k in ['schedule', 'prefix_schedule']]
    for start in [0, 1600, 3200]:
        for key in ['task', 'local_indices']:
            assert np.array_equal(schedules[0][key][start:start+1600], schedules[1][key])
    runs, products, values, histories, checks, prefix = {}, {}, {}, {}, {}, {}
    for arm, spec in design['arms'].items():
        path, plan_path, old_path = Path(spec['outdir'])/'run.json', Path(spec['plan']), Path(spec['control'])
        run, plan, old = [json.loads(p.read_text()) for p in [path, plan_path, old_path]]
        assert sha256(old_path) == spec['control_sha256']
        assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
        assert plan['design_sha256'] == sha256(design_path)
        assert run['code_sha256'] == sha256(Path(design['worker']))
        assert run['updates'] == 4800 and run['training_examples'] == 614400
        assert run['schedule_sha256'] == design['schedule_sha256']
        assert old['schedule_sha256'] == design['prefix_schedule_sha256']
        assert run['initial_validation'] == old['initial_validation']
        assert run['trainable_parameters'] == 25751552 and run['fixed_tensors_exact']
        assert not run['engineering_smoke'] and not run['test_data_read'] and not run['test_inference_performed']
        assert not any(run['numerical_settings'].values())
        for name, digest in plan['source_sha256'].items():
            assert sha256(Path(name)) == digest, name
        for key in ['schedule', 'initial_checkpoint', 'calibration', 'coverage_audit']:
            assert sha256(Path(plan[key])) == plan[key+'_sha256']
        assert set(run['prefix_comparison']) == {'800', '960', '1120', '1280', '1440', '1600'}
        for step, check in run['prefix_comparison'].items():
            assert check['reference'] == plan['prefix_reference_snapshots'][step]
            assert sha256(Path(check['reference']['path'])) == check['reference']['sha256']
        history_path = path.with_name('history.json')
        history = json.loads(history_path.read_text())
        old_history = json.loads(Path(plan['prefix_reference_history']).read_text())
        assert [r['step'] for r in history] == list(range(0, 4801, 160))
        prefix[arm] = {'state_comparison': run['prefix_comparison'],
                       'historical_metric_differences': {str(a['step']): contrast(
                           np.array([a['validation'][t]['BAcc'] for t in tasks]),
                           np.array([b['validation'][t]['BAcc'] for t in tasks]), tasks)
                           for a,b in zip(history[:len(old_history)], old_history)}}
        for task in tasks:
            a, b = run['dataset_audits'][task], old['dataset_audits'][task]
            if arm == 'merged':
                new_extra, old_extra = {'labels', 'instruction_hash', 'prototypes_hash'}, {'teacher_hash', 'teacher_expert_sha256'}
                assert set(a)-set(b) == new_extra and set(b)-set(a) == old_extra
                assert {k:v for k,v in a.items() if k not in new_extra} == {k:v for k,v in b.items() if k not in old_extra}
            else:
                assert a == b
        for duration, product, window in [(1600, run['short_duration'], design['short_average_steps']),
                                           (4800, run, design['average_steps'])]:
            key = f'{arm}_{duration}'
            checks[key] = verify_average(product, window)
            products[key] = product
            values[key] = np.array([product['average_validation'][t]['BAcc'] for t in tasks])
        values[f'{arm}_historical1600'] = np.array([old['average_validation'][t]['BAcc'] for t in tasks])
        histories[arm], runs[arm] = history, run
        sources.update({str(p): sha256(p) for p in [path, plan_path, old_path, history_path]})
    assert runs['merged']['dataset_audits'] == runs['base']['dataset_audits']
    comparisons = {f'{arm}_4800_vs_1600': contrast(values[f'{arm}_4800'], values[f'{arm}_1600'], tasks) for arm in runs}
    comparisons.update({f'merged_vs_base_{duration}': contrast(values[f'merged_{duration}'], values[f'base_{duration}'], tasks) for duration in [1600, 4800]})
    comparisons['initialization_by_duration_interaction'] = contrast(
        values['merged_4800']-values['merged_1600'], values['base_4800']-values['base_1600'], tasks)
    historical_comparisons = {arm: contrast(values[f'{arm}_1600'], values[f'{arm}_historical1600'], tasks) for arm in runs}
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    sources[str(joint_path)] = sha256(joint_path)
    joint_mean = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    merger_mean = runs['merged']['initial_validation']['macro_bacc']
    means = {k: float(v.mean()) for k,v in values.items()}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.1, 1.2]})
    for arm, color in [('merged', '#0072b2'), ('base', '#d55e00')]:
        h = histories[arm]
        ax.plot([r['step'] for r in h], [100*r['macro_bacc'] for r in h], color=color, label=arm+' trajectory')
        ax.scatter([1600, 4800], [100*means[f'{arm}_{d}'] for d in [1600, 4800]], color=color, marker='D', s=55, zorder=3,
                   label=arm+' fixed means from this trajectory')
        ax.scatter([1600], [100*means[f'{arm}_historical1600']], edgecolors=color, facecolors='none', marker='o', s=90,
                   label=arm+' historical1600 mean')
    ax.axvline(1600, color='#999999', ls=':', lw=.8)
    ax.axhline(joint_mean*100, color='#555555', ls='--', label='Historical Joint reference')
    ax.set_xlabel('Additional CE updates')
    ax.set_ylabel('Full16 validation macro BAcc (%)')
    ax.set_title('A. One continuous trajectory per initialization', loc='left')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    y = np.arange(16)
    for offset, arm, color in [(-.18, 'merged', '#0072b2'), (.18, 'base', '#d55e00')]:
        row = comparisons[f'{arm}_4800_vs_1600']
        bx.barh(y+offset, [row['per_task_gain_pp'][t] for t in tasks], height=.34, color=color,
                label=f'{arm}: mean {row["macro_gain_pp"]:+.3f} pp')
    bx.axvline(0, color='#777777', lw=.8)
    bx.set_yticks(y, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel('4800-step mean minus1600-step mean BAcc (pp)')
    bx.set_title('B. Same-trajectory contrasts, every task retained', loc='left')
    bx.legend(fontsize=8)
    fig.suptitle('Fixed CE budget control: same117785 fitting rows, three times the exposures')
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .015, 'Single exposed bank; expert and coefficient construction costs are additional for the merged initialization.', ha='center', fontsize=9)
    figures = {}
    for suffix in ['png', 'pdf']:
        p = root/f'analysis/figures/duration4800_ce_v2_controls.{suffix}'
        assert not p.exists()
        fig.savefig(p, dpi=170, bbox_inches='tight')
        figures[str(p)] = sha256(p)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'sources': sources,
              'datasets': tasks, 'macro_bacc': means, 'primary_within_trajectory_contrasts': comparisons,
              'historical1600_reproducibility_contrasts': historical_comparisons, 'historical_prefix_comparisons': prefix,
              'joint_macro_bacc': joint_mean, 'initial_merger_macro_bacc': merger_mean,
              'fraction_joint_gap_closed_from_merger': {str(d): (means[f'merged_{d}']-merger_mean)/(joint_mean-merger_mean) for d in [1600, 4800]},
              'calibration_macro_bacc': {k: float(np.mean([r['average_calibration'][t]['BAcc'] for t in tasks])) for k,r in products.items()},
              'process_seconds_per4800_trajectory': {k:r['seconds'] for k,r in runs.items()},
              'budget': design['budget'], 'average_reconstruction_checks': checks, 'figures': figures,
              'scope': design['scope']+' Task-deletion ranges are descriptive sensitivity, not confidence intervals. Fixed means remain primary even if another recorded checkpoint scores higher.'}
    target = root/'analysis/duration4800_ce_v2_summary.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'report': str(target), 'sha256': sha256(target), 'macro_bacc': means,
                      'contrasts': {k:{n:v for n,v in row.items() if n not in ['per_task_gain_pp', 'leave_one_task_out_pp']} for k,row in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
