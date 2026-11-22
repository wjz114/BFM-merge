"""Report both fixed4800-step controls after exact1600-step prefix replay."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from d_downstream_qformer_ce import sha256
from report_grouped_ce_controls import contrast


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/duration4800_ce_design.json'
    design = json.loads(design_path.read_text())
    tasks = design['datasets']
    sources = {str(design_path): sha256(design_path),
               'report_grouped_ce_controls.py': sha256(Path('report_grouped_ce_controls.py'))}
    new_schedule, old_schedule = [np.load(design[key]) for key in ['schedule', 'prefix_schedule']]
    assert sha256(Path(design['schedule'])) == design['schedule_sha256']
    assert sha256(Path(design['prefix_schedule'])) == design['prefix_schedule_sha256']
    for start in [0, 1600, 3200]:
        for key in ['task', 'local_indices']:
            assert np.array_equal(new_schedule[key][start:start+1600], old_schedule[key])
    runs, values, histories = {}, {}, {}
    for arm, spec in design['arms'].items():
        path, plan_path, old_path = Path(spec['outdir'])/'run.json', Path(spec['plan']), Path(spec['control'])
        run, plan, old = [json.loads(p.read_text()) for p in [path, plan_path, old_path]]
        assert sha256(old_path) == spec['control_sha256']
        assert run['plan'] == plan and run['plan_sha256'] == sha256(plan_path)
        assert plan['design_sha256'] == sha256(design_path)
        assert run['code_sha256'] == sha256(Path(design['worker']))
        assert run['updates'] == 4800 and run['training_examples'] == 614400
        assert run['average_steps'] == design['average_steps']
        assert run['schedule_sha256'] == design['schedule_sha256']
        assert old['schedule_sha256'] == design['prefix_schedule_sha256']
        assert run['initial_validation'] == old['initial_validation']
        assert run['trainable_parameters'] == 25751552 and run['fixed_tensors_exact']
        assert not run['engineering_smoke'] and not run['test_data_read'] and not run['test_inference_performed']
        assert run['checkpoint_sha256'] == sha256(path.with_name('merged.ckpt'))
        assert old['checkpoint_sha256'] == sha256(old_path.with_name('merged.ckpt'))
        for check in run['export_checks'].values():
            assert check['max_abs_logit_error'] <= 1e-5
            assert all(r['margin_within_twice_error'] for r in check['changed_rows'])
        for name, digest in plan['source_sha256'].items():
            assert sha256(Path(name)) == digest, name
        assert sha256(Path(plan['initial_checkpoint'])) == plan['initial_checkpoint_sha256']
        assert set(run['prefix_replay']) == {'800', '960', '1120', '1280', '1440', '1600'}
        for step, check in run['prefix_replay'].items():
            assert check['all_validation_metrics_exact'] and not check['nonidentical_state_tensors']
            assert check['reference'] == plan['prefix_reference_snapshots'][step]
            assert sha256(Path(check['reference']['path'])) == check['reference']['sha256']
        history_path = path.with_name('history.json')
        history = json.loads(history_path.read_text())
        old_history = json.loads(Path(plan['prefix_reference_history']).read_text())
        assert history[:len(old_history)] == old_history
        histories[arm] = history
        for task in tasks:
            a, b = run['dataset_audits'][task], old['dataset_audits'][task]
            if arm == 'merged':
                new_extra, old_extra = {'labels', 'instruction_hash', 'prototypes_hash'}, {'teacher_hash', 'teacher_expert_sha256'}
                assert set(a)-set(b) == new_extra and set(b)-set(a) == old_extra
                assert {k:v for k,v in a.items() if k not in new_extra} == {k:v for k,v in b.items() if k not in old_extra}
            else:
                assert a == b
        for duration, data in [(4800, run), (1600, old)]:
            key = f'{arm}_{duration}'
            runs[key] = data
            values[key] = np.array([data['average_validation'][t]['BAcc'] for t in tasks])
        sources.update({str(p): sha256(p) for p in [path, plan_path, old_path, history_path]})
    assert runs['merged_4800']['dataset_audits'] == runs['base_4800']['dataset_audits']
    comparisons = {f'{arm}_4800_vs_1600': contrast(values[f'{arm}_4800'], values[f'{arm}_1600'], tasks) for arm in design['arms']}
    comparisons.update({f'merged_vs_base_{duration}': contrast(values[f'merged_{duration}'], values[f'base_{duration}'], tasks) for duration in [1600, 4800]})
    comparisons['initialization_by_duration_interaction'] = contrast(
        values['merged_4800']-values['merged_1600'], values['base_4800']-values['base_1600'], tasks)
    joint_path = root/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    sources[str(joint_path)] = sha256(joint_path)
    joint_mean = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    merger_mean = runs['merged_4800']['initial_validation']['macro_bacc']
    means = {k: float(v.mean()) for k,v in values.items()}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.1, 1.2]})
    for arm, color in [('merged', '#0072b2'), ('base', '#d55e00')]:
        h = histories[arm]
        ax.plot([r['step'] for r in h], [100*r['macro_bacc'] for r in h], color=color, label=arm+' trajectory')
        ax.scatter([1600, 4800], [100*means[f'{arm}_{d}'] for d in [1600, 4800]], color=color, marker='D', s=55, zorder=3,
                   label=arm+' fixed half-window means')
    ax.axvline(1600, color='#999999', ls=':', lw=.8)
    ax.axhline(joint_mean*100, color='#555555', ls='--', label='Historical Joint reference')
    ax.set_xlabel('Additional CE updates')
    ax.set_ylabel('Full16 validation macro BAcc (%)')
    ax.set_title('A. Identical prefix, repeated training stream', loc='left')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    y = np.arange(16)
    for offset, arm, color in [(-.18, 'merged', '#0072b2'), (.18, 'base', '#d55e00')]:
        row = comparisons[f'{arm}_4800_vs_1600']
        bx.barh(y+offset, [row['per_task_gain_pp'][t] for t in tasks], height=.34, color=color,
                label=f'{arm}: mean {row["macro_gain_pp"]:+.3f} pp')
    bx.axvline(0, color='#777777', lw=.8)
    bx.set_yticks(y, tasks, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel('4800-step mean minus1600-step mean BAcc (pp)')
    bx.set_title('B. Every task retained', loc='left')
    bx.legend(fontsize=8)
    fig.suptitle('Fixed CE budget control: same117785 fitting rows, three times the exposures')
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .015, 'Single exposed bank; expert and coefficient construction costs are additional for the merged initialization.', ha='center', fontsize=9)
    figures = {}
    for suffix in ['png', 'pdf']:
        p = root/f'analysis/figures/duration4800_ce_controls.{suffix}'
        assert not p.exists()
        fig.savefig(p, dpi=170, bbox_inches='tight')
        figures[str(p)] = sha256(p)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'sources': sources,
              'datasets': tasks, 'macro_bacc': means, 'contrasts': comparisons,
              'joint_macro_bacc': joint_mean, 'initial_merger_macro_bacc': merger_mean,
              'fraction_joint_gap_closed_from_merger': {str(d): (means[f'merged_{d}']-merger_mean)/(joint_mean-merger_mean) for d in [1600, 4800]},
              'calibration_macro_bacc': {k: float(np.mean([r['average_calibration'][t]['BAcc'] for t in tasks])) for k,r in runs.items()},
              'process_seconds': {k:r['seconds'] for k,r in runs.items()},
              'budget': design['budget'], 'all_prefix_states_and_metrics_exact': True,
              'figures': figures,
              'scope': design['scope']+' Task-deletion ranges are descriptive sensitivity, not confidence intervals. Fixed means remain primary even if another recorded checkpoint scores higher.'}
    target = root/'analysis/duration4800_ce_summary.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'report': str(target), 'sha256': sha256(target), 'macro_bacc': means,
                      'contrasts': {k:{n:v for n,v in row.items() if n not in ['per_task_gain_pp', 'leave_one_task_out_pp']} for k,row in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
