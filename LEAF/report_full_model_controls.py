"""Compare fixed full-model controls with their common merged initialization."""
import argparse
import json
from pathlib import Path
from statistics import mean

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=root/'inputs/full_model_controls_plan.json')
    parser.add_argument('--prefix', default='full_control_')
    parser.add_argument('--output', type=Path, default=root/'analysis/full_model_controls_summary.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    original = json.loads(Path(plan['initial_checkpoint']).with_name('run.json').read_text())
    joint = json.loads((root/'seed0/joint_span_diagnostic/run.json').read_text())['evaluations']['joint']['macro_bacc']
    tasks = original['datasets']
    cells = {'initial': original['folded_val']}
    runs, costs = {}, {}
    for condition in plan['conditions']:
        folder = root/f'seed0/{args.prefix}{condition}'
        run = json.loads((folder/'run.json').read_text())
        assert run['plan'] == plan and not run['engineering_smoke']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['fixed_tensors_exact'] and not run['test_inference_performed']
        assert len(run['export_checks']) == 16
        assert max(row['max_abs_logit_error'] for row in run['export_checks'].values()) <= 1e-5
        assert sha256(folder/'merged.ckpt') == run['checkpoint_sha256']
        assert run['trainable_parameters'] == 25751552
        assert all(abs(run['initial_validation']['validation'][task]['BAcc']-cells['initial'][task]['BAcc']) < 1e-12 for task in tasks)
        for variant in ['average', 'selected', 'final']:
            cells[condition+'_'+variant] = run[variant+'_validation']
        runs[condition] = run
        costs[condition] = {'seconds': run['seconds'], 'training_examples': run['training_examples'],
                            'selected_step_secondary': run['selected_step'], 'source_run_sha256': sha256(folder/'run.json')}
    first_run = next(iter(runs.values()))
    assert all(r['dataset_audits'] == first_run['dataset_audits'] for r in runs.values())
    assert all(r['schedule_sha256'] == first_run['schedule_sha256'] for r in runs.values())
    baseline = mean(row['BAcc'] for row in cells['initial'].values())
    summaries = {name: {'macro_bacc': mean(values[t]['BAcc'] for t in tasks),
                        'joint_gap_pp': 100*(joint-mean(values[t]['BAcc'] for t in tasks)),
                        'fraction_initial_joint_gap_closed': (mean(values[t]['BAcc'] for t in tasks)-baseline)/(joint-baseline)}
                 for name, values in cells.items()}
    contrasts = {}
    comparisons = [(name, 'initial') for name in cells if name!='initial']
    if {'balanced_ce', 'own_teacher_kl'} <= set(runs):
        comparisons.append(('balanced_ce_average', 'own_teacher_kl_average'))
    for treatment, control in comparisons:
        deltas = {task: 100*(cells[treatment][task]['BAcc']-cells[control][task]['BAcc']) for task in tasks}
        contrasts[treatment+'_vs_'+control] = {
            'macro_change_pp': mean(deltas.values()), 'per_task_change_pp': deltas,
            'improved': sum(v>0 for v in deltas.values()), 'declined': sum(v<0 for v in deltas.values()),
            'leave_one_task_out_pp': {t: mean(v for other, v in deltas.items() if t!=other) for t in tasks},
            'signed_net_contribution': {t: v/sum(deltas.values()) if abs(sum(deltas.values()))>1e-10 else None for t,v in deltas.items()}}
    result = {'summaries': summaries, 'contrasts': contrasts, 'per_task': cells, 'costs': costs,
              'average_calibration': {c: r['average_calibration'] for c,r in runs.items()},
              'joint_validation_bacc': joint, 'plan_sha256': sha256(args.plan), 'learning_rate': plan['lr'],
              'optimizer_sampling_seed': plan['seed'],
              'code_sha256': sha256(Path(__file__)),
              'scope': 'Original seed0 merger and calibration, full exposed source validation. Optimizer sampling seed is recorded separately. Same initialization, teacher cache, labels and replayed batches across conditions in this plan. Fixed average is primary; final/selected secondary. Ordinary full-weight optimization, not a novel merging contribution or independent confirmation.',
              'budget_scope': 'Each adds1600 full-model updates/204800examples after the common original1600-step coefficient merger. Equal to the merger-fitting stage count, not total RI bank adaptation cost. Original calibration IDs are exact; each plan specifies its task/example schedule. Historical individual coefficient-fitting batches were not logged or replayed.'}
    calibration_source = root/'seed0/cho_replacement_diagnostic/run.json'
    previous = json.loads(calibration_source.read_text())
    assert previous['checkpoint_hashes']['original'] == plan['initial_checkpoint_sha256']
    initial_calibration = {t: previous['evaluations']['original'][t]['calibration'] for t in tasks}
    validation_replay_max_errors = {
        m: max(abs(previous['evaluations']['original'][t]['validation'][m]-cells['initial'][t][m]) for t in tasks)
        for m in next(iter(cells['initial'].values()))}
    assert validation_replay_max_errors['BAcc'] < 1e-12
    assert max(validation_replay_max_errors.values()) < 1e-6
    result['initial_calibration'] = initial_calibration
    result['calibration_source_sha256'] = sha256(calibration_source)
    result['calibration_source_validation_replay_max_errors'] = validation_replay_max_errors
    result['calibration_fit_diagnostic'] = {
        c: {'initial_macro_bacc': mean(v['BAcc'] for v in initial_calibration.values()),
            'average_macro_bacc': mean(v['BAcc'] for v in r['average_calibration'].values()),
            'macro_change_pp': 100*mean(r['average_calibration'][t]['BAcc']-initial_calibration[t]['BAcc'] for t in tasks),
            'per_task_change_pp': {t: 100*(r['average_calibration'][t]['BAcc']-initial_calibration[t]['BAcc']) for t in tasks}}
        for c, r in runs.items()}
    result['calibration_fit_interpretation'] = 'Calibration accuracy improves while full validation does not. This documents a fitting/generalization gap under these recipes; it is not a causal identification of why generalization fails, or proof that a different learning rate cannot improve validation.'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summaries': summaries, 'primary': {k:v['macro_change_pp'] for k,v in contrasts.items() if '_average_vs_' in k}}, indent=2))


if __name__ == '__main__':
    main()
