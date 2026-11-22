"""Report the fixed angular normalization controls with matched earlier arms."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    previous_path = ROOT/'analysis/full_model_distillation_summary.json'
    previous = json.loads(previous_path.read_text())
    design_path = ROOT/'inputs/angular_distillation_design.json'
    scale_path = ROOT/'analysis/angular_distillation_scales.json'
    design, scales = [json.loads(p.read_text()) for p in [design_path, scale_path]]
    tasks = previous['datasets']
    sources = dict(previous['sources'])
    runs = {}
    for condition, source in sources.items():
        if condition == 'Joint':
            continue
        assert sha(source['path']) == source['sha256']
        runs[condition] = json.loads(open(source['path']).read())
    baseline, matched = runs['balanced_ce'], runs['ce_full_feature']
    for condition in design['conditions']:
        if condition == 'balanced_ce':
            continue
        path = ROOT/f'seed0/full_distillation_{condition}/run.json'
        run = json.loads(path.read_text())
        assert run['condition'] == condition and run['datasets'] == tasks
        assert not run['engineering_smoke'] and run['fixed_tensors_exact']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['average_steps'] == baseline['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['training_pool'] == 'full_training'
        for key in ['schedule_sha256', 'trainable_parameters', 'dataset_audits', 'target_cache_sha256']:
            assert run[key] == matched[key], key
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'lr', 'seed', 'batch_size']:
            assert run['plan'][key] == baseline['plan'][key], key
        assert run['plan']['design_sha256'] == sha(design_path) == scales['design_sha256']
        assert run['scales_sha256'] == sha(scale_path)
        assert run['target_cache_sha256'] == scales['target_run_sha256']
        assert run['code_sha256'] == scales['trainer_sha256']
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == baseline['initial_validation']['validation'][task]['BAcc']
            assert run['distillation_audits'][task]['target_sha256'] == matched['distillation_audits'][task]['target_sha256']
            row = scales['per_task'][task][condition]
            assert run['distillation_audits'][task]['auxiliary_scale'] == row['scale']
            assert abs(row['scaled_aux_raw_gradient_rms']/row['ce_raw_gradient_rms']-1) < 1e-12
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        assert sha(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
        assert not run['test_data_read'] and not run['test_inference_performed']
        runs[condition] = run
        sources[condition] = {'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': run['checkpoint_sha256']}
    values = {c: np.array([run['average_validation'][t]['BAcc'] for t in tasks]) for c, run in runs.items()}
    values['initial_merger'] = np.array([baseline['initial_validation']['validation'][t]['BAcc'] for t in tasks])
    joint_source = sources['Joint']
    assert sha(joint_source['path']) == joint_source['sha256']
    joint = json.loads(open(joint_source['path']).read())
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    for condition, value in previous['macro_bacc'].items():
        assert abs(values[condition].mean()-value) < 1e-12
    pairs = [('ce_angular_prototype_contrast', c) for c in ['balanced_ce', 'ce_prototype_contrast', 'ce_angular_full_feature', 'ce_full_feature', 'ce_teacher_kl', 'initial_merger']]
    pairs += [('ce_angular_full_feature', c) for c in ['balanced_ce', 'ce_full_feature', 'ce_teacher_kl', 'initial_merger']]
    contrasts = {}
    for treatment, control in pairs:
        gains = 100*(values[treatment]-values[control])
        contrasts[treatment+'_vs_'+control] = {
            'macro_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
            'improved': int((gains > 0).sum()), 'declined': int((gains < 0).sum()),
            'tied': int((gains == 0).sum()),
            'leave_one_task_out_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)}}
    initial = values['initial_merger'].mean()
    result = {'unix': time.time(), 'code_sha256': sha(__file__), 'datasets': tasks,
              'sources': sources, 'additional_sources': {str(p): sha(p) for p in [previous_path, design_path, scale_path]},
              'macro_bacc': {c: float(v.mean()) for c, v in values.items()},
              'contrasts': contrasts,
              'fraction_initial_joint_gap_closed': {c: float((v.mean()-initial)/(values['Joint'].mean()-initial)) for c, v in values.items() if c not in ['initial_merger', 'Joint']},
              'calibration_macro_bacc': {c: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for c, run in runs.items()},
              'process_seconds': {c: run['seconds'] for c, run in runs.items()},
              'additional_angular_scale_calibration_seconds': scales['seconds'],
              'reused_teacher_construction_cost': previous['construction_cost'],
              'scope': 'Fixed two-arm current-student-norm control after the fixed-norm pilot. Same fitting rows, weights, order, optimizer, initial checkpoint and prescribed six-state mean. Same initial raw-output RMS matching rule changes alpha; parameter gradients and later dynamics are not matched. No new teacher forwards. Unit-feature distillation and norm/direction separation are prior art; projected angular loss is a centered-logit quadratic, not extra teacher information. Single-bank exposed validation, no hyperparameter sweep or independent confirmation.'}
    path = ROOT/'analysis/angular_distillation_summary.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
