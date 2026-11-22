"""Report fixed full-model CE plus distillation objectives against matched CE."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    design_path = ROOT/'inputs/full_model_distillation_design.json'
    design = json.loads(design_path.read_text())
    baseline_path = ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'
    baseline = json.loads(baseline_path.read_text())
    tasks = baseline['datasets']
    text_reference_path = ROOT/'seed0/local_gradient_cache/run.json'
    text_reference = json.loads(text_reference_path.read_text())
    source_fit_path = ROOT/'seed0/original_fp32_tensor_balanced_ce/run.json'
    source_fit = json.loads(source_fit_path.read_text())
    runs, sources = {'balanced_ce': baseline}, {'balanced_ce': {'path': str(baseline_path), 'sha256': sha(baseline_path)}}
    target_path = ROOT/'seed0/full_model_distillation_targets/run.json'
    target = json.loads(target_path.read_text())
    scale_path = ROOT/'analysis/full_model_distillation_scales.json'
    scales = json.loads(scale_path.read_text())
    assert scales['target_run_sha256'] == sha(target_path)
    for condition in design['conditions']:
        if condition == 'balanced_ce':
            continue
        path = ROOT/f'seed0/full_distillation_{condition}/run.json'
        run = json.loads(path.read_text())
        assert run['condition'] == condition and run['datasets'] == tasks
        assert not run['engineering_smoke'] and run['fixed_tensors_exact']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['average_steps'] == baseline['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['training_pool'] == baseline['training_pool'] == 'full_training'
        assert run['schedule_sha256'] == baseline['schedule_sha256']
        assert run['trainable_parameters'] == baseline['trainable_parameters']
        assert run['target_cache_sha256'] == sha(target_path)
        assert run['scales_sha256'] == sha(scale_path)
        assert run['plan']['design_sha256'] == sha(design_path)
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'lr', 'seed', 'batch_size']:
            assert run['plan'][key] == baseline['plan'][key], key
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == baseline['initial_validation']['validation'][task]['BAcc']
            assert run['distillation_audits'][task]['target_sha256'] == target['records'][task]['target_sha256']
            assert run['distillation_audits'][task]['auxiliary_scale'] == scales['per_task'][task][condition]['scale']
            assert abs(scales['per_task'][task][condition]['scaled_aux_raw_gradient_rms']/scales['per_task'][task][condition]['ce_raw_gradient_rms']-1) < 1e-12
            a, b = run['dataset_audits'][task], baseline['dataset_audits'][task]
            assert set(a)-set(b) == {'labels', 'instruction_hash', 'prototypes_hash'}
            assert set(b)-set(a) == {'teacher_hash', 'teacher_expert_sha256'}
            assert all(a[k] == b[k] for k in set(a)&set(b)), task
            assert a['labels'] == source_fit['labels'][task]
            for key in ['instruction_hash', 'prototypes_hash']:
                assert a[key] == text_reference['records'][task][key]
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        assert run['checkpoint_sha256'] == sha(path.with_name('merged.ckpt'))
        assert not run['test_data_read'] and not run['test_inference_performed']
        runs[condition] = run
        sources[condition] = {'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': run['checkpoint_sha256']}
    joint_path = ROOT/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    values = {c: np.array([r['average_validation'][t]['BAcc'] for t in tasks]) for c, r in runs.items()}
    values['initial_merger'] = np.array([baseline['initial_validation']['validation'][t]['BAcc'] for t in tasks])
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    sources['Joint'] = {'path': str(joint_path), 'sha256': sha(joint_path)}
    contrasts = {}
    pairs = [('ce_prototype_contrast', c) for c in ['balanced_ce', 'ce_teacher_kl', 'ce_full_feature', 'initial_merger']]
    pairs += [(c, control) for c in ['ce_teacher_kl', 'ce_full_feature'] for control in ['balanced_ce', 'initial_merger']]
    for treatment, control in pairs:
        gains = 100*(values[treatment]-values[control])
        contrasts[treatment+'_vs_'+control] = {
            'macro_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
            'improved': int((gains > 0).sum()), 'declined': int((gains < 0).sum()),
            'leave_one_task_out_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)}}
    initial, joint_target = values['initial_merger'].mean(), values['Joint'].mean()
    result = {
        'unix': time.time(), 'code_sha256': sha(__file__), 'design_source': {'path': str(design_path), 'sha256': sha(design_path)},
        'sources': sources, 'audit_schema_sources': {str(p): sha(p) for p in [text_reference_path, source_fit_path]},
        'audit_schema_note': 'All shared dataset fields exactly match. Legacy-only unused teacher fields differ by design; new-only text hashes and labels independently replay the frozen gradient/source-fit metadata.', 'datasets': tasks, 'macro_bacc': {k: float(v.mean()) for k, v in values.items()},
        'contrasts': contrasts,
        'fraction_initial_joint_gap_closed': {k: float((values[k].mean()-initial)/(joint_target-initial)) for k in runs},
        'calibration_macro_bacc': {c: float(np.mean([r['average_calibration'][t]['BAcc'] for t in tasks])) for c, r in runs.items()},
        'process_seconds': {c: r['seconds'] for c, r in runs.items()},
        'construction_cost': {'target_cache_seconds': target['seconds'], 'scale_calibration_seconds': scales['seconds'],
                              'student_forward_examples': target['student_forward_examples'], 'teacher_forward_examples': target['teacher_forward_examples']},
        'additional_sources': {str(p): sha(p) for p in [target_path, scale_path]},
        'scope': 'Fixed single-bank full-model objective pilot on identical204800 fitting entries and full-pool class weights. '
                 'Three auxiliaries use the same own-expert targets and labels; their initial weighted per-example raw-output gradient RMS matches CE per task. '
                 'This does not match parameter gradients or later dynamics. Same1600 updates, Adam1e-5 and fixed six-state mean; native cosine inference, no runtime extras. '
                 'Teacher construction cost is additional. Prior normalized MSE, feature/KL distillation and gradient normalization preclude broad novelty claims. '
                 'Exposed validation, no selected-checkpoint substitution, no scale/LR sweep or independent-confirmation claim.'}
    path = ROOT/'analysis/full_model_distillation_summary.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
