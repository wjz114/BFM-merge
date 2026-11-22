"""Report the fixed same-task versus mixed-task micro4 CE pair."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    design_path = ROOT/'inputs/task_mixing_design.json'
    gate_path = ROOT/'analysis/task_mixing_ce_replay_verification.json'
    design = json.loads(design_path.read_text())
    native_path = ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'
    native = json.loads(native_path.read_text())
    tasks = native['datasets']
    runs = {'native128': native}
    sources = {'native128': {'path': str(native_path), 'sha256': sha(native_path)}}
    for mode in design['arms']:
        path = ROOT/f'seed0/task_mixing_{mode}/run.json'
        run = json.loads(path.read_text())
        assert run['condition'] == 'balanced_ce' and not run['engineering_smoke']
        assert run['batch_organization'] == mode and run['datasets'] == tasks
        assert run['microbatches_per_update'] == 4 and run['microbatch_size'] == 32
        assert run['training_forward_backward_calls'] == 6400
        assert run['updates'] == native['updates'] == 1600
        assert run['training_examples'] == native['training_examples'] == 204800
        assert run['average_steps'] == native['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['schedule_sha256'] == native['schedule_sha256']
        assert run['trainable_parameters'] == native['trainable_parameters']
        assert run['training_pool'] == 'full_training' and run['fixed_tensors_exact']
        assert run['plan']['design_sha256'] == sha(design_path)
        assert run['plan']['engineering_gate_sha256'] == sha(gate_path)
        assert run['micro_schedule_sha256'] == design['schedules'][mode]['micro_schedule_sha256']
        assert run['code_sha256'] == design['plan_template']['source_sha256']['train_task_mixing_ce.py']
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'lr', 'seed', 'batch_size']:
            assert run['plan'][key] == native['plan'][key], key
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == native['initial_validation']['validation'][task]['BAcc']
            a, b = run['dataset_audits'][task], native['dataset_audits'][task]
            assert set(a)-set(b) == {'labels', 'instruction_hash', 'prototypes_hash'}
            assert set(b)-set(a) == {'teacher_hash', 'teacher_expert_sha256'}
            assert all(a[k] == b[k] for k in set(a)&set(b)), task
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        assert sha(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
        assert not run['test_data_read'] and not run['test_inference_performed']
        runs[mode] = run
        sources[mode] = {'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': run['checkpoint_sha256']}
    assert runs['same_task_micro4']['dataset_audits'] == runs['mixed_task_micro4']['dataset_audits']
    assert runs['same_task_micro4']['initial_validation'] == runs['mixed_task_micro4']['initial_validation']
    values = {c: np.array([run['average_validation'][t]['BAcc'] for t in tasks]) for c, run in runs.items()}
    values['initial_merger'] = np.array([native['initial_validation']['validation'][t]['BAcc'] for t in tasks])
    joint_path = ROOT/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    pairs = [('mixed_task_micro4', c) for c in ['same_task_micro4', 'native128', 'initial_merger']]
    pairs += [('same_task_micro4', c) for c in ['native128', 'initial_merger']]
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
              'sources': sources, 'additional_sources': {str(p): sha(p) for p in [design_path, gate_path, joint_path]},
              'macro_bacc': {c: float(v.mean()) for c, v in values.items()}, 'contrasts': contrasts,
              'fraction_initial_joint_gap_closed': {c: float((v.mean()-initial)/(values['Joint'].mean()-initial)) for c, v in values.items() if c not in ['initial_merger', 'Joint']},
              'calibration_macro_bacc': {c: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for c, run in runs.items()},
              'process_seconds': {c: run['seconds'] for c, run in runs.items()},
              'schedule_audits': design['schedule_audits'],
              'scope': design['scope']+' All original task/example multiplicities preserved in each16-step block. Both micro4 arms use the same accumulation implementation. Native128 differs in floating-point grouping. Local cached task-mean variance is not a measurement of full-pool minibatch covariance, actual Adam steps or a unique generalization mechanism. Leave-one-task-out ranges are sensitivity analyses, not confidence intervals.'}
    with (ROOT/'analysis/task_mixing_summary.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
