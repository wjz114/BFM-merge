"""Compare full-pool and original512 balanced CE at the fixed additional budget."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path('runs/paper_merge_v2')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    paths = {'original512': ROOT/'seed0/full_control_lr1e5_balanced_ce/run.json',
             'full_training': ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    ref, full = runs['original512'], runs['full_training']
    tasks = ref['datasets']
    initial = ref['initial_validation']['validation']
    joint_path = ROOT/'analysis/full_model_controls_lr1e5_summary.json'
    joint_report = json.loads(joint_path.read_text())
    joint_macro = float(joint_report['joint_validation_bacc'])
    initial_macro = float(np.mean([initial[t]['BAcc'] for t in tasks]))
    coverage_path = Path(full['plan']['coverage_audit'])
    assert sha(coverage_path) == full['plan']['coverage_audit_sha256']
    coverage = json.loads(coverage_path.read_text())
    assert full['training_pool'] == 'full_training'
    assert not full['test_data_read'] and not full['test_inference_performed']
    for key in ['initial_checkpoint_sha256','calibration_sha256','source_fit','teacher_cache_sha256',
                'steps','batch_size','eval_every','average_steps','seed','lr']:
        assert full['plan'][key] == ref['plan'][key], key
    schedules = {key: np.load(run['plan']['schedule']) for key, run in runs.items()}
    assert np.array_equal(schedules['original512']['task'], schedules['full_training']['task'])
    assert coverage['original512_schedule_exact_replay']
    all_metrics = {'initial': initial}
    checks, costs, data_access = {}, {}, {}
    for name, run in runs.items():
        assert not run['engineering_smoke'] and run['condition'] == 'balanced_ce'
        assert run['datasets'] == tasks and run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['trainable_parameters'] == ref['trainable_parameters'] and run['fixed_tensors_exact']
        assert sha(paths[name].with_name('merged.ckpt')) == run['checkpoint_sha256']
        assert all(run['initial_validation']['validation'][t]['BAcc'] == initial[t]['BAcc'] for t in tasks)
        assert all(run['export_checks'][t]['max_abs_logit_error'] <= 1e-5 for t in tasks)
        for task in tasks:
            for key in ['calibration_size','validation_size','eeg_hash','teacher_hash','teacher_expert_sha256']:
                assert run['dataset_audits'][task][key] == ref['dataset_audits'][task][key], (name,task,key)
        if name == 'full_training':
            for task in tasks:
                audit = coverage['audits'][task]
                actual = run['dataset_audits'][task]
                assert actual['training_pool_size'] == audit['train_size']
                assert actual['class_counts'] == audit['train_class_counts']
                assert actual['training_pool_eeg_hash'] == audit['train_eeg_hash']
        for suffix in ['average','selected','final']:
            all_metrics[name+'_'+suffix] = run[suffix+'_validation']
        costs[name] = {'additional_updates':run['updates'],'additional_training_examples':run['training_examples'],
                       'seconds':run['seconds'],'selected_step_secondary':run['selected_step']}
        checks[name] = {'run':str(paths[name]),'run_sha256':sha(paths[name]),
                        'checkpoint_sha256':run['checkpoint_sha256'],
                        'all16_initial_bacc_exact':True,'fixed_tensors_exact':True,
                        'maximum_export_logit_error':max(run['export_checks'][t]['max_abs_logit_error'] for t in tasks)}
        data_access[name] = {'available_training_rows':sum(run['dataset_audits'][t].get('training_pool_size',512) for t in tasks),
                             'unique_training_rows_sampled':sum(len(np.unique(schedules[name]['local_indices'][schedules[name]['task']==i]))
                                                                for i,t in enumerate(tasks)),
                             'reference_calibration_rows':sum(run['dataset_audits'][t]['calibration_size'] for t in tasks)}
    summaries = {}
    for name, metrics in all_metrics.items():
        value = float(np.mean([metrics[t]['BAcc'] for t in tasks]))
        summaries[name] = {'macro_bacc':value,'gain_over_initial_pp':100*(value-initial_macro),
                           'joint_gap_pp':100*(joint_macro-value),
                           'fraction_initial_joint_gap_closed':(value-initial_macro)/(joint_macro-initial_macro)}
    contrasts = {}
    for left, right in [('full_training_average','initial'),('full_training_average','original512_average')]:
        gains = {t:100*(all_metrics[left][t]['BAcc']-all_metrics[right][t]['BAcc']) for t in tasks}
        contrasts[left+'_vs_'+right] = {'macro_gain_pp':float(np.mean(list(gains.values()))),
                                      'per_task_gain_pp':gains,
                                      'improved_tasks':sum(g>0 for g in gains.values()),
                                      'declined_tasks':sum(g<0 for g in gains.values()),
                                      'leave_one_task_out_gain_pp':{t:float(np.mean([g for k,g in gains.items() if k!=t])) for t in tasks}}
    result = {'unix':time.time(),'summaries':summaries,'contrasts':contrasts,
              'joint_macro_bacc':joint_macro,'joint_source':str(joint_path),'joint_source_sha256':sha(joint_path),
              'per_task':{t:{name:metrics[t]['BAcc'] for name,metrics in all_metrics.items()} for t in tasks},
              'calibration_reference_macro_bacc':{name:float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks]))
                                                  for name,run in runs.items()},
              'costs':costs,'data_access':data_access,'verification':checks,
              'data_audit':str(coverage_path),'data_audit_sha256':sha(coverage_path),
              'code_sha256':sha(__file__),
              'scope':'One fixed seed0 comparison on exposed validation. Same initialization, task order, optimizer, learning rate,204800 additional sample exposures and fixed six-state mean. Full pool changes available examples, within-task sample order and empirical inverse-frequency class weights. Both objectives target equal class risk; this is increased training-data access, not an equal-data algorithmic gain. No independent confirmation or statistical significance claim.',
              'primary':'Fixed average of steps800/960/1120/1280/1440/1600. Selected and final models are secondary. Calibration evaluation always uses the same original512/task reference, not the enlarged training pool.'}
    output = ROOT/'analysis/full_control_coverage_summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'data_access':data_access,
                      'full_vs_original':contrasts['full_training_average_vs_original512_average'],
                      'calibration_reference_macro_bacc':result['calibration_reference_macro_bacc']},indent=2))


if __name__ == '__main__':
    main()
