"""Retain every fixed full-pool optimizer/parameterization control at the same budget."""
import json
import time

import numpy as np

from report_coverage_control import ROOT,sha


def main():
    names = {'full_weight_lr1e5':'full_control_coverage_full_balanced_ce',
             'full_weight_lr1e4':'full_control_coverage_lr1e4_balanced_ce',
             'tensor_coefficients_lr1e2':'coefficient_control_full_pool_balanced_ce'}
    paths = {name:ROOT/'seed0'/directory/'run.json' for name,directory in names.items()}
    runs = {name:json.loads(path.read_text()) for name,path in paths.items()}
    reference = runs['full_weight_lr1e5']
    tasks = reference['datasets']
    initial_metrics = reference['initial_validation']['validation']
    initial = float(np.mean([initial_metrics[t]['BAcc'] for t in tasks]))
    joint_path = ROOT/'analysis/full_model_controls_lr1e5_summary.json'
    joint = json.loads(joint_path.read_text())['joint_validation_bacc']
    metrics = {'initial':initial_metrics}
    costs,sources = {},{}
    for name,run in runs.items():
        assert not run['engineering_smoke'] and run['datasets'] == tasks
        assert run['training_pool'] == 'full_training' and run['condition'] == 'balanced_ce'
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['fixed_tensors_exact'] and not run['test_data_read'] and not run['test_inference_performed']
        for key in ['initial_checkpoint_sha256','calibration_sha256','schedule_sha256','coverage_audit_sha256',
                    'teacher_cache_sha256','source_fit','steps','batch_size','eval_every','average_steps','seed']:
            assert run['plan'][key] == reference['plan'][key],(name,key)
        assert run['dataset_audits'] == reference['dataset_audits']
        assert all(run['initial_validation']['validation'][t]['BAcc'] == initial_metrics[t]['BAcc'] for t in tasks)
        assert all(run['export_checks'][t]['max_abs_logit_error'] <= 1e-5 for t in tasks)
        assert sha(paths[name].with_name('merged.ckpt')) == run['checkpoint_sha256']
        if name.startswith('tensor_'):
            assert run['plan']['lr'] == .01 and run['trainable_parameters'] == 6128
            assert run['method'] == 'residual_tensor_coefficients' and run['initial_parameter_and_probe_replay_exact']
        else:
            assert run['trainable_parameters'] == 25751552
            assert run['plan']['lr'] == (1e-5 if name.endswith('1e5') else 1e-4)
        metrics[name] = run['average_validation']
        costs[name] = {'seconds':run['seconds'],'trainable_scalars':run['trainable_parameters'],
                       'additional_training_exposures':run['training_examples'],
                       'selected_step_secondary':run['selected_step']}
        sources[name] = {'path':str(paths[name]),'sha256':sha(paths[name]),'checkpoint_sha256':run['checkpoint_sha256']}
    summaries = {}
    for name,value in metrics.items():
        macro = float(np.mean([value[t]['BAcc'] for t in tasks]))
        summaries[name] = {'macro_bacc':macro,'gain_over_initial_pp':100*(macro-initial),
                           'joint_gap_pp':100*(joint-macro),'fraction_initial_joint_gap_closed':(macro-initial)/(joint-initial)}
    comparisons = [(name,'initial') for name in runs]
    comparisons += [('tensor_coefficients_lr1e2',name) for name in ['full_weight_lr1e5','full_weight_lr1e4']]
    comparisons += [('full_weight_lr1e4','full_weight_lr1e5')]
    contrasts = {}
    for left,right in comparisons:
        changes = {t:100*(metrics[left][t]['BAcc']-metrics[right][t]['BAcc']) for t in tasks}
        contrasts[left+'_vs_'+right] = {'mean_change_pp':float(np.mean(list(changes.values()))),
                                       'per_task_change_pp':changes,'improved_tasks':sum(v>0 for v in changes.values()),
                                       'declined_tasks':sum(v<0 for v in changes.values()),
                                       'leave_one_task_out_change_pp':{t:float(np.mean([v for k,v in changes.items() if k!=t])) for t in tasks}}
    result = {'unix':time.time(),'summaries':summaries,'contrasts':contrasts,'sources':sources,'costs':costs,
              'joint_macro_bacc':joint,'joint_source':str(joint_path),'joint_source_sha256':sha(joint_path),
              'calibration_reference_macro_bacc':{name:float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for name,run in runs.items()},
              'secondary':{name:{key:float(np.mean([run[key+'_validation'][t]['BAcc'] for t in tasks])) for key in ['selected','final']} for name,run in runs.items()},
              'per_task':{t:{name:value[t]['BAcc'] for name,value in metrics.items()} for t in tasks},
              'code_sha256':sha(__file__),
              'primary':'Every fixed mean of800/960/1120/1280/1440/1600. Initial model, full training data, class weights and actual204800 sampled examples are identical across controls.',
              'scope':'Sequential exploratory seed0 controls. Coefficient LR.01 reuses its original family recipe; unrestricted weights use1e-5 and1e-4. Different parameterizations have different numeric LR scales and compute. No negative result is an optimization capacity bound. These cells do not form a full data-by-parameterization factorial because an additional-budget original512 coefficient cell is absent. The existing1e-5 sampling replicas do not replicate1e-4 or coefficient optimization. No new-method or independent-validation claim.'}
    output = ROOT/'analysis/full_pool_parameterization_summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'calibration_reference_macro_bacc':result['calibration_reference_macro_bacc'],
                      'coefficient_vs_full1e5':contrasts['tensor_coefficients_lr1e2_vs_full_weight_lr1e5']},indent=2))


if __name__ == '__main__':
    main()
