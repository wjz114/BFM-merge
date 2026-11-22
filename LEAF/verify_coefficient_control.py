"""Gate the scientific coefficient control on independent and whole-pipeline checks."""
import hashlib
import json
import time
from pathlib import Path

import torch


ROOT = Path('runs/paper_merge_v2')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def main():
    cpu_path = ROOT/'seed0/coefficient_control_cpu_check_v2/run.json'
    gpu_path = ROOT/'seed0/coefficient_control_full_pool_smoke/run.json'
    reference_path = ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'
    cpu,gpu,reference = [json.loads(p.read_text()) for p in [cpu_path,gpu_path,reference_path]]
    assert cpu['passed'] and cpu['engineering_only']
    assert cpu['initial_model_state_exact'] and cpu['fixed_state_exact'] and cpu['functional_call_restores_template']
    assert all(r['exact_logits'] for r in cpu['initial_probe_checks'].values())
    assert all(r['chain_rule_max_abs_error'] <= 1e-6 for r in cpu['gradient_checks'].values())
    assert gpu['engineering_smoke'] and gpu['method'] == 'residual_tensor_coefficients'
    assert gpu['updates'] == 32 and gpu['average_steps'] == [16,32]
    assert gpu['trainable_parameters'] == 6128 and gpu['coefficient_shape'] == [16,383]
    assert gpu['updated_model_parameter_elements'] == reference['trainable_parameters']
    assert gpu['initial_parameter_and_probe_replay_exact'] and gpu['fixed_tensors_exact']
    assert not gpu['test_data_read'] and not gpu['test_inference_performed']
    assert all(value>0 for value in gpu['component_gradient_max'].values())
    assert gpu['datasets'] == reference['datasets']
    for task in gpu['datasets']:
        assert gpu['initial_probe_checks'][task]['max_abs_logit_error'] == 0
        assert gpu['export_checks'][task]['max_abs_logit_error'] <= 1e-5
        for key in ['calibration_size','training_pool','training_pool_size','class_counts','class_weights',
                    'eeg_hash','training_pool_eeg_hash','teacher_hash','teacher_expert_sha256']:
            assert gpu['dataset_audits'][task][key] == reference['dataset_audits'][task][key], (task,key)
    for path,expected in cpu['source_sha256'].items():
        assert sha(path) == expected == gpu['plan']['source_sha256'][path], path
    assert gpu['mean_coefficient_vs_mean_state_max_abs'] <= 1e-5
    assert sha(gpu_path.with_name('merged.ckpt')) == gpu['checkpoint_sha256']
    beta = torch.load(gpu_path.with_name('mean_beta.pt'),map_location='cpu',weights_only=True)
    assert beta.shape == (16,383) and torch.isfinite(beta).all() and beta.norm()>0
    result = {'unix':time.time(),'passed':True,'checks':{
        'CPU_real_EEG_chain_rule_max_abs_error':max(r['chain_rule_max_abs_error'] for r in cpu['gradient_checks'].values()),
        'CPU_zero_update_and_native_export_exact':True,'GPU_all16_initial_probes_exact':True,
        'GPU_coefficient_gradients_both_components_nonzero':True,
        'GPU_training_pool_and_class_weights_match_full_weight_control':True,
        'GPU_export_logit_max_abs':max(r['max_abs_logit_error'] for r in gpu['export_checks'].values()),
        'GPU_mean_coefficient_vs_mean_state_max_abs':gpu['mean_coefficient_vs_mean_state_max_abs'],
        'GPU_mean_beta_norm':float(beta.norm()),'GPU_fixed_state_exact':True},
        'sources':[{'path':str(p),'sha256':sha(p)} for p in [cpu_path,gpu_path,reference_path]],
        'plan_sha256':gpu['plan_sha256'],'source_sha256':gpu['plan']['source_sha256'],
        'verification_code_sha256':sha(__file__),
        'scope':'Engineering verification only. The complete scientific coefficient run still must replay full16 initial validation and evaluate the fixed1600-update average.'}
    output = ROOT/'analysis/coefficient_control_engineering_verification.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':result['checks']},indent=2))


if __name__ == '__main__':
    main()
