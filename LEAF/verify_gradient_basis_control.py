"""Verify actual-gradient and whole-pipeline engineering evidence for the pilot."""
import json
import time

import torch

from report_coverage_control import ROOT, sha


def main():
    cpu_path = ROOT/'seed0/gradient_basis_cpu_check/run.json'
    gpu_path = ROOT/'seed0/gradient_basis_gradient_smoke/run.json'
    reference_path = ROOT/'seed0/original512_coefficient_balanced_ce/run.json'
    basis_path = ROOT/'seed0/local_gradient_basis/run.json'
    cpu, gpu, reference, basis = [json.loads(p.read_text())
                                  for p in [cpu_path, gpu_path, reference_path, basis_path]]
    assert cpu['passed'] and cpu['engineering_only'] and cpu['basis_kind'] == 'gradient'
    assert cpu['initial_model_state_exact'] and cpu['fixed_state_exact']
    assert cpu['functional_call_restores_template']
    assert cpu['coefficient_shape'] == [32, 383]
    for row in cpu['gradient_checks'].values():
        assert row['chain_rule_max_abs_error'] <= 1e-6
        assert row['original_basis_gradient_norm'] > 0 and row['added_basis_gradient_norm'] > 0
    assert gpu['engineering_smoke'] and gpu['method'] == 'residual_tensor_coefficients_gradient'
    assert gpu['basis_kind'] == 'gradient' and gpu['updates'] == 32
    assert gpu['average_steps'] == [16, 32] and gpu['plan']['lr'] == .001
    assert gpu['trainable_parameters'] == gpu['effective_trainable_parameters'] == 12256
    assert gpu['coefficient_shape'] == [32, 383]
    assert gpu['initial_parameter_and_probe_replay_exact'] and gpu['fixed_tensors_exact']
    assert not gpu['test_data_read'] and not gpu['test_inference_performed']
    assert all(v > 0 for v in gpu['component_gradient_max'].values())
    assert gpu['datasets'] == reference['datasets']
    assert gpu['plan']['basis_run_sha256'] == cpu['basis_run_sha256'] == sha(basis_path)
    assert gpu['basis_artifact']['sha256'] == basis['outputs']['gradient']['sha256']
    for task in gpu['datasets']:
        assert gpu['initial_probe_checks'][task]['max_abs_logit_error'] == 0
        assert gpu['export_checks'][task]['max_abs_logit_error'] <= 1e-5
        for key in ['calibration_size', 'training_pool', 'training_pool_size', 'class_counts',
                    'class_weights', 'eeg_hash', 'training_pool_eeg_hash', 'teacher_hash', 'teacher_expert_sha256']:
            assert gpu['dataset_audits'][task][key] == reference['dataset_audits'][task][key], (task, key)
    for filename, expected in cpu['source_sha256'].items():
        assert sha(filename) == expected == gpu['plan']['source_sha256'][filename], filename
    assert gpu['mean_coefficient_vs_mean_state_max_abs'] <= 1e-5
    assert sha(gpu_path.with_name('merged.ckpt')) == gpu['checkpoint_sha256']
    beta = torch.load(gpu_path.with_name('mean_beta.pt'), map_location='cpu', weights_only=True)
    assert beta.shape == (32, 383) and torch.isfinite(beta).all()
    assert beta[:16].norm() > 0 and beta[16:].norm() > 0
    result = {'unix': time.time(), 'passed': True,
              'CPU_chain_rule_max_abs_error': max(x['chain_rule_max_abs_error'] for x in cpu['gradient_checks'].values()),
              'all16_GPU_initial_probes_exact': True, 'all16_GPU_data_weights_match_original512_control': True,
              'GPU_export_max_logit_error': max(x['max_abs_logit_error'] for x in gpu['export_checks'].values()),
              'GPU_mean_coefficient_vs_mean_state_max_abs': gpu['mean_coefficient_vs_mean_state_max_abs'],
              'GPU_original_beta_norm': float(beta[:16].norm()), 'GPU_added_beta_norm': float(beta[16:].norm()),
              'sources': [{'path': str(p), 'sha256': sha(p)} for p in [cpu_path, gpu_path, reference_path, basis_path]],
              'source_sha256': cpu['source_sha256'], 'code_sha256': sha(__file__),
              'scope': 'Engineering checks only: real EEG chain rule, actual Adam updates, all16 zero-update probes, fixed data/weights and ordinary exports. Scientific1600-step fixed averages and effectiveness remain to be measured.'}
    output = ROOT/'analysis/gradient_basis_engineering_verification.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
