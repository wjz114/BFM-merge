"""Report paired repair-pool results across all three expert/merger seeds."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    design_path = ROOT/'inputs/bank_coverage_replication_design.json'
    design = json.loads(design_path.read_text())
    joint_path = ROOT/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    assert joint['optimizer_updates'] == 0 and not joint['test_data_read']
    assert not joint['test_inference_performed']
    assert all(row['seed0_Joint_BAcc_replay_exact'] and row['validation_hashes_exact']
               for row in joint['data_checks'].values())
    tasks = joint['datasets']
    replicas, all_gains, all_contrasts = [], [], []
    for seed in [0, 1, 2]:
        initial_path = ROOT/f'seed{seed}/original_fp32_tensor_balanced_avg/run.json'
        initial = json.loads(initial_path.read_text())
        assert initial['seed'] == seed and initial['datasets'] == tasks
        assert initial['averaging_steps'] == design['average_steps']
        assert initial['coefficient_mean_exact'] and initial['fixed_base_buffers_and_inactive_exact']
        initial_values = np.array([initial['folded_val'][t]['BAcc'] for t in tasks])
        initial_macro = float(initial_values.mean())
        names = {'full': 'full_control_coverage_full_balanced_ce',
                 'original512': ('full_control_lr1e5_balanced_ce' if seed == 0
                                 else 'full_control_coverage_original512_balanced_ce')}
        paths = {name: ROOT/f'seed{seed}'/directory/'run.json' for name, directory in names.items()}
        runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
        for name, run in runs.items():
            assert run['datasets'] == tasks and run['condition'] == 'balanced_ce'
            assert not run['engineering_smoke'] and run['fixed_tensors_exact']
            assert not run['test_inference_performed']
            assert run['updates'] == 1600 and run['training_examples'] == 204800
            assert run['plan']['lr'] == design['lr'] and run['plan']['seed'] == seed
            assert run['average_steps'] == design['average_steps']
            assert run['plan']['initial_checkpoint_sha256'] == sha(initial_path.with_name('merged.ckpt'))
            assert run['checkpoint_sha256'] == sha(paths[name].with_name('merged.ckpt'))
            assert all(run['initial_validation']['validation'][t]['BAcc'] == initial['folded_val'][t]['BAcc']
                       for t in tasks)
            assert max(row['max_abs_logit_error'] for row in run['export_checks'].values()) <= 1e-5
            if seed > 0:
                assert run['plan']['design_sha256'] == sha(design_path)
                pool = 'full_training' if name == 'full' else 'original512'
                assert run['training_pool'] == pool and not run['test_data_read']
                assert run['plan']['schedule_sha256'] == design['schedules'][str(seed)][pool]['sha256']
        full, old = runs['full'], runs['original512']
        assert full['training_pool'] == 'full_training' and not full['test_data_read']
        assert full['plan']['coverage_audit_sha256'] == design['pool_data_audit_sha256']
        assert all(full['dataset_audits'][t]['eeg_hash'] == old['dataset_audits'][t]['eeg_hash'] for t in tasks)
        schedules = {name: np.load(run['plan']['schedule']) for name, run in runs.items()}
        assert np.array_equal(schedules['full']['task'], schedules['original512']['task'])
        values = {name: np.array([run['average_validation'][t]['BAcc'] for t in tasks])
                  for name, run in runs.items()}
        gains = 100*(values['full']-initial_values)
        contrast = 100*(values['full']-values['original512'])
        all_gains.append(gains)
        all_contrasts.append(contrast)
        macros = {name: float(value.mean()) for name, value in values.items()}
        joint_macro = joint['macro_bacc'][str(seed)]
        replicas.append({
            'bank_seed': seed, 'initial_macro_bacc': initial_macro, 'joint_macro_bacc': joint_macro,
            'fixed_average_macro_bacc': macros,
            'full_gain_over_initial_pp': float(gains.mean()),
            'full_minus_original512_pp': float(contrast.mean()),
            'initial_joint_gap_pp': 100*(joint_macro-initial_macro),
            'remaining_joint_gap_pp': 100*(joint_macro-macros['full']),
            'fraction_initial_joint_gap_closed': (macros['full']-initial_macro)/(joint_macro-initial_macro),
            'full_improved_tasks': int((gains > 0).sum()), 'full_declined_tasks': int((gains < 0).sum()),
            'full_unique_training_rows_sampled': sum(
                len(np.unique(schedules['full']['local_indices'][schedules['full']['task'] == i]))
                for i in range(len(tasks))),
            'sources': {name: {'path': str(path), 'sha256': sha(path),
                               'checkpoint_sha256': runs[name]['checkpoint_sha256']}
                        for name, path in paths.items()},
            'initial_source': {'path': str(initial_path), 'sha256': sha(initial_path)},
            'seconds': {name: run['seconds'] for name, run in runs.items()},
            'test_data_read_field': {name: run.get('test_data_read') for name, run in runs.items()},
        })
    gains, contrasts = np.stack(all_gains), np.stack(all_contrasts)
    result = {
        'unix': time.time(), 'replicas': replicas,
        'mean_gain_over_initial_pp': float(gains.mean()),
        'sample_sd_gain_over_initial_pp': float(gains.mean(1).std(ddof=1)),
        'mean_gain_over_original512_pp': float(contrasts.mean()),
        'sample_sd_gain_over_original512_pp': float(contrasts.mean(1).std(ddof=1)),
        'mean_fraction_initial_joint_gap_closed': float(np.mean([
            row['fraction_initial_joint_gap_closed'] for row in replicas])),
        'ratio_of_mean_gain_to_mean_initial_joint_gap': float(gains.mean()/np.mean([
            row['initial_joint_gap_pp'] for row in replicas])),
        'per_task': {t: {'full_gains_over_initial_pp': gains[:, i].tolist(),
                          'full_minus_original512_pp': contrasts[:, i].tolist()}
                     for i, t in enumerate(tasks)},
        'all_three_improve_over_initial': [t for i, t in enumerate(tasks) if (gains[:, i] > 0).all()],
        'all_three_improve_over_original512': [t for i, t in enumerate(tasks) if (contrasts[:, i] > 0).all()],
        'leave_one_task_out_mean_gain_pp': {
            t: float(np.delete(gains, i, axis=1).mean()) for i, t in enumerate(tasks)},
        'leave_one_task_out_mean_paired_gain_pp': {
            t: float(np.delete(contrasts, i, axis=1).mean()) for i, t in enumerate(tasks)},
        'joint_source': {'path': str(joint_path), 'sha256': sha(joint_path)},
        'design_source': {'path': str(design_path), 'sha256': sha(design_path)},
        'code_sha256': sha(__file__),
        'scope': 'Fixed1600-update/six-state means on full16 exposed validation. Bank seeds change expert training, calibration subset, initial coefficient fitting and repair sample order together. Same pretrained backbone; not independent external confirmation. Full-pool access changes unique examples and empirical class weights. Seed0 original512 worker read/discarded test arrays without inference; newer workers avoid test reads. Three-seed sample SD is descriptive, not a significance test or proof of a new method.',
    }
    output = ROOT/'analysis/bank_coverage_replication_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in [
        'mean_gain_over_initial_pp', 'sample_sd_gain_over_initial_pp',
        'mean_gain_over_original512_pp', 'sample_sd_gain_over_original512_pp',
        'mean_fraction_initial_joint_gap_closed', 'all_three_improve_over_initial',
    ]}, indent=2))


if __name__ == '__main__':
    main()
