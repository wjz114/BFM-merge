"""Report every frozen gradient-basis pilot arm with construction cost and controls."""
import json
import time
from pathlib import Path

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {
        'original_only': 'gradient_basis_original_only_balanced_ce',
        'gradient': 'gradient_basis_gradient_balanced_ce',
        'random': 'gradient_basis_random_balanced_ce',
        'original_only_lr1e2': 'original512_coefficient_balanced_ce',
        'original512_full_weights': 'full_control_lr1e5_balanced_ce',
        'full_pool_full_weights': 'full_control_coverage_full_balanced_ce',
    }
    paths = {name: ROOT/'seed0'/directory/'run.json' for name, directory in directories.items()}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    reference = runs['original512_full_weights']
    tasks = reference['datasets']
    initial = reference['initial_validation']['validation']
    metrics, sources, costs = {'initial': initial}, {}, {}
    basis_path = ROOT/'seed0/local_gradient_basis/run.json'
    basis = json.loads(basis_path.read_text())
    cache_path = Path(basis['gradient_cache'])
    assert sha(cache_path) == basis['gradient_cache_sha256']
    cache = json.loads(cache_path.read_text())
    capture_path = ROOT/'seed0/gradient_basis_capture/run.json'
    capture = json.loads(capture_path.read_text())
    assert capture['tasks'] == tasks
    assert capture['plan']['input_sha256'][str(basis_path)] == sha(basis_path)
    assert not capture['model_inference_performed'] and capture['optimizer_updates'] == 0
    repair_capture_path = ROOT/'seed0/repair_basis_capture/run.json'
    repair_capture = json.loads(repair_capture_path.read_text())
    assert repair_capture['plan']['input_sha256'][str(basis_path)] == sha(basis_path)
    construction_seconds = basis['seconds']+cache['seconds']
    assert cache['optimizer_updates'] == 0 and cache['backpropagated_examples'] == 8192
    assert basis['effective_added_coefficients'] == basis['nominal_added_coefficients'] == 6128
    for name, run in runs.items():
        assert not run['engineering_smoke'] and run['datasets'] == tasks
        assert run['condition'] == 'balanced_ce'
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['fixed_tensors_exact'] and not run['test_inference_performed']
        assert sha(paths[name].with_name('merged.ckpt')) == run['checkpoint_sha256']
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'source_fit',
                    'steps', 'batch_size', 'eval_every', 'average_steps', 'seed']:
            assert run['plan'][key] == reference['plan'][key], (name, key)
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == initial[task]['BAcc']
            assert run['export_checks'][task]['max_abs_logit_error'] <= 1e-5
            assert run['dataset_audits'][task]['eeg_hash'] == reference['dataset_audits'][task]['eeg_hash']
        if name in ['original_only', 'gradient', 'random']:
            assert run['basis_kind'] == name and run['plan']['basis_kind'] == name
            assert run['plan']['lr'] == .001 and run['training_pool'] == 'original512'
            assert run['initial_parameter_and_probe_replay_exact']
            assert run['plan']['schedule_sha256'] == reference['plan']['schedule_sha256']
            assert run['dataset_audits'] == runs['original_only']['dataset_audits']
            assert run['mean_coefficient_vs_mean_state_max_abs'] <= 1e-5
            assert run['trainable_parameters'] == (6128 if name == 'original_only' else 12256)
            assert run['effective_trainable_parameters'] == run['trainable_parameters']
            assert not run['test_data_read']
            assert all(v > 0 for v in run['component_gradient_max'].values())
            if name != 'original_only':
                assert run['plan']['basis_run_sha256'] == sha(basis_path)
                assert run['basis_artifact']['sha256'] == basis['outputs'][name]['sha256']
                assert sha(run['basis_artifact']['path']) == run['basis_artifact']['sha256']
        metrics[name] = run['average_validation']
        sources[name] = {'path': str(paths[name]), 'sha256': sha(paths[name]),
                         'checkpoint_sha256': run['checkpoint_sha256']}
        uses_basis = name in ['gradient', 'random']
        costs[name] = {'fit_process_seconds': run['seconds'],
                       'additional_basis_construction_seconds': construction_seconds if uses_basis else 0.,
                       'standalone_process_plus_construction_seconds': run['seconds']['total']+(construction_seconds if uses_basis else 0.),
                       'fit_backward_examples': run['training_examples'],
                       'basis_construction_backward_examples': 8192 if uses_basis else 0,
                       'trainable_scalars': run['trainable_parameters']}
    joint_path = ROOT/'analysis/full_model_controls_lr1e5_summary.json'
    joint = json.loads(joint_path.read_text())['joint_validation_bacc']
    initial_macro = float(np.mean([initial[t]['BAcc'] for t in tasks]))
    summaries = {}
    for name, value in metrics.items():
        macro = float(np.mean([value[t]['BAcc'] for t in tasks]))
        summaries[name] = {'macro_bacc': macro, 'gain_over_initial_pp': 100*(macro-initial_macro),
                           'remaining_joint_gap_pp': 100*(joint-macro),
                           'fraction_initial_joint_gap_closed': (macro-initial_macro)/(joint-initial_macro)}
    pairs = [(name, 'initial') for name in runs]
    pairs += [('gradient', other) for other in ['original_only', 'random', 'original_only_lr1e2',
                                               'original512_full_weights', 'full_pool_full_weights']]
    pairs += [('random', 'original_only')]
    contrasts = {}
    for left, right in pairs:
        gains = {t: 100*(metrics[left][t]['BAcc']-metrics[right][t]['BAcc']) for t in tasks}
        contrasts[left+'_vs_'+right] = {
            'mean_change_pp': float(np.mean(list(gains.values()))), 'per_task_change_pp': gains,
            'improved_tasks': sum(v > 0 for v in gains.values()),
            'declined_tasks': sum(v < 0 for v in gains.values()),
            'leave_one_task_out_change_pp': {
                t: float(np.mean([v for other, v in gains.items() if other != t])) for t in tasks
            },
        }
    result = {
        'unix': time.time(), 'sources': sources, 'summaries': summaries, 'contrasts': contrasts,
        'per_task': {t: {name: value[t]['BAcc'] for name, value in metrics.items()} for t in tasks},
        'calibration_reference_macro_bacc': {
            name: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for name, run in runs.items()
        },
        'secondary': {
            name: {key: float(np.mean([run[key+'_validation'][t]['BAcc'] for t in tasks]))
                   for key in ['selected', 'final']} for name, run in runs.items()
        },
        'costs': costs, 'actual_shared_construction_seconds': construction_seconds,
        'basis_source': {'path': str(basis_path), 'sha256': sha(basis_path)},
        'local_initial_gradient_coverage': {
            'source': {'path': str(capture_path), 'sha256': sha(capture_path)},
            'energy_weighted_captured_fraction': capture['energy_weighted_captured_fraction'],
            'per_task_captured_fraction': capture['per_task_captured_fraction'],
            'diagnostic_seconds_excluded_from_method_construction': capture['seconds'],
            'scope': capture['scope'],
        },
        'finite_repair_update_capture': {
            'source': {'path': str(repair_capture_path), 'sha256': sha(repair_capture_path)},
            'captured_fraction': repair_capture['captured_fraction'],
            'diagnostic_seconds_excluded_from_method_construction': repair_capture['seconds'],
            'scope': repair_capture['scope'],
        },
        'joint_source': {'path': str(joint_path), 'sha256': sha(joint_path)}, 'joint_macro_bacc': joint,
        'code_sha256': sha(__file__),
        'primary': 'All frozen1600-update means of800/960/1120/1280/1440/1600, full16 exposed validation. No selected/final substitution.',
        'scope': 'Single bank0 pilot. All three arms use original512 rows/task, the identical sampled batches, Adam.001 and the same initialization. Gradient/random add6128 directions with matched rank/Gram/norm. Basis construction adds64 backward batches/8192 exposures on the same rows and shared CPU construction; report this overhead rather than claiming equal total compute. Historical full-pool reference accesses substantially more unique rows and has different class weights. Geometry/gradient checks are not performance evidence. A positive result still needs closest-prior aggregate-gradient comparison, seed stability and independent confirmation. Historical original512 full-weight code read/discarded test arrays without inference; new pilot avoids test reads.',
    }
    output = ROOT/'analysis/gradient_basis_pilot_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summaries': summaries,
                      'gradient_contrasts_pp': {k: v['mean_change_pp'] for k, v in contrasts.items()
                                                if k.startswith('gradient_vs_')}}, indent=2))


if __name__ == '__main__':
    main()
