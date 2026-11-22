"""Report the missing data/parameterization cell and one smaller coefficient LR."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {
        'original512_full': 'full_control_lr1e5_balanced_ce',
        'full_pool_full': 'full_control_coverage_full_balanced_ce',
        'original512_coeff_lr1e2': 'original512_coefficient_balanced_ce',
        'full_pool_coeff_lr1e2': 'coefficient_control_full_pool_balanced_ce',
        'full_pool_coeff_lr1e3': 'full_pool_coefficient_lr1e3_balanced_ce',
    }
    paths = {name: ROOT/'seed0'/directory/'run.json' for name, directory in directories.items()}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    reference = runs['full_pool_full']
    tasks = reference['datasets']
    initial = reference['initial_validation']['validation']
    records, sources = {}, {}
    for name, run in runs.items():
        assert run['datasets'] == tasks and not run['engineering_smoke']
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['fixed_tensors_exact'] and not run['test_inference_performed']
        if name == 'original512_full':
            assert run['code_sha256'] == sha('train_full_model_control.py')
            test_access = 'Historical load_dataset allocated test arrays and immediately discarded them; no test inference.'
        else:
            assert not run['test_data_read']
            test_access = 'No test arrays read or inferred.'
        for field in ['initial_checkpoint_sha256', 'calibration_sha256', 'source_fit',
                      'steps', 'batch_size', 'eval_every', 'average_steps', 'seed']:
            assert run['plan'][field] == reference['plan'][field], (name, field)
        matched = runs['original512_full' if name.startswith('original512') else 'full_pool_full']
        assert run['schedule_sha256'] == matched['schedule_sha256']
        for task in tasks:
            assert run['initial_validation']['validation'][task]['BAcc'] == initial[task]['BAcc']
            assert run['dataset_audits'][task]['eeg_hash'] == matched['dataset_audits'][task]['eeg_hash']
            assert run['export_checks'][task]['max_abs_logit_error'] <= 1e-5
        assert sha(paths[name].with_name('merged.ckpt')) == run['checkpoint_sha256']
        if '_coeff_' in name:
            assert run['trainable_parameters'] == 6128 and run['initial_parameter_and_probe_replay_exact']
            assert run['plan']['lr'] == (.001 if name.endswith('lr1e3') else .01)
        else:
            assert run['trainable_parameters'] == 25751552 and run['plan']['lr'] == 1e-5
        records[name] = {
            'macro_bacc': float(np.mean([run['average_validation'][t]['BAcc'] for t in tasks])),
            'calibration_reference_macro_bacc': float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])),
            'seconds': run['seconds'],
            'test_data_access': test_access,
            'per_task': {t: run['average_validation'][t]['BAcc'] for t in tasks},
        }
        sources[name] = {'path': str(paths[name]), 'sha256': sha(paths[name]), 'checkpoint_sha256': run['checkpoint_sha256']}
    pairs = {
        'coverage_gain_full': ('full_pool_full', 'original512_full'),
        'coverage_gain_coeff': ('full_pool_coeff_lr1e2', 'original512_coeff_lr1e2'),
        'full_minus_coeff_original512': ('original512_full', 'original512_coeff_lr1e2'),
        'full_minus_coeff_full_pool': ('full_pool_full', 'full_pool_coeff_lr1e2'),
        'smaller_coefficient_lr': ('full_pool_coeff_lr1e3', 'full_pool_coeff_lr1e2'),
        'full_minus_smaller_lr_coeff': ('full_pool_full', 'full_pool_coeff_lr1e3'),
    }
    contrasts = {}
    for name, (left, right) in pairs.items():
        changes = {t: 100*(records[left]['per_task'][t]-records[right]['per_task'][t]) for t in tasks}
        contrasts[name] = {
            'mean_change_pp': float(np.mean(list(changes.values()))), 'per_task_change_pp': changes,
            'improved_tasks': sum(v > 0 for v in changes.values()),
            'declined_tasks': sum(v < 0 for v in changes.values()),
        }
    interaction = {t: contrasts['coverage_gain_full']['per_task_change_pp'][t]
                  - contrasts['coverage_gain_coeff']['per_task_change_pp'][t] for t in tasks}
    result = {
        'unix': time.time(), 'sources': sources, 'records': records, 'contrasts': contrasts,
        'coverage_by_parameterization_interaction_pp': float(np.mean(list(interaction.values()))),
        'per_task_interaction_pp': interaction, 'code_sha256': sha(__file__),
        'scope': 'Sequential exploratory seed0 two-pool/two-parameterization comparison at each family\'s fixed LR, plus one smaller full-pool coefficient LR. Same initialization, exposures, task order and fixed mean. Within each pool the exact sampled rows match across parameterizations. Pool changes include empirical class weights. No optimizer-independent capacity bound, full-pipeline replication, independent evaluation or new-method claim.',
    }
    output = ROOT/'analysis/coefficient_followup_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'macro_bacc': {k: v['macro_bacc'] for k, v in records.items()},
                      'contrasts_pp': {k: v['mean_change_pp'] for k, v in contrasts.items()},
                      'interaction_pp': result['coverage_by_parameterization_interaction_pp']}, indent=2))


if __name__ == '__main__':
    main()
