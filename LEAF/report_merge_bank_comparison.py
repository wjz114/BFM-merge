"""Compare fixed source-validation merges across original and adapted banks."""
import argparse
import json
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256


def mean(values):
    return float(np.mean(list(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ri-only', action='store_true')
    parser.add_argument('--original-fp32', action='store_true')
    args = parser.parse_args()
    root = Path('runs/paper_merge_v2')
    old = Path('runs/aligned_merge_v1/seed0')
    paths = {'original': (old/'tensor_balanced_ce', old/'tensor_balanced_avg'),
             'ri': (root/'seed0/ri_tensor_balanced_ce', root/'seed0/ri_tensor_balanced_avg')}
    if args.original_fp32:
        paths['original'] = (root/'seed0/original_fp32_tensor_balanced_ce', root/'seed0/original_fp32_tensor_balanced_avg')
    if not args.ri_only:
        paths.update({c: (root/f'seed0/transfer_{c}_tensor_balanced_ce', root/f'seed0/transfer_{c}_tensor_balanced_avg')
                      for c in ['helpful', 'random']})
    source = json.loads((paths['original'][0]/'run.json').read_text())
    tasks = source['datasets']
    joint_run = json.loads((root/'seed0/joint_span_diagnostic/run.json').read_text())
    joint = joint_run['evaluations']['joint']
    results = {}
    for condition, (fit_dir, avg_dir) in paths.items():
        fit = json.loads((fit_dir/'run.json').read_text())
        avg = json.loads((avg_dir/'run.json').read_text())
        assert fit['datasets'] == avg['datasets'] == tasks and len(tasks) == 16
        for key in ['base_checkpoint_sha256', 'groups', 'granularity', 'class_balanced_ce',
                    'optimizer', 'lr', 'batch_size', 'seed', 'alpha_updates_actual', 'calibration_sizes',
                    'ce_class_weights', 'precision']:
            assert fit[key] == source[key], (condition, key)
        assert sha256(fit_dir/'calibration_indices.npz') == sha256(paths['original'][0]/'calibration_indices.npz')
        assert avg['source_run_sha256'] == sha256(fit_dir/'run.json')
        assert avg['averaging_steps'] == [800,960,1120,1280,1440,1600]
        assert avg['coefficient_mean_exact'] and avg['fixed_base_buffers_and_inactive_exact']
        assert max(avg['export_logit_max_abs'].values()) <= 1e-5
        assert set(avg['folded_val']) == set(tasks) and set(fit['folded_val']) == set(tasks)
        macro = mean(avg['folded_val'][task]['BAcc'] for task in tasks)
        assert abs(macro-avg['folded_val_macro_BAcc']) < 1e-12
        adaptation = []
        if condition != 'original':
            for task in tasks:
                expert_run = json.loads(Path(fit['experts'][task]).with_name('run.json').read_text())
                key = 'transfer_adaptation' if 'transfer_adaptation' in expert_run else 'ri_adaptation'
                record = expert_run[key]
                assert record['updates'] == 2500 and not record['engineering_smoke']
                adaptation.append({'task': task, 'kind': key, 'fit_seconds': record['seconds']['fit'],
                                   'expert_total_seconds': record['seconds']['expert_total']})
        results[condition] = {'selected_validation_bacc': fit['folded_val_macro_BAcc'],
                              'fixed_average_validation_bacc': macro, 'per_task': avg['folded_val'],
                              'validation_joint_gap_pp': 100*(joint['macro_bacc']-macro),
                              'fit_seconds': fit['seconds'], 'average_seconds': avg['seconds'],
                              'adaptation_records': adaptation,
                              'sum_recorded_expert_fit_seconds': sum(row['fit_seconds'] for row in adaptation),
                              'adaptation_training_examples_processed': 16*2500*64 if adaptation else 0,
                              'checkpoint_sha256': sha256(avg_dir/'merged.ckpt'),
                              'source_run_hashes': {'fit': sha256(fit_dir/'run.json'), 'average': sha256(avg_dir/'run.json')}}
    comparisons = {}
    pairs = [('ri','original')]
    if not args.ri_only:
        pairs += [('helpful','ri'), ('random','ri'), ('helpful','random'), ('helpful','original')]
    for treatment, control in pairs:
        gains = {task: 100*(results[treatment]['per_task'][task]['BAcc']-results[control]['per_task'][task]['BAcc']) for task in tasks}
        total = sum(gains.values())
        comparisons[f'{treatment}_vs_{control}'] = {
            'macro_gain_pp': mean(gains.values()), 'per_task_gain_pp': gains,
            'improved_tasks': sum(value > 0 for value in gains.values()),
            'declining_tasks': sum(value < 0 for value in gains.values()),
            'worst_task': min(gains, key=gains.get), 'worst_task_change_pp': min(gains.values()),
            'signed_net_gain_contribution': {task: value/total if abs(total)>1e-10 else None for task,value in gains.items()},
            'leave_one_task_out_gain_pp': {task: mean(value for other,value in gains.items() if other != task) for task in tasks}}
    baseline = results['original']['fixed_average_validation_bacc']
    for row in results.values():
        row['fraction_original_validation_joint_gap_closed'] = (row['fixed_average_validation_bacc']-baseline)/(joint['macro_bacc']-baseline)
    result = {'original_training_precision': source['precision'], 'conditions': results, 'comparisons': comparisons, 'joint_validation_bacc': joint['macro_bacc'],
              'scope': 'Seed0, full development-exposed source validation. Fixed average is primary; selected checkpoint secondary. This is not official test or independent confirmation.',
              'cost_scope': 'Recorded fit seconds sum expert workloads; they are not concurrent wall time. Include reused RI experts once for each condition. Gate teacher forwards, cache/setup, diagnostics and exploratory trial costs require separate reporting; the fit sum is not full method cost.',
              'code_sha256': sha256(Path(__file__))}
    filename = 'ri_merge_comparison.json' if args.ri_only else 'transfer_merge_comparison.json'
    if args.original_fp32:
        filename = filename.replace('.json', '_fp32_control.json')
    (root/'analysis'/filename).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'validation_bacc': {c:r['fixed_average_validation_bacc'] for c,r in results.items()},
                      'paired_gains_pp': {c:r['macro_gain_pp'] for c,r in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
