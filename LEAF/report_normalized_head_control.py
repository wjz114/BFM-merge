"""Report the fixed normalized-versus-raw head regression experiment."""
import json
import time
from pathlib import Path

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    paths = {
        'design': ROOT/'inputs/normalized_head_sample_stream_design.json',
        'statistics': ROOT/'seed0/normalized_head_statistics/run.json',
        'fit': ROOT/'seed0/normalized_head_fit/run.json',
        'evaluation': ROOT/'seed0/normalized_head_evaluation/run.json',
        'raw_statistics': ROOT/'seed0/head_sample_stream_statistics/run.json',
        'raw_fit': ROOT/'seed0/head_sample_stream_fit/run.json',
        'raw_evaluation': ROOT/'seed0/head_sample_stream_evaluation/run.json',
        'raw_report': ROOT/'analysis/head_sample_stream_control_summary.json',
    }
    source = {k: json.loads(p.read_text()) for k, p in paths.items()}
    design, stats, fit, evaluation = (source[k] for k in ['design', 'statistics', 'fit', 'evaluation'])
    raw_stats, raw_fit, raw_eval, raw_report = (source[k] for k in ['raw_statistics', 'raw_fit', 'raw_evaluation', 'raw_report'])
    tasks = evaluation['datasets']
    assert all(source[k]['datasets'] == tasks for k in paths if k != 'design')
    assert stats['statistics_normalization'] == 'frozen_initial_raw_norm'
    assert evaluation['plan']['fit_run'] == str(paths['fit'])
    assert fit['plan']['statistics_cache'] == str(paths['statistics'])
    assert stats['plan']['raw_statistics_cache'] == str(paths['raw_statistics'])
    assert fit['plan']['design'] == stats['plan']['design'] == str(paths['design'])
    assert raw_report['sources']['evaluation']['sha256'] == sha(paths['raw_evaluation'])
    assert stats['plan']['schedule'] == raw_stats['plan']['schedule']
    assert fit['lambda'] == raw_fit['lambda'] == design['lambda'] == .05
    raw_design = json.loads(Path(raw_fit['plan']['design']).read_text())
    assert design['random_seed'] == raw_design['random_seed']
    assert stats['unique_training_examples'] == raw_stats['unique_training_examples'] == 117785
    assert stats['fitting_stream_examples'] == raw_stats['fitting_stream_examples'] == 204800
    assert not stats['validation_used_in_statistics'] and not fit['validation_used_in_fit']
    assert not any(source[k]['test_data_read'] for k in ['statistics', 'fit', 'evaluation'])
    assert evaluation['initial_bacc_replays_exact'] and evaluation['trunk_tensors_unchanged']
    max_raw_error = 0.
    for task in tasks:
        row, raw_row = stats['records'][task], raw_stats['records'][task]
        for key in ['fit_examples', 'unique_fit_examples', 'stream_indices_hash', 'unique_fit_indices_hash',
                    'class_weights', 'class_counts', 'train_eeg_hash', 'train_labels_hash',
                    'instruction_hash', 'prototypes_hash', 'expert_sha256', 'fitting_stream_class_counts']:
            assert row[key] == raw_row[key], (task, key)
        assert row['initial_raw_norm_in_stream']['min'] > 1e-12
        max_raw_error = max(max_raw_error, *row['raw_stream_moment_relative_errors'].values())
        assert max(row['old512_moment_relative_errors'].values()) < 5e-5
        for key in ['old512_initial_replay', 'old512_teacher_replay']:
            assert row[key]['max_abs_logit_error'] <= 1e-5
            assert all(r['margin_within_twice_error'] for r in row[key]['changed_rows'])
        assert evaluation['dataset_audits'][task] == raw_eval['dataset_audits'][task]
        for split in ['calibration', 'validation']:
            assert evaluation['evaluations']['initial'][task][split]['BAcc'] == raw_eval['evaluations']['initial'][task][split]['BAcc']
    assert max_raw_error < 5e-5
    for condition, row in fit['outputs'].items():
        assert row['solver']['actual_relative_residual'] <= 1.01e-8
        assert sha(row['path']) == row['sha256'] == evaluation['checkpoint_sha256'][condition]
    baseline_path = Path(raw_report['sources']['full_weight_repair']['path'])
    joint_path = Path(raw_report['sources']['Joint']['path'])
    assert sha(baseline_path) == raw_report['sources']['full_weight_repair']['sha256']
    assert sha(joint_path) == raw_report['sources']['Joint']['sha256']
    full, joint = json.loads(baseline_path.read_text()), json.loads(joint_path.read_text())
    values = {}
    for regime, result in [('raw', raw_eval), ('normalized', evaluation)]:
        for condition in design['conditions']:
            values[regime+'_'+condition] = np.array([result['evaluations'][condition][t]['validation']['BAcc'] for t in tasks])
    values['full_weight_repair'] = np.array([full['average_validation'][t]['BAcc'] for t in tasks])
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    contrasts = {}

    def contrast(name, gains):
        contrasts[name] = {'macro_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
                           'improved': int((gains > 0).sum()), 'declined': int((gains < 0).sum()),
                           'leave_one_task_out_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)}}

    pairs = [('normalized_prototype_contrast', c) for c in
             ['raw_prototype_contrast', 'normalized_initial', 'normalized_full_feature_ridge',
              'normalized_rotated_contrast_control', 'full_weight_repair']]
    pairs += [('normalized_'+c, 'raw_'+c) for c in ['full_feature_ridge', 'rotated_contrast_control']]
    pairs += [('normalized_'+c, 'normalized_initial') for c in ['full_feature_ridge', 'rotated_contrast_control']]
    for treatment, control in pairs:
        contrast(treatment+'_vs_'+control, 100*(values[treatment]-values[control]))
    contrast('normalization_x_semantic_vs_full_feature', 100*(
        values['normalized_prototype_contrast']-values['normalized_full_feature_ridge']
        -values['raw_prototype_contrast']+values['raw_full_feature_ridge']))
    initial, target = values['normalized_initial'].mean(), values['Joint'].mean()
    result = {
        'unix': time.time(), 'code_sha256': sha(__file__), 'datasets': tasks,
        'sources': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'macro_bacc': {k: float(v.mean()) for k, v in values.items()}, 'contrasts': contrasts,
        'fraction_initial_joint_gap_closed': {c: float((values['normalized_'+c].mean()-initial)/(target-initial))
                                             for c in design['conditions'] if c != 'initial'},
        'old512_calibration_bacc': {'raw': raw_eval['summaries']['calibration'], 'normalized': evaluation['summaries']['calibration']},
        'raw_stream_moment_replay_max_relative_error': max_raw_error,
        'ridge_rule': {'lambda': fit['lambda'], 'raw_mu': raw_fit['mu'], 'normalized_mu': fit['mu']},
        'costs': {'statistics_seconds': stats['seconds'], 'fit_seconds': fit['seconds'],
                  'native_evaluation_seconds': evaluation['seconds'],
                  'student_forward_examples': stats['student_forward_examples'],
                  'teacher_forward_examples': stats['teacher_forward_examples'],
                  'unique_fitting_examples': stats['unique_training_examples'],
                  'fitting_multiset_examples': stats['fitting_stream_examples']},
        'scope': 'Fixed single-bank normalized-MSE control on the exact raw-regression/full-weight-repair sample stream. '
                 'Same full-pool class weights, frames, lambda and trace rule; absolute ridge changes with normalized input trace. '
                 'Native ordinary checkpoints evaluated on exposed validation. Normalized MSE is prior art. '
                 'No ridge/scale sweep, independent-confirmation or novelty claim; no test reads.'}
    path = ROOT/'analysis/normalized_head_control_summary.json'
    with path.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'],
                      'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
