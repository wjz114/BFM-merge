"""Report all fixed ridge heads on the exact full-pool repair sample stream."""
import json
import time

import numpy as np

from aligned_support import array_hash
from report_coverage_control import ROOT, sha


def main():
    paths = {
        'design': ROOT/'inputs/contrast_head_sample_stream_design.json',
        'statistics': ROOT/'seed0/head_sample_stream_statistics/run.json',
        'fit': ROOT/'seed0/head_sample_stream_fit/run.json',
        'evaluation': ROOT/'seed0/head_sample_stream_evaluation/run.json',
        'old512_fit': ROOT/'seed0/prototype_contrast_head_fit_v3/run.json',
        'old512_evaluation': ROOT/'seed0/prototype_contrast_head_evaluation/run.json',
        'full_weight_repair': ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json',
        'Joint': ROOT/'analysis/bank_joint_fp32_validation/run.json',
    }
    source = {k: json.loads(p.read_text()) for k, p in paths.items()}
    design, stats, fit, evaluation = (source[k] for k in ['design', 'statistics', 'fit', 'evaluation'])
    old, full, joint = (source[k] for k in ['old512_evaluation', 'full_weight_repair', 'Joint'])
    tasks = evaluation['datasets']
    assert all(source[k]['datasets'] == tasks for k in ['statistics', 'fit', 'old512_fit', 'old512_evaluation', 'full_weight_repair', 'Joint'])
    assert evaluation['plan']['fit_run'] == str(paths['fit'])
    assert fit['plan']['statistics_cache'] == str(paths['statistics'])
    assert fit['plan']['design'] == stats['plan']['design'] == str(paths['design'])
    assert fit['lambda'] == source['old512_fit']['lambda'] == design['lambda'] == .05
    old_design = json.loads((ROOT/'inputs/prototype_contrast_calibration_design.json').read_text())
    assert design['random_seed'] == old_design['random_seed']
    assert stats['unique_training_examples'] == design['unique_fit_examples_total'] == 117785
    assert stats['fitting_stream_examples'] == full['training_examples'] == 204800
    assert not stats['validation_used_in_statistics'] and not fit['validation_used_in_fit']
    assert not any(source[k]['test_data_read'] for k in ['statistics', 'fit', 'evaluation'])
    assert evaluation['initial_bacc_replays_exact'] and evaluation['trunk_tensors_unchanged']
    schedule = np.load(full['plan']['schedule'])
    assert sha(stats['plan']['schedule']) == full['schedule_sha256']
    for index, task in enumerate(tasks):
        row = stats['records'][task]
        ids = schedule['local_indices'][schedule['task'] == index].reshape(-1).astype(np.int64)
        assert row['fit_examples'] == len(ids) == 12800
        assert row['unique_fit_examples'] == len(np.unique(ids)) == design['unique_fit_examples_per_task'][task]
        assert row['stream_indices_hash'] == array_hash(ids)
        assert row['unique_fit_indices_hash'] == array_hash(np.unique(ids))
        assert row['class_weights'] == full['dataset_audits'][task]['class_weights']
        assert row['train_eeg_hash'] == full['dataset_audits'][task]['training_pool_eeg_hash']
        assert max(row['old512_moment_relative_errors'].values()) < 5e-5
        for key in ['old512_initial_replay', 'old512_teacher_replay']:
            assert row[key]['max_abs_logit_error'] <= 1e-5
            assert all(r['margin_within_twice_error'] for r in row[key]['changed_rows'])
        assert evaluation['evaluations']['initial'][task]['validation']['BAcc'] == old['evaluations']['initial'][task]['validation']['BAcc']
        assert evaluation['evaluations']['initial'][task]['validation']['BAcc'] == full['initial_validation']['validation'][task]['BAcc']
    for condition, row in fit['outputs'].items():
        assert row['solver']['actual_relative_residual'] <= 1.01e-8
        assert sha(row['path']) == row['sha256'] == evaluation['checkpoint_sha256'][condition]
    values = {}
    for regime, result in [('original512', old), ('sample_stream', evaluation)]:
        for condition in design['conditions']:
            values[regime+'_'+condition] = np.array([result['evaluations'][condition][t]['validation']['BAcc'] for t in tasks])
    values['full_weight_repair'] = np.array([full['average_validation'][t]['BAcc'] for t in tasks])
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    contrasts = {}

    def contrast(name, gains):
        contrasts[name] = {
            'macro_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
            'improved': int((gains > 0).sum()), 'declined': int((gains < 0).sum()),
            'leave_one_task_out_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)}}

    pairs = [('sample_stream_prototype_contrast', c) for c in
             ['sample_stream_initial', 'sample_stream_full_feature_ridge', 'sample_stream_rotated_contrast_control',
              'original512_prototype_contrast', 'full_weight_repair']]
    pairs += [('sample_stream_'+c, 'original512_'+c) for c in ['full_feature_ridge', 'rotated_contrast_control']]
    pairs += [('sample_stream_'+c, 'sample_stream_initial') for c in ['full_feature_ridge', 'rotated_contrast_control']]
    for treatment, control in pairs:
        contrast(treatment+'_vs_'+control, 100*(values[treatment]-values[control]))
    contrast('data_regime_x_semantic_vs_full_feature', 100*(
        values['sample_stream_prototype_contrast']-values['sample_stream_full_feature_ridge']
        -values['original512_prototype_contrast']+values['original512_full_feature_ridge']))
    initial, target = values['sample_stream_initial'].mean(), values['Joint'].mean()
    result = {
        'unix': time.time(), 'code_sha256': sha(__file__), 'datasets': tasks,
        'sources': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'macro_bacc': {k: float(v.mean()) for k, v in values.items()}, 'contrasts': contrasts,
        'fraction_initial_joint_gap_closed': {c: float((values['sample_stream_'+c].mean()-initial)/(target-initial))
                                             for c in design['conditions'] if c != 'initial'},
        'old512_calibration_bacc': {'original512': old['summaries']['calibration'], 'sample_stream': evaluation['summaries']['calibration']},
        'costs': {'statistics_seconds': stats['seconds'], 'fit_seconds': fit['seconds'],
                  'native_evaluation_seconds': evaluation['seconds'],
                  'student_forward_examples': stats['student_forward_examples'],
                  'teacher_forward_examples': stats['teacher_forward_examples'],
                  'unique_fitting_examples': stats['unique_training_examples'],
                  'fitting_multiset_examples': stats['fitting_stream_examples']},
        'scope': 'Single bank0 fixed data-regime intervention. The expanded ridge fits use exactly the full-weight repair fitting identities, '
                 'multiplicities and class weights, but require own-expert targets and frozen-student sufficient statistics. '
                 'Original512 versus sample stream also changes empirical class weights/multiplicities; it is not a pure sample-count experiment. '
                 'Constraint counts do not prove a unique failure mechanism. Random output frames match rank and overlap geometry, not fitted update norm. '
                 'All16 validation has been exposed; no novelty or independent-confirmation claim. No scale/ridge sweep or selected-checkpoint substitution.'}
    path = ROOT/'analysis/head_sample_stream_control_summary.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
