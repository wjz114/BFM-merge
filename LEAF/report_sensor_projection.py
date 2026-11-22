"""Report the full frozen sensor intervention, including its negligible effect."""
import json
from pathlib import Path
from statistics import mean

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'seed0/sensor_projection_diagnostic_v2/run.json'
    run = json.loads(source.read_text())
    subspace_file = root/'seed0/sensor_subspace_memory/run.json'
    subspace = json.loads(subspace_file.read_text())
    assert sha256(subspace_file) == run['subspace_run_sha256']
    assert not run['test_inference_performed'] and not run['test_data_read']
    tasks = list(subspace['records'])
    assert len(tasks) == 16
    assert all(list(v) == tasks for v in run['evaluations'].values())
    for condition, path in run['exported_paths'].items():
        assert sha256(Path(path)) == run['checkpoint_sha256'][condition]
    joint = json.loads((root/'seed0/joint_span_diagnostic/run.json').read_text())['evaluations']['joint']['macro_bacc']
    baseline = mean(row['validation']['BAcc'] for row in run['evaluations']['original'].values())
    summaries = {
        c: {'validation_macro_bacc': mean(v['validation']['BAcc'] for v in rows.values()),
            'calibration_macro_bacc': mean(v['calibration']['BAcc'] for v in rows.values()),
            'joint_gap_pp': 100*(joint-mean(v['validation']['BAcc'] for v in rows.values())),
            'fraction_original_joint_gap_closed': (mean(v['validation']['BAcc'] for v in rows.values())-baseline)/(joint-baseline)}
        for c, rows in run['evaluations'].items()}
    own = {}
    for split in ['calibration', 'validation']:
        checks = [run['own_checks'][t][split] for t in tasks]
        own[split] = {
            'max_abs_logit_error': max(v['comparison']['max_abs_logit_error'] for v in checks),
            'changed_predictions': sum(v['comparison']['changed_predictions'] for v in checks),
            'max_abs_task_bacc_change_pp': max(100*abs(v['canonical']['BAcc']-v['original']['BAcc']) for v in checks),
            'changed_predictions_within_numerical_margin': all(row['margin_within_twice_error'] for v in checks for row in v['comparison']['changed_rows'])}
    rows = {t: {**{k: subspace['records'][t][k] for k in ['rank', 'nullity', 'relative_calibration_null_energy_root', 'relative_validation_null_energy_root']},
                'spatial_delta_null_energy_fraction': run['geometry'][t]['own_null_delta_energy_fraction'],
                'validation_canonical_gain_pp': run['contrasts']['validation']['canonical_vs_original']['per_task_change_pp'][t],
                'own_validation_logit_max_error': run['own_checks'][t]['validation']['comparison']['max_abs_logit_error']}
            for t in tasks}
    result = {'summaries': summaries, 'own_function_checks': own, 'per_task_structure_and_effect': rows,
              'contrasts': run['contrasts'], 'joint_validation_bacc': joint,
              'subspace_seconds': subspace['seconds'], 'model_diagnostic_seconds': run['seconds'],
              'source_run_sha256': sha256(source), 'code_sha256': sha256(Path(__file__)),
              'conclusion': 'The calibration-derived own-null intervention preserves source functions numerically and is less damaging than norm/direction controls, but its absolute macro validation effect is negligible relative to the Joint gap. It does not establish the proposed main bottleneck or a useful merging contribution.',
              'decision': 'Do not expand this projection into coefficient refitting or multi-seed training on the strength of a0.0014pp mean effect. Keep the mechanism diagnostic and negative practical result. This is a scope decision, not a claim that all possible sensor-space interventions are ruled out.',
              'scope': 'Seed0, all16 full exposed validation tasks, fixed original FP32 tensor coefficients, original512 calibration. No independent outer confirmation or novelty claim.'}
    (root/'analysis/sensor_projection_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summaries': summaries, 'own_function_checks': own}, indent=2))


if __name__ == '__main__':
    main()
