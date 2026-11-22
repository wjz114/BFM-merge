"""Report the complete conditional FiLM intervention and its limitations."""
import json
from pathlib import Path
from statistics import mean

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source = root/'seed0/film_response_diagnostic/run.json'
    run = json.loads(source.read_text())
    geometry = json.loads((root/'analysis/film_identifiability_geometry.json').read_text())
    tasks = geometry['tasks']
    assert len(tasks)==16 and all(list(v)==tasks for v in run['evaluations'].values())
    assert not run['test_data_read'] and not run['test_inference_performed']
    for condition, path in run['exported_paths'].items():
        assert sha256(Path(path)) == run['checkpoint_sha256'][condition]
    joint = json.loads((root/'seed0/joint_span_diagnostic/run.json').read_text())['evaluations']['joint']['macro_bacc']
    baseline = mean(v['validation']['BAcc'] for v in run['evaluations']['original'].values())
    summaries = {
        name: {'validation_macro_bacc': mean(v['validation']['BAcc'] for v in rows.values()),
               'calibration_macro_bacc': mean(v['calibration']['BAcc'] for v in rows.values()),
               'fraction_joint_gap_closed': (mean(v['validation']['BAcc'] for v in rows.values())-baseline)/(joint-baseline),
               'oracle_only': name=='joint_film_oracle'}
        for name,rows in run['evaluations'].items()}
    result = {'summaries': summaries, 'contrasts': run['contrasts'],
              'construction': run['construction'], 'instruction_groups': geometry['instruction_groups'],
              'mean_own_null_weight_energy_fraction': mean(v['own_null_weight_energy_fraction'] for v in geometry['per_expert'].values()),
              'mean_union_invisible_weight_energy_fraction': mean(v['union_null_weight_energy_fraction'] for v in geometry['per_expert'].values()),
              'mean_own_null_union_visible_weight_energy_fraction': mean(v['own_null_but_union_visible_weight_energy_fraction'] for v in geometry['per_expert'].values()),
              'source_run_sha256': sha256(source), 'code_sha256': sha256(Path(__file__)), 'seconds': run['seconds'],
              'conclusion': 'Own-condition response matching reduces the original experts\' FiLM response mismatch to the exact floor imposed by six identical MI conditions, but lowers final merged BAcc. Matching affine responses is not sufficient for improved composed behavior. Joint FiLM substitution also hurts on average. The generic affine regression operation is existing methodology.',
              'decision': 'No immediate expansion of response matching or own-null FiLM projection into trained banks. Preserve all task results and focus next on the positive ordinary CE/weight-average baseline and its sampling stability.',
              'scope': 'Seed0, full16 exposed validation, fixed existing instructions and original merged initialization. No independent confirmation, new architecture, or runtime model bank.'}
    (root/'analysis/film_response_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
