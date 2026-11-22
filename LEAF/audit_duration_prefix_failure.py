"""Preserve the stopped duration runs and audit the invalid exact-replay assumption."""
import json
import time
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/duration4800_ce_design.json'
    design = json.loads(design_path.read_text())
    replay_path = root/'analysis/bank_ce_replay_verification.json'
    replay = json.loads(replay_path.read_text())
    sources = {str(p): sha256(p) for p in [design_path, replay_path, Path(__file__)]}
    for name, digest in design['source_sha256'].items():
        assert sha256(Path(name)) == digest, name
    schedules = [np.load(design[k]) for k in ['schedule', 'prefix_schedule']]
    for start in [0, 1600, 3200]:
        for key in ['task', 'local_indices']:
            assert np.array_equal(schedules[0][key][start:start+1600], schedules[1][key])
    arms = {}
    for arm, spec in design['arms'].items():
        out = Path(spec['outdir'])
        assert not (out/'run.json').exists()
        plan = json.loads(Path(spec['plan']).read_text())
        for name, digest in plan['source_sha256'].items():
            assert sha256(Path(name)) == digest, name
        for key in ['initial_checkpoint', 'calibration', 'coverage_audit', 'schedule']:
            assert sha256(Path(plan[key])) == plan[key+'_sha256']
        old_dir = Path(spec['control']).parent
        paths = [out/'prefix_replay.json', out/'history.json', out/'dynamics.jsonl',
                 old_dir/'history.json', old_dir/'dynamics.jsonl', Path(spec['log'])]
        sources.update({str(p): sha256(p) for p in paths})
        prefix = json.loads(paths[0].read_text())
        assert set(prefix) == {'800'} and 'AssertionError: (800,' in paths[-1].read_text()
        histories = [json.loads(p.read_text()) for p in [paths[1], paths[3]]]
        dynamics = [[json.loads(s) for s in p.read_text().splitlines()] for p in [paths[2], paths[4]]]
        first = next(({'new': x, 'old': y} for x, y in zip(*dynamics) if x != y), None)
        history_delta = [{'step': x['step'], 'macro_bacc_delta_pp': 100*(x['macro_bacc']-y['macro_bacc']),
                          'metrics_exact': x['validation'] == y['validation']}
                         for x, y in zip(*histories)]
        assert histories[0][0] == histories[1][0] and dynamics[0][0] == dynamics[1][0]
        changed = prefix['800']['nonidentical_state_tensors']
        arms[arm] = {'stopped_at_update': 800, 'first_logged_dynamics_difference': first,
                     'history_comparison': history_delta, 'prefix_reference': prefix['800']['reference'],
                     'nonidentical_state_tensors': len(changed), 'max_abs_state_difference': max(changed.values()),
                     'all_validation_metrics_exact_at800': prefix['800']['all_validation_metrics_exact'],
                     'initial_metrics_and_first_logged_gradient_exact': True,
                     'source_and_input_hashes_unchanged': True,
                     'complete_scientific_result': False}
    result = {'unix': time.time(), 'source_sha256': sources, 'arms': arms,
              'original_engineering_replay_scope': replay['scope'],
              'diagnosis': 'The bitwise gate was incorrectly extrapolated from a deterministic32-step engineering replay to scientific runs whose configure() does not enable deterministic algorithms or cuDNN deterministic mode. Frozen sources, schedules and input hashes pass; initial metrics and step1 logged gradients match. The subsequent small differences are consistent with native numerical nondeterminism; this audit does not isolate a specific CUDA operator.',
              'resolution': 'Preserve both stopped runs and frozenv1 artifacts. A new version will retain the native scientific update recipe and compare fixed1600/4800 means from the SAME4800-step trajectory. Historical-prefix differences remain descriptive and are not accepted by a newly chosen tolerance. No historical checkpoint is substituted into a live optimizer trajectory.',
              'wasted_optimizer_updates': 1600, 'new_official_test_access': False}
    target = root/'analysis/duration_prefix_failure.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'path': str(target), 'sha256': sha256(target), 'arms': arms}, indent=2))


if __name__ == '__main__':
    main()
