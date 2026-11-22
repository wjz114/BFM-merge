"""Compare fixed-update span interventions under the two training pools."""
import json
import time
from pathlib import Path

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {
        'original512': 'full_control_span_diagnostic',
        'full_training': 'full_pool_span_diagnostic',
    }
    sources, records = {}, {}
    for pool, directory in directories.items():
        path = ROOT/'seed0'/directory/'run.json'
        run = json.loads(path.read_text())
        assert run['exact_validation_bacc_replays'] and run['fixed_tensors_exact']
        assert run['exact_fp32_update_recomposition']
        assert not run['test_data_read'] and not run['test_inference_performed']
        for filename, expected in run['plan']['input_sha256'].items():
            assert sha(filename) == expected, filename
        assert sha('diagnose_full_control_span.py') == run['code_sha256']
        for condition, filename in run['exported_paths'].items():
            assert sha(filename) == run['checkpoint_sha256'][condition]
        evaluations = run['evaluations']
        tasks = list(evaluations['initial'])
        contrasts = {}
        for condition in ['full_update', 'parallel_only', 'orthogonal_only']:
            changes = {
                task: 100*(evaluations[condition][task]['validation']['BAcc']
                           - evaluations['initial'][task]['validation']['BAcc'])
                for task in tasks
            }
            contrasts[condition] = {
                'mean_gain_pp': float(np.mean(list(changes.values()))),
                'per_task_gain_pp': changes,
                'improved_tasks': sum(v > 0 for v in changes.values()),
                'declined_tasks': sum(v < 0 for v in changes.values()),
                'leave_one_task_out_gain_pp': {
                    task: float(np.mean([v for t, v in changes.items() if t != task]))
                    for task in tasks
                },
            }
        records[pool] = {
            'summaries': run['summaries'], 'contrasts': contrasts,
            'geometry_totals': run['geometry_totals'], 'seconds': run['seconds'],
            'initial_checkpoint_sha256': run['checkpoint_sha256']['initial'],
        }
        sources[pool] = {'path': str(path), 'sha256': sha(path)}
    assert records['original512']['initial_checkpoint_sha256'] == records['full_training']['initial_checkpoint_sha256']
    for pool in records:
        assert records[pool]['summaries']['validation']['initial'] == records['original512']['summaries']['validation']['initial']
    result = {
        'unix': time.time(), 'sources': sources, 'records': records,
        'code_sha256': sha(Path(__file__)),
        'scope': 'Same initial checkpoint and existing tensorwise expert-span decomposition of two fixed seed0 full-weight updates. Full-pool sampling replicas do not replicate this intervention. No tuning, Joint assistance or official-test inference. Parameter energy is coordinate-dependent; functional ablations are not a coefficient-capacity bound or a new mechanism. Training-data access differs between the two sources.',
    }
    output = ROOT/'analysis/coverage_span_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({pool: {
        'validation': row['summaries']['validation'],
        'orthogonal_energy_fraction': row['geometry_totals']['orthogonal_energy_fraction'],
        'task_counts': {c: {k: v for k, v in contrast.items() if k in ['improved_tasks', 'declined_tasks']}
                        for c, contrast in row['contrasts'].items()},
    } for pool, row in records.items()}, indent=2))


if __name__ == '__main__':
    main()
