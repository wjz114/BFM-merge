"""Compare full-pool repair interventions in original and both augmented spaces."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    directories = {'original_only': 'full_pool_span_diagnostic',
                   'gradient': 'full_pool_augmented_gradient_span_diagnostic',
                   'random': 'full_pool_augmented_random_span_diagnostic'}
    paths = {kind: ROOT/'seed0'/name/'run.json' for kind, name in directories.items()}
    runs = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
    capture_path = ROOT/'seed0/repair_basis_capture/run.json'
    capture = json.loads(capture_path.read_text())
    reference = runs['original_only']
    tasks = list(reference['evaluations']['initial'])
    values, summaries = {}, {}
    for kind, run in runs.items():
        assert run['plan']['source'] == reference['plan']['source']
        assert run['exact_validation_bacc_replays'] and run['fixed_tensors_exact']
        assert run['exact_fp32_update_recomposition']
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert abs(1-run['geometry_totals']['orthogonal_energy_fraction']
                   -capture['captured_fraction']['full_pool'][kind]) < 1e-10
        if kind != 'original_only':
            assert run['basis_kind'] == kind
        for condition, path in run['exported_paths'].items():
            assert sha(path) == run['checkpoint_sha256'][condition]
            assert list(run['evaluations'][condition]) == tasks
        for condition in ['initial', 'full_update']:
            assert all(run['evaluations'][condition][t]['validation']['BAcc']
                       == reference['evaluations'][condition][t]['validation']['BAcc'] for t in tasks)
        values[kind] = {condition: np.array([run['evaluations'][condition][t]['validation']['BAcc'] for t in tasks])
                        for condition in run['evaluations']}
        summaries[kind] = {
            'macro_bacc': {condition: float(array.mean()) for condition, array in values[kind].items()},
            'captured_update_energy_fraction': 1-run['geometry_totals']['orthogonal_energy_fraction'],
            'factorial_interaction_pp': run['summaries']['validation']['factorial_interaction_pp'],
            'seconds': run['seconds'],
        }
    contrasts = {}
    for kind in values:
        for condition in ['parallel_only', 'orthogonal_only']:
            gains = 100*(values[kind][condition]-values[kind]['initial'])
            contrasts[kind+'_'+condition+'_vs_initial'] = {
                'mean_gain_pp': float(gains.mean()),
                'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
                'improved_tasks': int((gains > 0).sum()), 'declined_tasks': int((gains < 0).sum()),
                'leave_one_task_out_gain_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)},
            }
    for other in ['random', 'original_only']:
        gains = 100*(values['gradient']['parallel_only']-values[other]['parallel_only'])
        contrasts['gradient_parallel_vs_'+other+'_parallel'] = {
            'mean_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
            'improved_tasks': int((gains > 0).sum()), 'declined_tasks': int((gains < 0).sum()),
        }
    result = {
        'unix': time.time(), 'summaries': summaries, 'contrasts': contrasts,
        'sources': {kind: {'path': str(path), 'sha256': sha(path)} for kind, path in paths.items()},
        'capture_source': {'path': str(capture_path), 'sha256': sha(capture_path)},
        'code_sha256': sha(__file__),
        'scope': 'All fixed unit-scale projections and complements of one already-trained full-pool repair. The original expert span and both frozen extensions are evaluated with exact initial/full-model BAcc replays. Projected states use the completed full-weight update and are diagnostics, not cheap candidate results or bounds on the best model in a space. No tuning, new fitting or Joint assistance; exposed validation and one bank only.',
    }
    (ROOT/'analysis/augmented_span_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
