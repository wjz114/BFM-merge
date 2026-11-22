"""Cached task-mean gradient variance under the already frozen grouping rule."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    diagnostic_path = ROOT/'analysis/parameter_gradient_conflicts.json'
    design_path = ROOT/'inputs/task_mixing_design.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    design = json.loads(design_path.read_text())
    tasks = diagnostic['datasets']
    mixtures = {}
    for mode in design['arms']:
        entry = design['schedules'][mode]
        assert sha(entry['micro_schedule']) == entry['micro_schedule_sha256']
        with np.load(entry['micro_schedule']) as schedule:
            mixtures[mode] = np.stack([(schedule['task'] == i).mean(1) for i in range(16)], axis=1)
        assert np.array_equal(mixtures[mode].sum(1), np.ones(1600))
        assert np.array_equal(mixtures[mode].mean(0), np.full(16, 1/16))
    results = {}
    for stage, row in diagnostic['stages'].items():
        gram = np.asarray(row['components']['all']['gram'], dtype=np.float64)
        centered = gram-gram.mean(0, keepdims=True)-gram.mean(1, keepdims=True)+gram.mean()
        original_variance = np.trace(centered)/16
        output = {}
        for mode, mixture in mixtures.items():
            variance = np.einsum('bi,ij,bj->b', mixture, centered, mixture).mean()
            dots = mixture@gram
            output[mode] = {'between_task_mean_variance': float(variance),
                            'variance_over_single_task': float(variance/original_variance),
                            'variance_over_global_mean_norm_sq': float(variance/gram.mean()),
                            'fraction_update_task_pairs_with_negative_inner_product': float((dots < 0).mean()),
                            'negative_fraction_per_task': dict(zip(tasks, (dots < 0).mean(0).tolist()))}
        assert abs(output['same_task_micro4']['variance_over_single_task']-1) < 1e-12
        results[stage] = output
    result = {'unix': time.time(), 'code_sha256': sha(__file__),
              'input_sha256': {str(p): sha(p) for p in [diagnostic_path, design_path]},
              'stages': results, 'uniform_four_distinct_tasks_expected_variance_ratio': (16-4)/(4*(16-1)),
              'new_model_inference': False, 'new_backpropagation': False,
              'validation_used_in_statistics': False, 'test_data_read': False,
              'scope': 'Apply fixed full-training schedule task identities to existing original512 task-mean gradients at three frozen checkpoints. Exact algebra for these cached means. It ignores within-task minibatch variance, full-pool class-weight/data differences, temporal correlation, changing parameters and Adam moments. It does not predict validation improvement or prove actual optimizer-noise reduction.'}
    with (ROOT/'analysis/task_mixing_cached_variance.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: {m: v['variance_over_single_task'] for m, v in row.items()} for k, row in results.items()}, indent=2))


if __name__ == '__main__':
    main()
