"""Fixed-state first-moment fluctuations under the frozen task schedules."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    diagnostic_path = ROOT/'analysis/parameter_gradient_conflicts.json'
    design_path = ROOT/'inputs/task_mixing_design.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    design = json.loads(design_path.read_text())
    mixtures, moment_coefficients = {}, {}
    for mode in design['arms']:
        entry = design['schedules'][mode]
        assert sha(entry['micro_schedule']) == entry['micro_schedule_sha256']
        with np.load(entry['micro_schedule']) as schedule:
            mixtures[mode] = np.stack([(schedule['task'] == i).mean(1) for i in range(16)], axis=1)
        coefficients, current = [], np.zeros(16)
        for i, weights in enumerate(mixtures[mode]):
            current = .9*current + .1*weights
            coefficients.append(current/(1-.9**(i+1)))
        moment_coefficients[mode] = np.stack(coefficients)
        assert np.allclose(moment_coefficients[mode].sum(1), 1, rtol=0, atol=1e-14)
    results = {}
    for stage, row in diagnostic['stages'].items():
        gram = np.asarray(row['components']['all']['gram'], dtype=np.float64)
        output = {}
        for mode in design['arms']:
            center = moment_coefficients[mode]-1/16
            squared_errors = np.einsum('bi,ij,bj->b', center, gram, center)
            task_dots = moment_coefficients[mode]@gram
            output[mode] = {}
            for name, rows in [('all1600', slice(None)), ('last800', slice(800, None))]:
                output[mode][name] = {
                    'mean_first_moment_deviation_norm_sq': float(squared_errors[rows].mean()),
                    'deviation_over_global_mean_gradient_norm_sq': float(squared_errors[rows].mean()/gram.mean()),
                    'fraction_update_task_pairs_with_negative_inner_product': float((task_dots[rows] < 0).mean()),
                    'min_task_inner_product': float(task_dots[rows].min())}
        output['mixed_over_same_first_moment_deviation'] = {
            name: output['mixed_task_micro4'][name]['mean_first_moment_deviation_norm_sq']/output['same_task_micro4'][name]['mean_first_moment_deviation_norm_sq']
            for name in ['all1600', 'last800']}
        results[stage] = output
    result = {'unix': time.time(), 'code_sha256': sha(__file__),
              'input_sha256': {str(p): sha(p) for p in [diagnostic_path, design_path]},
              'beta1': .9, 'initial_first_moment': 'zero', 'bias_correction': True,
              'stages': results,
              'scope': 'Counterfactual: reuse each cached original512 task-mean gradient at every step, without changing parameters. Exact first-moment recursion for that surrogate and the actual1600-step task schedules. No within-task noise, full-pool gradients, Adam second moment, parameter motion, inference or fitting. This measures temporal filtering in the fixed-state surrogate; it is not observed Adam-update variance or an accuracy prediction.'}
    with (ROOT/'analysis/task_mixing_cached_momentum.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v['mixed_over_same_first_moment_deviation'] for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
