"""Algebraic norm/direction decomposition of saved input-Jacobian measurements."""
import json
import math
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    source_path = root/'analysis/input_response_summary.json'
    source = json.loads(source_path.read_text())
    rows, aggregate = {}, {}
    for pool in ['original', 'replacement']:
        rows[pool], aggregate[pool] = {}, {}
        for kind in ['raw', 'support', 'covariance']:
            values = {}
            for task in source['datasets']:
                models = source['records'][task]['splits'][pool]['models']
                ce, kl = [models[m]['jacobians'][kind] for m in ['ce', 'kl']]
                assert ce['teacher_energy'] == kl['teacher_energy']
                a = ce['teacher_energy']
                delta_error = kl['mismatch_energy']-ce['mismatch_energy']
                delta_norm = kl['student_energy']-ce['student_energy']
                delta_cross = -2*(kl['inner_product']-ce['inner_product'])
                assert math.isclose(delta_error, delta_norm+delta_cross, rel_tol=1e-10, abs_tol=1e-12)
                for m in [ce, kl]:
                    r, c = m['student_teacher_norm_ratio'], m['cosine']
                    assert math.isclose(m['relative_mismatch_norm']**2, 1+r*r-2*r*c, rel_tol=1e-10, abs_tol=1e-12)
                values[task] = {'teacher_energy': a, 'ce_student_energy': ce['student_energy'],
                                'kl_student_energy': kl['student_energy'],
                                'kl_minus_ce_error_energy': delta_error,
                                'student_norm_energy_contribution': delta_norm,
                                'cross_inner_product_contribution': delta_cross,
                                'norm_fraction_of_absolute_contributions': abs(delta_norm)/(abs(delta_norm)+abs(delta_cross)),
                                'ce_cosine': ce['cosine'], 'kl_cosine': kl['cosine'],
                                'teacher_top10percent_energy_fraction': ce['per_example_description']['teacher_energy_top10percent_examples_fraction'],
                                'teacher_largest_example_energy_fraction': ce['per_example_description']['teacher_energy_largest_example_fraction']}
            rows[pool][kind] = values
            v = list(values.values())
            aggregate[pool][kind] = {
                'tasks_with_decreased_error': sum(x['kl_minus_ce_error_energy'] < 0 for x in v),
                'tasks_with_decreased_student_norm_energy': sum(x['student_norm_energy_contribution'] < 0 for x in v),
                'tasks_with_improved_cosine': sum(x['kl_cosine'] > x['ce_cosine'] for x in v),
                'median_norm_fraction_of_absolute_contributions': float(np.median([x['norm_fraction_of_absolute_contributions'] for x in v])),
                'minimum_norm_fraction_of_absolute_contributions': min(x['norm_fraction_of_absolute_contributions'] for x in v),
                'median_teacher_top10percent_energy_fraction': float(np.median([x['teacher_top10percent_energy_fraction'] for x in v])),
                'median_teacher_largest_example_energy_fraction': float(np.median([x['teacher_largest_example_energy_fraction'] for x in v])),
                'teacher_energy_range': [min(x['teacher_energy'] for x in v), max(x['teacher_energy'] for x in v)]}
    target = root/'analysis/input_response_geometry.json'
    result = {'source': str(source_path), 'source_sha256': sha256(source_path),
              'code_sha256': sha256(Path(__file__)), 'records': rows, 'aggregates': aggregate,
              'identity': 'Eerror = Ateacher + Bstudent - 2Dcross; Eerror/A = 1+r^2-2*r*cosine.',
              'scope': 'Descriptive identity, not a causal intervention. Cross-inner-product changes combine norm and direction. Tiny teacher denominators and row concentration make raw relative mismatch a poor cross-task performance surrogate. No new inference or candidate.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
