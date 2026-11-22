"""Report the frozen seed0 tokenizer-coordinate intervention and controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-fp32', action='store_true')
    args = parser.parse_args()
    root = Path('runs/paper_merge_v2')
    old = Path('runs/aligned_merge_v1/seed0')
    bank = json.loads((root/'seed0/tokenizer_canonical_experts/run.json').read_text())
    plan = json.loads((root/'inputs/tokenizer_pilot_plan.json').read_text())
    paths = {
        'original_selected': old/'tensor_balanced_ce',
        'original_average': old/'tensor_balanced_avg',
        'canonical_selected': root/'seed0/tokenizer_canonical_tensor_balanced_ce',
        'canonical_average': root/'seed0/tokenizer_canonical_tensor_balanced_avg',
    }
    if args.original_fp32:
        paths['original_selected'] = root/'seed0/original_fp32_tensor_balanced_ce'
        paths['original_average'] = root/'seed0/original_fp32_tensor_balanced_avg'
    runs = {name: json.loads((path/'run.json').read_text()) for name, path in paths.items()}
    if args.original_fp32:
        for key in ['precision', 'optimizer', 'lr', 'batch_size', 'seed', 'alpha_updates_actual', 'ce_class_weights']:
            assert runs['original_selected'][key] == runs['canonical_selected'][key], key
    tasks = runs['original_selected']['datasets']
    reference_ids = np.load(paths['original_selected']/'calibration_indices.npz')
    for name, path in paths.items():
        ids = np.load(path/'calibration_indices.npz')
        assert all(np.array_equal(ids[t], reference_ids[t]) for t in tasks)
        assert runs[name]['datasets'] == tasks
        assert runs[name]['base_checkpoint_sha256'] == runs['original_selected']['base_checkpoint_sha256']
        assert runs[name]['class_balanced_ce']
    assert len(bank['records']) == len(tasks) == 16
    avg = runs['canonical_average']
    assert avg['averaging_steps'] == [800, 960, 1120, 1280, 1440, 1600]
    assert avg['coefficient_mean_exact'] and avg['fixed_base_buffers_and_inactive_exact']
    cells = {name: {'validation_macro_bacc': run['folded_val_macro_BAcc'],
                    'best_step': run['best_step'], 'source': str(paths[name])}
             for name, run in runs.items()}
    contrasts = {}
    for name in ['selected', 'average']:
        original, canonical = runs[f'original_{name}'], runs[f'canonical_{name}']
        delta = {t: 100*(canonical['folded_val'][t]['BAcc']-original['folded_val'][t]['BAcc'])
                 for t in tasks}
        contrasts[name] = {
            'macro_gain_pp': float(np.mean(list(delta.values()))),
            'per_task_gain_pp': delta,
            'negative_task_count': sum(v < 0 for v in delta.values()),
            'worst_task': min(delta, key=delta.get),
        }
    result = {
        'training_precision_matched': runs['original_selected']['precision']==runs['canonical_selected']['precision'],
        'validation_only': True, 'seed': 0, 'test_inference_performed': False,
        'cells': cells, 'paired_contrasts': contrasts,
        'expert_function_checks': {
            'count': len(bank['records']),
            'max_logit_error': max(v['function_check']['max_abs_logit_error'] for v in bank['records'].values()),
            'max_relative_kernel_error': max(v['geometry']['relative_kernel_error_after_fp32_export'] for v in bank['records'].values()),
        },
        'average_export_max_logit_error': max(avg['export_logit_max_abs'].values()),
        'calibration_identity_exact': True,
        'effective_coefficients_note': plan['effective_coefficients_note'],
        'expert_bank_seconds': bank['seconds'],
        'stage_costs': [json.loads(line) for line in (root/'inputs/tokenizer_pilot_events.jsonl').read_text().splitlines()],
        'interpretation': 'Exact expert function preservation changes the merged search space and its optimization geometry. '
                          'Any effect cannot be assigned solely to factor cross-terms without further controls. '
                          'Operator/gauge-invariant merging is existing prior art, including SLOA and GeoMerge. '
                          'One exposed validation seed is not independent confirmation.',
    }
    filename = 'tokenizer_operator_summary_fp32_control.json' if args.original_fp32 else 'tokenizer_operator_summary.json'
    (root/'analysis'/filename).write_text(json.dumps(result, indent=2)+'\n')
    if args.original_fp32:
        print(json.dumps({'cells': cells, 'contrasts': contrasts}, indent=2))
        return
    lines = ['# Tokenizer operator-coordinate pilot', '',
             'All16 original expert functions pass real-data preservation checks. The base, '
             'architecture, calibration IDs, optimizer budget, and average window are fixed.', '',
             '| Coordinates | Selected validation BAcc | Fixed-average validation BAcc |',
             '| --- | ---: | ---: |']
    for name in ['original', 'canonical']:
        lines.append(f'| {name} | {cells[name+"_selected"]["validation_macro_bacc"]*100:.4f}% | '
                     f'{cells[name+"_average"]["validation_macro_bacc"]*100:.4f}% |')
    lines += ['', f'Canonical-minus-original: selected {contrasts["selected"]["macro_gain_pp"]:+.4f} pp; '
              f'fixed average {contrasts["average"]["macro_gain_pp"]:+.4f} pp.', '',
              result['interpretation'], '', 'Full per-task deltas, export checks, and costs are in '
              '`tokenizer_operator_summary.json`. No official test was performed.']
    (root/'analysis/tokenizer_operator_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cells': cells, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
