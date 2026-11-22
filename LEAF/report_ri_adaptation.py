"""Summarize fixed RI expert adaptation without treating drift as merge success."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import classification_metrics


def mean(values):
    return float(np.mean(list(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path('runs/paper_merge_v2')
    plan = json.loads((root/'inputs/ri_adaptation_plan.json').read_text())
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    tasks = source['datasets']
    completed = [t for t in tasks if (root/'seed0/ri_experts'/t/'run.json').exists()]
    if not args.partial:
        assert len(completed) == len(tasks) == 16
    base_metrics = {}
    for task in tasks:
        cache = torch.load(root/'seed0/ri_cache'/f'{task}.pt', map_location='cpu', weights_only=True)
        n = cache['metadata']['n_calibration']
        base_metrics[task] = {
            split: classification_metrics(cache['base_logits'][idx], cache['truth'][idx])
            for split, idx in [('calibration', slice(0, n)), ('validation', slice(n, None))]
        }
    records, pairs = {}, []
    for own in completed:
        r = json.loads((root/'seed0/ri_experts'/own/'run.json').read_text())['ri_adaptation']
        assert r['updates'] == plan['steps_per_expert'] and not r['engineering_smoke']
        assert r['fixed_buffers_and_inactive_exact']
        own_rows = {}
        for split in ['calibration', 'validation']:
            before, after = r['before'][own][split], r['after'][own][split]
            own_rows[split] = {'before_bacc': before['BAcc'], 'after_bacc': after['BAcc'],
                               'gain_pp': 100*(after['BAcc']-before['BAcc']),
                               'after_kl_from_original': after['kl_from_target_domain_expert']}
        records[own] = {'own': own_rows, 'seconds': r['seconds'],
                        'other_domain_update_counts': r['other_domain_update_counts'],
                        'max_export_error': max(v['max_abs_logit_error'] for v in r['export_checks'].values())}
        for target in tasks:
            if target == own:
                continue
            row = {'expert': own, 'target': target, 'splits': {}}
            for split in ['calibration', 'validation']:
                before, after = r['before'][target][split], r['after'][target][split]
                base = base_metrics[target][split]['BAcc']
                row['splits'][split] = {
                    'before_gain_vs_base_pp': 100*(before['BAcc']-base),
                    'after_gain_vs_base_pp': 100*(after['BAcc']-base),
                    'adaptation_bacc_change_pp': 100*(after['BAcc']-before['BAcc']),
                    'before_kl_from_base': before['kl_from_base'],
                    'after_kl_from_base': after['kl_from_base'],
                }
            # Category is set by pre-adaptation calibration only, never after/validation.
            gain = row['splits']['calibration']['before_gain_vs_base_pp']
            row['calibration_category_before_adaptation'] = 'helpful' if gain > 0 else 'harmful' if gain < 0 else 'neutral'
            pairs.append(row)
    summary = {'completed_experts': completed, 'expected_experts': tasks,
               'partial': len(completed) != len(tasks), 'pairs': pairs, 'per_expert': records,
               'splits': {}, 'test_inference_performed': False,
               'metric_scope': 'Original calibration pools and fixed up-to512 stratified validation probes per domain. Not full validation or independent evaluation.',
               'interpretation': 'Own retention and off-domain localization are intermediate outcomes. They do not prove better merged accuracy or a new method. Pair categories are defined on pre-adaptation calibration gains. Pairs are statistically dependent.'}
    if completed:
        for split in ['calibration', 'validation']:
            summary['splits'][split] = {
                'own_macro_bacc_change_pp': mean(r['own'][split]['gain_pp'] for r in records.values()),
                'other_macro_kl_before': mean(row['splits'][split]['before_kl_from_base'] for row in pairs),
                'other_macro_kl_after': mean(row['splits'][split]['after_kl_from_base'] for row in pairs),
                'other_macro_bacc_change_pp': mean(row['splits'][split]['adaptation_bacc_change_pp'] for row in pairs),
                'categories_defined_on_calibration': {},
            }
            for category in ['helpful', 'harmful', 'neutral']:
                rows = [p['splits'][split] for p in pairs if p['calibration_category_before_adaptation'] == category]
                summary['splits'][split]['categories_defined_on_calibration'][category] = {
                    'pair_count': len(rows),
                    **({key: mean(row[key] for row in rows) for key in [
                        'before_gain_vs_base_pp', 'after_gain_vs_base_pp', 'adaptation_bacc_change_pp']}
                       if rows else {}),
                }
    suffix = 'progress' if args.partial else 'summary'
    (root/f'analysis/ri_adaptation_{suffix}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'completed_experts': completed, 'splits': summary['splits']}, indent=2))


if __name__ == '__main__':
    main()
