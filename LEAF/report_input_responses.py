"""Summarize every frozen input-response diagnostic arm without new inference."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from diagnose_input_responses_v3 import summarize_energy


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    assert len(a) == 16 and np.isfinite(a).all()
    loo = (a.sum()-a)/15
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'minimum': float(a.min()), 'maximum': float(a.max()),
            'positive_tasks': int((a > 0).sum()), 'negative_tasks': int((a < 0).sum()),
            'zero_tasks': int((a == 0).sum()),
            'leave_one_task_out_mean_range': [float(loo.min()), float(loo.max())]}


def compare_all(actual, reference):
    assert actual.keys() == reference.keys()
    for key in actual:
        assert math.isclose(actual[key], reference[key], rel_tol=1e-11, abs_tol=1e-14), (key, actual[key], reference[key])


def row_distribution(energies, weights):
    e = energies.double().numpy()
    w = weights.numpy()
    valid = e[:, 0] > 0
    relative = np.sqrt(e[valid, 3]/e[valid, 0])
    result = {'positive_teacher_energy_examples': int(valid.sum()),
              'relative_mismatch_quantiles': dict(zip(['p10', 'p25', 'p50', 'p75', 'p90'],
                                                     np.quantile(relative, [.1, .25, .5, .75, .9]).tolist()))}
    for col, label in [(0, 'teacher'), (1, 'student'), (3, 'mismatch')]:
        ordered = np.sort(w*e[:, col])[::-1]
        total = ordered.sum()
        result[label+'_energy_largest_example_fraction'] = float(ordered[0]/total)
        result[label+'_energy_top10percent_examples_fraction'] = float(ordered[:math.ceil(len(e)/10)].sum()/total)
    return result


def main():
    root = Path('runs/paper_merge_v2')
    design_path = root/'inputs/input_response_v3_design.json'
    design = json.loads(design_path.read_text())
    tasks = design['all_tasks']
    assert len(tasks) == len(set(tasks)) == 16
    records, provenance, checks = {}, {}, {}
    forwards, backwards, seconds = 0, 0, []
    for shard in [0, 1]:
        path = root/f'seed0/input_response_v3_shard{shard}/run.json'
        run = json.loads(path.read_text())
        plan_path = root/f'inputs/input_response_v3_shard{shard}_plan.json'
        assert not run['engineering_smoke'] and run['all_model_states_unchanged']
        assert run['optimizer_updates'] == 0 and not run['validation_inference_performed']
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert run['plan_sha256'] == sha256(plan_path)
        assert run['plan']['design_sha256'] == sha256(design_path)
        assert run['code_sha256'] == sha256(Path('diagnose_input_responses_v3.py'))
        for name, digest in {**run['plan']['source_sha256'], **run['plan']['input_sha256']}.items():
            assert sha256(Path(name)) == digest, name
        assert run['datasets'] == tasks[shard::2]
        assert not set(records).intersection(run['records'])
        records.update(run['records'])
        provenance[str(path)] = sha256(path)
        forwards += run['main_forward_examples']
        backwards += run['backward_example_contrasts']
        seconds.append(run['seconds'])
    output = {}
    for task in tasks:
        record = records[task]
        assert sha256(Path(record['path'])) == record['sha256']
        assert record['teacher_state_unchanged']
        for check in record['old512_native_replay'].values():
            assert check['max_abs_logit_error'] < 2e-5
            assert all(row['margin_within_twice_error'] for row in check['changed_rows'])
        for rows in record['native_vs_gradient_execution'].values():
            for check in rows:
                assert check['max_abs_logit_error'] < 2e-5
                assert all(row['margin_within_twice_error'] for row in check['changed_rows'])
        data = torch.load(record['path'], map_location='cpu', weights_only=True)
        n = data['n_original']
        assert n == 128 and len(data['truth']) == 256
        assert torch.equal(data['truth'][:n], data['truth'][n:])
        output[task] = {'sensor_support_rank': record['sensor_support_rank'],
                        'selected_pool_overlap': record['selected_pool_overlap'],
                        'splits': record['splits']}
        for pool, select in [('original', slice(0, n)), ('replacement', slice(n, 2*n))]:
            truth = data['truth'][select]
            counts = torch.bincount(truth)
            weights = (n/(len(counts)*counts.double()))[truth]
            for label in ['initial', 'ce', 'kl']:
                for kind in ['raw', 'support', 'covariance']:
                    energies = data['jacobian_energies'][label][kind][select]
                    measured = summarize_energy(energies, weights)
                    reference = output[task]['splits'][pool]['models'][label]['jacobians'][kind]
                    compare_all(measured, reference)
                    reference['per_example_description'] = row_distribution(energies, weights)
                agree = data['logits'][label][select].argmax(-1) == data['logits']['teacher'][select].argmax(-1)
                output[task]['splits'][pool]['models'][label]['agreeing_examples'] = int(agree.sum())
                if agree.any():
                    conditional = summarize_energy(data['jacobian_energies'][label]['support'][select][agree], weights[agree])
                    compare_all(conditional, output[task]['splits'][pool]['models'][label]['support_jacobian_on_agreeing_examples'])
        checks[task] = {'native_replay_max_error': max(r['max_abs_logit_error'] for r in record['old512_native_replay'].values()),
                         'native_autograd_max_error': max(r['max_abs_logit_error'] for rows in record['native_vs_gradient_execution'].values() for r in rows),
                         'all_saved_energy_aggregates_recomputed': True, 'teacher_state_unchanged': True}
    aggregates, contrasts = {}, {}
    for pool in ['original', 'replacement']:
        aggregates[pool], contrasts[pool] = {}, {}
        for label in ['initial', 'ce', 'kl']:
            rows = [output[t]['splits'][pool]['models'][label] for t in tasks]
            aggregates[pool][label] = {key: summary([r[key] for r in rows]) for key in
                                      ['balanced_centered_logit_mse', 'balanced_teacher_kl', 'balanced_teacher_top1_agreement']}
            aggregates[pool][label]['jacobians'] = {
                kind: {key: summary([r['jacobians'][kind][key] for r in rows]) for key in
                       ['relative_mismatch_norm', 'student_teacher_norm_ratio', 'cosine']}
                for kind in ['raw', 'support', 'covariance']}
            conditional = [r['support_jacobian_on_agreeing_examples']['relative_mismatch_norm'] for r in rows]
            aggregates[pool][label]['support_mismatch_on_agreeing_examples'] = summary(conditional)
        for after, before in [('ce', 'initial'), ('kl', 'initial'), ('kl', 'ce')]:
            paired = {}
            for kind in ['raw', 'support', 'covariance']:
                a = [output[t]['splits'][pool]['models'][after]['jacobians'][kind] for t in tasks]
                b = [output[t]['splits'][pool]['models'][before]['jacobians'][kind] for t in tasks]
                paired[kind] = {key+'_difference': summary([x[key]-y[key] for x, y in zip(a, b)]) for key in
                                ['relative_mismatch_norm', 'student_teacher_norm_ratio', 'cosine']}
                log_ratio = summary([math.log(x['relative_mismatch_norm']/y['relative_mismatch_norm']) for x, y in zip(a, b)])
                paired[kind]['log_mismatch_ratio'] = log_ratio
                paired[kind]['geometric_mean_mismatch_ratio'] = math.exp(log_ratio['mean'])
            paired['zero_order'] = {key+'_difference': summary([
                output[t]['splits'][pool]['models'][after][key]-output[t]['splits'][pool]['models'][before][key]
                for t in tasks]) for key in ['balanced_centered_logit_mse', 'balanced_teacher_kl', 'balanced_teacher_top1_agreement']}
            contrasts[pool][after+'_minus_'+before] = paired
    figure_dir = root/'analysis/figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey=True)
    names = [t.removeprefix('MI_').removeprefix('EMO_').replace('_seg4', '').replace('BCIC_', '') for t in tasks]
    colors = {'initial': '#666666', 'ce': '#0072b2', 'kl': '#d55e00'}
    for i, pool in enumerate(['original', 'replacement']):
        for j, kind in enumerate(['raw', 'support', 'covariance']):
            ax = axes[i, j]
            for label in ['initial', 'ce', 'kl']:
                y = [output[t]['splits'][pool]['models'][label]['jacobians'][kind]['relative_mismatch_norm'] for t in tasks]
                ax.plot(range(16), y, marker='o', markersize=3, linewidth=.8, label=label, color=colors[label])
            ax.set_yscale('log')
            ax.axhline(1, color='#aaaaaa', lw=.7, ls='--')
            ax.set_title(pool+' / '+kind)
            ax.set_xticks(range(16), names, rotation=65, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=.2)
            if j == 0:
                ax.set_ylabel('Relative Jacobian mismatch (log scale)')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Frozen models versus own experts: 128 training rows per pool and task\nAll centered-logit contrasts; local derivatives, not finite-noise or causal evidence', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    figures = {}
    for ext in ['png', 'pdf']:
        p = figure_dir/f'input_response_diagnostic.{ext}'
        assert not p.exists()
        fig.savefig(p, dpi=160)
        figures[str(p)] = sha256(p)
    result = {'reporter_sha256': sha256(Path(__file__)), 'design': str(design_path),
              'design_sha256': sha256(design_path), 'source_runs': provenance, 'datasets': tasks,
              'records': output, 'aggregates': aggregates, 'paired_contrasts': contrasts,
              'execution_checks': checks, 'main_forward_examples': forwards,
              'backward_example_contrasts': backwards, 'worker_seconds': seconds,
              'figures': figures, 'new_inference_performed': False,
              'scope': 'Training-only one-bank descriptive diagnostic. Ratio/log-ratio/median and row-concentration summaries do not provide independent replication or a causal mechanism. Conditional agreement subsets differ across models; compare them descriptively only. All tasks and three channel metrics retained.'}
    target = root/'analysis/input_response_summary.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target))
    for pool in ['original', 'replacement']:
        for pair in contrasts[pool]:
            r = contrasts[pool][pair]['support']
            print(pool, pair, 'support relative mismatch difference', r['relative_mismatch_norm_difference'],
                  'geometric ratio', r['geometric_mean_mismatch_ratio'])


if __name__ == '__main__':
    main()
