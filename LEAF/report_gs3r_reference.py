"""Compare the fixed official GS3R reference with existing bank0 results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_bilateral_merge import contrast
from run_streaming_subspace_merge import write_json


def main():
    torch.set_num_threads(4)
    root = Path('runs/gs3r_reference_v1')
    output = root / 'comparison.json'
    assert not output.exists()
    reference_plan = json.loads((root / 'plan.json').read_text())
    plan = json.loads(Path(reference_plan['arrival_plan']).read_text())
    tasks = plan['tasks']
    sources = {}

    def read(path):
        path = Path(path)
        record = json.loads(path.read_text())
        sources[str(path)] = sha256(path)
        assert record['status'] == 'completed'
        assert record['plan_sha256'] == sha256(Path(record['plan']))
        return record

    pure = read('runs/fisher_moment_v1/references/run.json')
    reports = {}
    for order in ('forward', 'reverse'):
        gs3r = read(root / order / 'run.json')
        expected_order = tasks if order == 'forward' else list(reversed(tasks))
        assert gs3r['order'] == expected_order and len(gs3r['stages']) == 16
        assert gs3r['updates'] == gs3r['training_examples'] == 0
        assert gs3r['training_arrays_read'] is False and gs3r['official_test_read'] is False
        assert [s['stage'] for s in gs3r['stages'] if 'validation' in s] == plan['validation_stages']
        for stage in gs3r['stages']:
            checkpoint = root / order / f"stage_{stage['stage']:02d}" / 'merged.ckpt'
            assert sha256(checkpoint) == stage['checkpoint_sha256']
        resume = torch.load(root / order / 'resume_state.pt', map_location='cpu', weights_only=True)
        assert resume['count'] == 16
        vector_bytes = sum(v.numel() * v.element_size() for v in resume['vector'].values())
        assert vector_bytes == gs3r['resume_vector_bytes']
        del resume
        runs = {'gs3r': gs3r}
        for condition, folder in [('incoming_rows', 'streaming_subspace_v1'),
                                  ('history_sketch', 'streaming_history_sketch_v1')]:
            runs[condition] = read(Path('runs') / folder / f'{condition}_{order}' / 'run.json')
            assert runs[condition]['order'] == expected_order
        for task in tasks:
            assert gs3r['dataset_audits'][task]['training_arrays_read'] is False
            sizes = [r['dataset_audits'][task]['validation_size'] for r in [*runs.values(), pure]]
            assert len(set(sizes)) == 1
        # The first expert is unchanged in both the reference and learned protocols.
        assert gs3r['stages'][0]['validation'] == runs['incoming_rows']['stages'][0]['validation']
        validation = {k: v['stages'][-1]['validation'] for k, v in runs.items()}
        validation.update({k: v['validation'] for k, v in pure['results'].items()})
        macro = {k: float(np.mean([v[t]['BAcc'] for t in tasks])) for k, v in validation.items()}
        reports[order] = {
            'macro_BAcc': macro,
            'gs3r_minus_reference': {k: contrast(validation['gs3r'], validation[k], tasks)
                                     for k in ('uniform', 'fisher', 'incoming_rows', 'history_sketch')},
            'gs3r_stage_macro_BAcc': {str(s['stage']): s['macro_BAcc'] for s in gs3r['stages']
                                     if 'macro_BAcc' in s},
            'gs3r_cost': {
                'elapsed_seconds': gs3r['elapsed_seconds'],
                'merging_seconds': sum(s['merging_seconds'] for s in gs3r['stages']),
                'peak_reserved_bytes': gs3r['peak_reserved_bytes'],
                'resume_vector_bytes': vector_bytes,
                'restoration_bytes': gs3r['restoration_bytes'],
                'coefficient_updates': 0, 'calibration_examples': 0}}
    result = {
        'sources': sources, 'orders': reports,
        'checks': {'gs3r_checkpoint_hashes': True, 'first_expert_validation_exact': True,
                   'validation_sizes_match': True, 'resume_vector_bytes_match': True},
        'scope': 'One expert bank and two correlated arrival orders on exposed development validation. '
                 'GS3R is the pinned official full-rank core without auxiliary optimization; nonmatrix '
                 'parameters use its recursive half averaging. Uniform is the full original-expert mean. '
                 'Fisher uses a separate predictive-Fisher calibration budget; incoming/history methods '
                 'use supervised replay. These are direct performance references, not equal-data or '
                 'equal-compute algorithmic ablations. No official-test results or novel-method claim. '
                 'Per-stage means cover only tasks seen by that stage. Wall times on shared hardware '
                 'are descriptive. Task-deletion ranges are sensitivity, not confidence intervals.'}
    write_json(output, result)
    labels = ['uniform', 'fisher', 'gs3r', 'incoming_rows', 'history_sketch']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    for j, order in enumerate(('forward', 'reverse')):
        offset = (j - .5) * .36
        values = [100 * reports[order]['macro_BAcc'][k] for k in labels]
        axes[0].bar(np.arange(len(labels)) + offset, values, width=.36, label=order)
        gains = reports[order]['gs3r_minus_reference']['uniform']['per_task_pp']
        axes[1].barh(np.arange(len(tasks)) + offset, [gains[t] for t in tasks], height=.36, label=order)
    axes[0].set_xticks(np.arange(len(labels)), ['Uniform', 'Fisher', 'GS3R', 'Incoming\nrows', 'History\nsketch'])
    axes[0].set_ylabel('Final macro balanced accuracy (%)')
    axes[0].legend()
    axes[1].set_yticks(np.arange(len(tasks)), [t.replace('MI_', '').replace('EMO_', '') for t in tasks], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', lw=.8)
    axes[1].set_xlabel('GS3R minus uniform (percentage points)')
    axes[1].legend()
    fig.suptitle('Bank0 references — calibration and compute budgets differ')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    print(json.dumps({order: r['macro_BAcc'] for order, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
