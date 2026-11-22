"""Measure the supervised correction in existing completed streaming runs.

This reads native checkpoints only. It does not read datasets, evaluate a
model, optimize coefficients, or claim that a new memory policy will improve.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    root = Path('runs/streaming_unit_feedback_v1')
    output = root / 'calibration_correction_geometry.json'
    assert not output.exists()
    sources, runs = {}, {}
    for seed in (0, 2):
        plan = json.loads((root / f'seed{seed}/plan.json').read_text())
        source = json.loads(Path(plan['source_fit']).read_text())
        base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
        for order in ('forward', 'reverse'):
            folder = root / f'seed{seed}/unit_feedback_{order}'
            path = folder / 'run.json'
            run = json.loads(path.read_text())
            assert run['status'] == 'completed'
            sources[str(path)] = digest(path)
            previous, records = None, []
            for count, stage in enumerate(run['stages']):
                path = folder / f'stage_{count + 1:02d}/merged.ckpt'
                assert digest(path) == stage['checkpoint_sha256']
                current = torch.load(path, map_location='cpu', weights_only=True)
                if count == 0:
                    previous = current
                    continue
                path = Path(source['experts'][stage['arriving_task']])
                assert digest(path) == plan['expert_sha256'][stage['arriving_task']]
                incoming = torch.load(path, map_location='cpu', weights_only=True)
                energies = {'previous': 0., 'incoming': 0., 'current': 0., 'correction': 0., 'correction_outside_old_new': 0.}
                row_ratios, significant_rows, row_count = [], 0, 0
                for name in plan['active']:
                    shape = base[name].shape
                    rows = shape[0] if shape else 1
                    a = (previous[name] - base[name]).double().reshape(rows, -1)
                    b = (incoming[name] - base[name]).double().reshape(rows, -1)
                    m = (current[name] - base[name]).double().reshape(rows, -1)
                    correction = m - (count * a + b) / (count + 1)
                    an, bn, cn = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True), correction.norm(dim=1, keepdim=True)
                    qa = a / an.clamp_min(1e-300)
                    rb = b - (b * qa).sum(1, keepdim=True) * qa
                    rn = rb.norm(dim=1, keepdim=True)
                    qb = torch.where(rn > 1e-6 * bn, rb / rn.clamp_min(1e-300), 0.)
                    residual = correction.clone()
                    for _ in range(2):
                        for q in (qa, qb):
                            residual -= (residual * q).sum(1, keepdim=True) * q
                    for key, value in [('previous', a), ('incoming', b), ('current', m), ('correction', correction), ('correction_outside_old_new', residual)]:
                        energies[key] += float(value.square().sum())
                    rounding_floor = 8 * torch.finfo(torch.float32).eps * sum(x[name].double().reshape(rows, -1).norm(dim=1, keepdim=True) for x in (base, previous, incoming, current))
                    significant_rows += int((cn > rounding_floor).sum())
                    row_count += rows
                    row_ratios.append((cn / bn.clamp_min(1e-300))[bn > 0].flatten())
                ratios = torch.cat(row_ratios)
                records.append({'stage': count + 1, 'task': stage['arriving_task'], 'squared_norms': energies,
                                'correction_norm_over_incoming': float(np.sqrt(energies['correction'] / energies['incoming'])),
                                'correction_outside_old_new_energy_fraction': energies['correction_outside_old_new'] / energies['correction'],
                                'per_row_correction_over_nonzero_incoming_norm_q10_q50_q90': torch.quantile(ratios, torch.tensor([.1, .5, .9], dtype=ratios.dtype)).tolist(),
                                'rows_above_rounding_floor': significant_rows, 'total_rows': row_count})
                previous = current
            runs[f'seed{seed}_{order}'] = records
            print('checked', seed, order, flush=True)
    result = {'sources': sources, 'runs': runs, 'optimizer_updates': 0, 'extra_model_forwards_or_backwards': 0,
              'datasets_read': False, 'elapsed_seconds': time.monotonic() - started,
              'formula': 'delta_current - (count*delta_previous + delta_incoming)/(count+1), with native FP32 deltas then FP64 geometry.',
              'scope': 'Descriptive geometry of the completed unit-feedback trajectory, selected after its unstable performance. Outside-plane energy measures representational contribution, not causal usefulness or a bound on achievable accuracy. Small rounding components are possible; a scale-relative rounding-floor row count is reported. No new method or performance result is produced.'}
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
