"""Geometry and training-only native checks for bounded projected merging."""
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from run_streaming_correction_retention import reload_checks
from run_streaming_projected_sketch import train_stage
from run_streaming_subspace_merge import TaskData, write_json
from streaming_correction_retention import initialize
from streaming_projected_sketch import RANK, build_bases, orthonormal_rows, update_sketch


def main():
    root = Path('runs/streaming_projected_sketch_v1')
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    generator = torch.Generator().manual_seed(27149)
    projector_errors = []
    for width in (1, 2, 7):
        directions = torch.randn(3, 4, width, generator=generator, dtype=torch.float64)
        directions[0, 1] = directions[0, 0]
        directions[1, 3] = 0
        q = orthonormal_rows(directions).double()
        independent = torch.linalg.pinv(directions, rtol=1e-6) @ directions
        error = float((q.transpose(1, 2) @ q - independent).abs().max())
        assert error <= 1e-6
        projector_errors.append(error)
    plan = json.loads(Path('runs/streaming_correction_retention_v1/plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4)
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    tasks, experts = plan['tasks'][:3], []
    for task in tasks:
        data.arrive(task)
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        experts.append(torch.load(path, map_location='cpu', weights_only=True))
    active, records, histories = plan['active'], [], {}
    for condition in ('bilateral', 'history_sketch', 'random_sketch'):
        history = {} if condition == 'bilateral' else {
            n: torch.zeros((RANK, *base[n].shape), dtype=torch.float32) for n in active}
        if history:
            update_sketch(history, experts[0], base, active)
        previous = experts[0]
        for count in (1, 2):
            out = root / f'{condition}_stage{count + 1:02d}'
            out.mkdir()
            initial = initialize(previous, experts[count], active, count, 'recursive')
            bases, audit = build_bases(base, previous, experts[count], history, count, active, condition)
            state, snapshots, probes, record = train_stage(leaf, initial, bases, data, tasks[:count + 1],
                                                          plan, count + 1, device, out)
            assert len(snapshots) == 2
            assert all(torch.equal(state[n], ((snapshots[0][n].double() + snapshots[1][n].double()) / 2).float())
                       for n in active)
            assert all(torch.equal(state[n], base[n]) for n in base if n not in active)
            assert record['projection_statistics']['max_actual_step_norm_relative_error'] < .005
            checkpoint = out / 'merged.ckpt'
            torch.save(state, checkpoint)
            record['export_checks'] = reload_checks(leaf, checkpoint, probes, data, tasks[:count + 1], 4, device)
            record.update(condition=condition, stage=count + 1, basis_audit=audit,
                          checkpoint_sha256=sha256(checkpoint))
            records.append(record)
            if history:
                update_sketch(history, experts[count], base, active)
            previous = state
            del initial, bases, snapshots, probes
        if history:
            histories[condition] = history
    assert all(torch.equal(histories['history_sketch'][n], histories['random_sketch'][n]) for n in active)
    result = {'passed': True, 'records': records, 'native_training_updates': sum(r['updates'] for r in records),
              'native_training_examples': sum(r['training_examples'] for r in records),
              'independent_projector_max_abs_error': max(projector_errors),
              'real_and_random_history_identical': True, 'fixed_parameter_means_exact': True,
              'elapsed_seconds': time.monotonic() - started,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'source_sha256': {p: sha256(Path(p)) for p in ['streaming_projected_sketch.py',
                  'run_streaming_projected_sketch.py', 'check_streaming_projected_sketch.py', 'row_projected_adam.py']},
              'scope': 'Three original experts, two two-round stages per arm, three projected spaces. '
                       'Only training-calibration arrays opened; no validation or official-test data. '
                       'Independent pseudoinverse checks include zero and dependent directions. '
                       'Native Adam/projection and history sketching are pre-existing components, not novelty claims.'}
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
