"""Training-only engineering for the two correction-propagation rules."""
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from check_merge_export_margins import compare
from run_streaming_correction_retention import reload_checks, train_stage
from run_streaming_subspace_merge import TaskData, write_json
from streaming_correction_retention import add_original, initialize


def main():
    root = Path('runs/streaming_correction_retention_v1/engineering_v2')
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    plan = json.loads(Path('runs/streaming_subspace_v1/plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4, learning_rate=1e-5)
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    tasks = plan['tasks'][:3]
    experts = []
    for task in tasks:
        data.arrive(task)
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        experts.append(torch.load(path, map_location='cpu', weights_only=True))
    active = plan['active']
    original_sum = {}
    add_original(original_sum, experts[0], active)
    a = initialize(experts[0], experts[1], active, 1, 'recursive')
    b = initialize(experts[0], experts[1], active, 1, 'retain_correction', original_sum)
    assert all(torch.equal(a[n], b[n]) for n in base)
    del a, b
    records, stage2_states = [], {}
    for condition in ('recursive', 'retain_correction'):
        previous = experts[0]
        history = {}
        add_original(history, experts[0], active)
        for count in (1, 2):
            initial = initialize(previous, experts[count], active, count, condition, history)
            if condition == 'retain_correction' and count == 2:
                # Independent formula: theta_previous + (expert - original_mean) / t.
                error = max(float((initial[n] - (previous[n].double() +
                            (experts[count][n].double() - history[n] / count) / (count + 1)).float()).abs().max())
                            for n in active)
                assert error <= 1e-7
            state, snapshots, probes, record = train_stage(leaf, initial, data, tasks[:count + 1],
                                                          plan, count + 1, device)
            assert len(snapshots) == 2
            assert all(torch.equal(state[n], ((snapshots[0][n].double() + snapshots[1][n].double()) / 2).float())
                       for n in active)
            assert all(torch.equal(state[n], base[n]) for n in base if n not in active)
            checkpoint = root / f'engineering_{condition}_stage{count + 1:02d}.ckpt'
            torch.save(state, checkpoint)
            record['export_checks'] = reload_checks(leaf, checkpoint, probes, data, tasks[:count + 1], 4, device)
            record.update(condition=condition, stage=count + 1, checkpoint_sha256=sha256(checkpoint))
            records.append(record)
            if count == 1:
                stage2_states[condition] = state
            add_original(history, experts[count], active)
            assert all(torch.equal(history[n], sum(e[n].double() for e in experts[:count + 1])) for n in active)
            previous = state
            del initial, snapshots, probes
    parameter_error = max(float((stage2_states['recursive'][n] - stage2_states['retain_correction'][n]).abs().max())
                          for n in active)
    assert parameter_error <= 1e-6
    replay_logits = {}
    with torch.no_grad():
        for condition, state in stage2_states.items():
            leaf.load_state_dict(state, strict=True)
            replay_logits[condition] = {
                t: PrototypeClassifier(leaf, *data.readouts[t]).eval()(data.train[t][0][:4].to(device)).cpu()
                for t in tasks[:2]}
    repeatability = {t: compare(replay_logits['recursive'][t], replay_logits['retain_correction'][t],
                                data.train[t][1][:4]) for t in tasks[:2]}
    assert all(v['max_abs_logit_error'] <= 1e-5 and v['changed_predictions'] == 0 for v in repeatability.values())
    result = {'passed': True, 'records': records, 'native_training_updates': sum(r['updates'] for r in records),
              'native_training_examples': sum(r['training_examples'] for r in records),
              'first_merge_initialization_identical': True,
              'first_calibrated_parameter_repeatability_max_abs_error': parameter_error,
              'first_calibrated_native_logit_repeatability': repeatability,
              'independent_retention_formula_max_abs_error': error,
              'fixed_means_exact': True, 'original_sum_contains_only_experts': True,
              'original_sum_bytes': sum(v.numel() * v.element_size() for v in history.values()),
              'elapsed_seconds': time.monotonic() - started,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'source_sha256': {p: sha256(Path(p)) for p in ['streaming_correction_retention.py',
                  'run_streaming_correction_retention.py', 'check_streaming_correction_retention.py']},
              'scope': 'Engineering only: three original experts, two two-round native CE stages per arm. '
                       'Only the original training-calibration arrays were opened. No validation or '
                       'official-test access, no performance selection, and no novelty claim.'}
    write_json(root / 'engineering.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
