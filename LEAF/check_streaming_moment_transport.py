"""Check strict integration, coefficient transfer, moments, and native exports."""
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from fisher_moment_merge import add_moments, moment_centers
from run_streaming_moment_transport import save_tensor_file
from run_streaming_strict_history import TaskData, strict_settings, write_json
from streaming_moment_transport import build_merger
from streaming_moment_transport_training import train_stage


def main():
    root = Path('runs/streaming_moment_transport_v1')
    assert not (root / 'engineering.json').exists()
    storage = Path('/var/tmp/jingzhou-bfm-merge-artifacts/streaming_moment_transport_v1/engineering')
    storage.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    strict_settings()
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    plan = json.loads(Path('runs/streaming_column_residual_v1/plan.json').read_text())
    plan.update(rounds=2, average_rounds=[1, 2], batch_size=4,
                per_arm_coefficients={'moment_restart': 347666, 'moment_transport': 347666})
    source = json.loads(Path(plan['source_fit']).read_text())
    cache = json.loads(Path('runs/fisher_moment_v1/fisher/run.json').read_text())
    assert cache['status'] == 'completed'
    tasks = ['MI_OpenBMI', 'EMO_FACED', 'ADHD_AliMotie', 'Workload']
    data = TaskData(plan, source, device)
    for task in tasks:
        data.arrive(task)
        assert sha256(Path(source['experts'][task])) == plan['expert_sha256'][task]
        assert cache['experts'][task]['expert_sha256'] == plan['expert_sha256'][task]
        for key, value in cache['experts'][task]['calibration_audit'].items():
            assert data.records[task][key] == value
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = {t: torch.load(source['experts'][t], map_location='cpu', weights_only=True) for t in tasks}
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    records, first_moments = {}, None
    for condition in ('moment_restart', 'moment_transport'):
        moments = {n: torch.zeros((3, *base[n].shape), dtype=torch.float64) for n in plan['active']}
        previous, previous_coefficients = experts[tasks[0]], None
        leaf.load_state_dict(previous, strict=True)
        records[condition] = []
        for count, task in enumerate(tasks):
            incoming = experts[task]
            payload = cache['experts'][task]
            assert sha256(Path(payload['path'])) == payload['sha256']
            fisher = torch.load(payload['path'], map_location='cpu', weights_only=True)
            add_moments(moments, fisher, incoming, base, plan['active'], plan['tasks'].index(task), plan['seed'])
            del fisher
            if count == 0:
                continue
            candidate = build_merger(leaf, base, previous, incoming, moments, count, plan['active'],
                                     condition, {}, previous_coefficients)
            initial = candidate.coefficients.detach().cpu().clone()
            assert initial.numel() == 347666
            if condition == 'moment_transport' and count > 1:
                assert torch.equal(initial, previous_coefficients)
            else:
                expected = torch.zeros_like(initial)
                expected[:len(plan['active'])] = 1.
                assert torch.equal(initial, expected)
                mean, _ = moment_centers(moments, count + 1)
                native = candidate.materialize()
                assert all(torch.equal(native[n], base[n] + mean[n]) for n in plan['active'])
                del native, mean
            if count == len(tasks) - 1:
                different_previous = build_merger(leaf, base, base, incoming, moments, count,
                    plan['active'], condition, {}, previous_coefficients)
                a, b = candidate.materialize(), different_previous.materialize()
                assert all(torch.equal(a[n], b[n]) for n in a)
                del different_previous, a, b
            del candidate
            state, coefficients, record = train_stage(leaf, base, previous, incoming, moments, count,
                data, tasks[:count + 1], plan, condition, device, {}, previous_coefficients)
            assert torch.equal(coefficients['initial'], initial)
            assert torch.equal(coefficients['snapshots'].double().mean(0).float(), coefficients['mean'])
            assert not torch.equal(coefficients['mean'], initial)
            if condition == 'moment_transport' and count == 1:
                reference = torch.load(root / 'engineering_moment_restart_stage2.ckpt', map_location='cpu', weights_only=True)
                reference_coefficients = torch.load(root / 'engineering_moment_restart_stage2_coefficients.pt', map_location='cpu', weights_only=True)
                assert all(torch.equal(state[n], reference[n]) for n in state)
                assert all(torch.equal(coefficients[k], reference_coefficients[k]) for k in coefficients)
                del reference, reference_coefficients
            record['initial_coefficients_match_rule_exactly'] = True
            save_tensor_file(state, root / f'engineering_{condition}_stage{count + 1}.ckpt', root, storage)
            save_tensor_file(coefficients, root / f'engineering_{condition}_stage{count + 1}_coefficients.pt', root, storage)
            records[condition].append(record)
            previous, previous_coefficients = state, coefficients['mean'].clone()
            del coefficients
            gc.collect()
            torch.cuda.empty_cache()
        if first_moments is None:
            first_moments = moments
        else:
            assert all(torch.equal(moments[n], first_moments[n]) for n in moments)
    reference_records = json.loads(Path('runs/streaming_complement_memory_v1/engineering.json').read_text())['records']['joint_memory']
    for items in records.values():
        assert [r['schedule_hash'] for r in items] == [r['schedule_hash'] for r in reference_records]
    updates = sum(r['updates'] for items in records.values() for r in items)
    assert updates == 36 and not data.valid
    moment_bytes = sum(v.numel() * v.element_size() for v in moments.values())
    assert moment_bytes == 618037248
    sources = ['fisher_moment_merge.py', 'streaming_moment_transport.py',
               'streaming_moment_transport_training.py', 'run_streaming_moment_transport.py',
               'check_streaming_moment_transport.py']
    write_json(root / 'engineering.json', {
        'passed': True, 'coefficient_optimizer_updates': updates, 'CE_example_passes': updates * 4,
        'new_Fisher_estimation_examples': 0, 'new_Fisher_estimation_VJPs': 0,
        'strict_deterministic_algorithms': True, 'coefficient_transfer_exact': True,
        'restart_initialization_is_uniform': True, 'stage2_models_and_all_coefficients_exact_between_arms': True,
        'previous_native_weights_do_not_affect_fixed_moment_coordinate_model': True,
        'moment_states_identical_despite_different_calibration_paths': True,
        'fixed_coefficient_means_exact': True, 'native_exports_pass': True,
        'paired_schedules_with_frozen_engineering': True, 'coefficient_count': 347666,
        'moment_bytes': moment_bytes, 'retained_coefficient_bytes': previous_coefficients.numel() * 4,
        'records': records, 'development_validation_read': False, 'official_test_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}})


if __name__ == '__main__':
    main()
