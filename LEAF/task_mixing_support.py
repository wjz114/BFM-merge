"""Fixed four-microbatch schedules preserving every task/example multiplicity."""
import numpy as np


def build_micro_schedule(schedule, mode):
    tasks = schedule['task']
    ids = schedule['local_indices']
    assert tasks.shape == (1600,) and ids.shape == (1600, 128)
    if mode == 'native128_replay':
        return {'task': tasks[:, None].copy(), 'local_indices': ids[:, None].copy()}
    assert mode in ['same_task_micro4', 'mixed_task_micro4']
    micro_tasks = np.repeat(tasks[:, None], 4, axis=1)
    micro_ids = ids.reshape(1600, 4, 32).copy()
    for start in range(0, len(tasks), 16):
        assert sorted(tasks[start:start+16].tolist()) == list(range(16))
        if mode == 'mixed_task_micro4':
            for k in range(16):
                group, chunk = k % 4, k // 4
                rows = start + group + 4*np.arange(4)
                micro_tasks[start+k] = tasks[rows]
                micro_ids[start+k] = ids[rows, 32*chunk:32*(chunk+1)]
        for task in range(16):
            original = ids[start:start+16][tasks[start:start+16] == task].reshape(-1)
            candidate = micro_ids[start:start+16][micro_tasks[start:start+16] == task].reshape(-1)
            assert np.array_equal(np.sort(original), np.sort(candidate))
        if mode == 'mixed_task_micro4':
            for k in range(0, 16, 4):
                assert sorted(micro_tasks[start+k:start+k+4].reshape(-1).tolist()) == list(range(16))
    return {'task': micro_tasks, 'local_indices': micro_ids}


def load_micro_schedule(plan, original):
    expected = build_micro_schedule(original, plan['batch_organization'])
    with np.load(plan['micro_schedule']) as stored:
        assert set(stored.files) == set(expected)
        assert all(np.array_equal(stored[k], expected[k]) for k in expected)
    return expected
