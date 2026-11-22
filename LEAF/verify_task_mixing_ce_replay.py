"""Exact32-step native CE replay for the new task-mixing accumulation loop."""
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def main():
    root = Path('runs/paper_merge_v2')
    reference = root/'seed0/bank_ce_deterministic_replay'
    candidate = root/'seed0/task_mixing_native128_replay'
    old, new = [json.loads((p/'run.json').read_text()) for p in [reference, candidate]]
    assert old['engineering_smoke'] and new['engineering_smoke']
    assert old['updates'] == new['updates'] == 32
    assert new['batch_organization'] == 'native128_replay'
    assert old['datasets'] == new['datasets'] and old['schedule_sha256'] == new['schedule_sha256']
    checks = {}
    for name in ['step_16.ckpt', 'step_32.ckpt', 'selected.ckpt', 'final.ckpt', 'merged.ckpt']:
        a, b = [torch.load(p/name, map_location='cpu', weights_only=True) for p in [reference, candidate]]
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a), name
        checks[name] = {'all_tensors_exact': True, 'tensor_count': len(a), 'max_abs_error': 0.}
    for key in ['initial_validation', 'selected_validation', 'final_validation', 'average_validation',
                'average_calibration', 'dataset_audits', 'component_gradient_max']:
        assert old[key] == new[key], key
    assert json.loads((reference/'history.json').read_text()) == json.loads((candidate/'history.json').read_text())
    assert max(v['max_abs_logit_error'] for v in new['export_checks'].values()) <= 1e-5
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'reference': str(reference/'run.json'), 'reference_sha256': sha256(reference/'run.json'),
              'candidate': str(candidate/'run.json'), 'candidate_sha256': sha256(candidate/'run.json'),
              'state_checks': checks,
              'all_native_metrics_histories_data_and_component_gradient_max_exact': True,
              'trainer_sha256': sha256(Path('train_task_mixing_ce.py')),
              'support_sha256': sha256(Path('task_mixing_support.py')),
              'scope': 'Real EEG32-step deterministic native128 replay across all16 tasks. Micro4 schedule construction separately verifies identical task/row multisets in every16-step block. Micro4 floating-point accumulation need not exactly equal native128; the scientific pair both use four32-example forwards/backwards.'}
    with (root/'analysis/task_mixing_ce_replay_verification.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('PASS: all396 tensors in five checkpoint states, metrics, histories and gradient maxima exactly replay CE.')


if __name__ == '__main__':
    main()
