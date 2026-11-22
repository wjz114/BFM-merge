"""Freeze a half-old/half-replacement schedule with identical batch labels."""
import json
import time
from pathlib import Path

import numpy as np

from aligned_support import array_hash
from prepare_fresh_calibration_control import sha, train_labels


def main():
    root = Path('runs/paper_merge_v2')
    out = root/'inputs/reuse_mixture_data'
    out.mkdir(exist_ok=False)
    parent_path = root/'inputs/fresh512_control_plan.json'
    plan = json.loads(parent_path.read_text())
    source = json.loads((Path(plan['source_fit'])/'run.json').read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    old, fresh = np.load(plan['calibration']), np.load(plan['training_indices'])
    schedule = np.load(plan['schedule'])
    use_old = np.zeros_like(schedule['local_indices'], dtype=bool)
    audits = {}
    for k, task in enumerate(source['datasets']):
        y = train_labels(task)
        assert array_hash(y) == coverage['audits'][task]['train_labels_hash']
        assert np.array_equal(y[old[task]], y[fresh[task]])
        rng = np.random.default_rng([20261115, k, 839])
        phase = rng.permutation(np.tile([0, 1], 256))
        appearances = np.zeros(512, dtype=np.int64)
        actual_rows, duplicate_counts = [], []
        for step in np.flatnonzero(schedule['task'] == k):
            local = schedule['local_indices'][step]
            assert len(np.unique(local)) == len(local)
            mask = ((phase[local]+appearances[local]) % 2) == 0
            use_old[step] = mask
            appearances[local] += 1
            rows = np.where(mask, old[task][local], fresh[task][local])
            assert np.array_equal(y[rows], y[old[task][local]])
            actual_rows.append(rows)
            duplicate_counts.append(len(rows)-len(np.unique(rows)))
        assert np.array_equal(appearances, np.full(512, 25))
        masks = use_old[schedule['task'] == k]
        assert int(masks.sum()) == masks.size//2 == 6400
        rows = np.concatenate(actual_rows)
        union = np.union1d(old[task], fresh[task])
        assert np.array_equal(np.unique(rows), union)
        audits[task] = {
            'old_pool_exposures': int(masks.sum()),
            'replacement_pool_exposures': int((~masks).sum()),
            'actual_new_row_exposures': int((~np.isin(rows, old[task])).sum()),
            'actual_old_row_exposures': int(np.isin(rows, old[task]).sum()),
            'unique_rows_sampled': len(union),
            'maximum_duplicate_rows_within_batch': max(duplicate_counts),
            'exact_original_batch_labels': True,
            'each_local_position_exposures': 25,
        }
    mask_path = out/'mixture_mask.npz'
    np.savez(mask_path, use_old=use_old)
    audit = {
        'unix': time.time(), 'datasets': audits, 'parent_plan': str(parent_path),
        'parent_plan_sha256': sha(parent_path), 'schedule_sha256': plan['schedule_sha256'],
        'mixture_mask_sha256': sha(mask_path), 'code_sha256': sha(__file__),
        'test_data_read': False, 'model_inference_performed': False,
        'scope': 'Every local label position alternates between the old and replacement row. Balanced random starting phases give6400 draws per pool per task while retaining every original batch label. ADHD\'s108 overlapping replacement rows make actual old-row exposure greater than50%; this is recorded. Same two-stage unique data union as the fresh512 control.',
    }
    audit_path = out/'run.json'
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    plan.update({
        'fixed_unix': time.time(), 'training_pool': 'reuse_mixture512',
        'mixture_mask': str(mask_path), 'mixture_mask_sha256': sha(mask_path),
        'mixture_audit': str(audit_path), 'mixture_audit_sha256': sha(audit_path),
        'design': str(root/'inputs/reuse_mixture_control_design.json'),
        'design_sha256': sha(root/'inputs/reuse_mixture_control_design.json'),
        'scope': audit['scope'],
    })
    for filename in ['train_reuse_mixture_control.py', 'prepare_reuse_mixture_control.py']:
        plan['source_sha256'][filename] = sha(filename)
    path = root/'inputs/reuse_mixture_control_plan.json'
    with path.open('x') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(json.dumps({'plan': str(path), 'plan_sha256': sha(path), 'datasets': audits}, indent=2))


if __name__ == '__main__':
    main()
