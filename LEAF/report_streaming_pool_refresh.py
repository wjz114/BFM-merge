"""Audit the unchanged merger and class-matched pool experiment."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from report_streaming_replica_filter import audit_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    plan = json.loads((root / 'plan.json').read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    torch.set_num_threads(4)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    original_ids = np.load(plan['calibration'])
    labels = {}
    for task in plan['pilot_tasks']:
        with h5py.File(DATA_FOLDER / f'{task}.h5', 'r', locking=False) as handle:
            _, labels[task] = row_map(handle, task, 'calibration')
            labels[task] = labels[task].astype(np.int64)
    results, audits, task_scores, union = {}, {}, {}, {}
    fixed = set(base) - set(plan['active'])
    for order in plan['orders']:
        ref_path = Path(plan['reference_runs'][order])
        ref = json.loads(ref_path.read_text())
        results.setdefault('original512', {})[order] = ref['stages'][-1]['macro_BAcc']
        task_scores.setdefault('original512', {})[order] = ref['stages'][-1]['validation']
        for condition in plan['conditions']:
            folder = root / f'{condition}_{order}'
            run = json.loads((folder / 'run.json').read_text())
            assert run['status'] == 'completed' and not run['official_test_read']
            assert run['updates'] == 108 and run['training_examples'] == 1728
            assert run['plan_sha256'] == sha256(root / 'plan.json')
            assert run['order'] == ref['order'] and len(run['stages']) == 4
            results.setdefault(condition, {})[order] = run['stages'][-1]['macro_BAcc']
            task_scores.setdefault(condition, {})[order] = run['stages'][-1]['validation']
            previous, maximum, seen_ids, fixed_ids = None, 0., {}, {}
            for index, stage in enumerate(run['stages'], 1):
                task = run['order'][index - 1]
                directory = folder / f'stage_{index:02d}'
                assert sha256(directory / 'merged.ckpt') == stage['checkpoint_sha256']
                state = torch.load(directory / 'merged.ckpt', map_location='cpu', weights_only=True)
                path = Path(source['experts'][task])
                assert sha256(path) == plan['expert_sha256'][str(path)]
                incoming = torch.load(path, map_location='cpu', weights_only=True)
                assert state.keys() == base.keys()
                assert all(torch.equal(state[k], base[k]) for k in fixed)
                if index == 1:
                    assert all(torch.equal(state[k], incoming[k]) for k in state)
                else:
                    assert stage['schedule_hash'] == ref['stages'][index - 1]['schedule_hash']
                    assert stage['updates'] == 12 * index
                    path = directory / 'coefficients.pt'
                    assert sha256(path) == stage['coefficients_sha256']
                    payload = torch.load(path, map_location='cpu', weights_only=True)
                    assert payload['snapshots'].shape == (6, 347666)
                    assert torch.equal(payload['snapshots'].double().mean(0).float(), payload['mean'])
                    maximum = max(maximum, audit_export(base, previous, incoming,
                                                       state, payload['mean'], plan['active']))
                    assert set(payload['training_pool_ids']) == set(run['order'][:index])
                    for old, ids in payload['training_pool_ids'].items():
                        ids = ids.numpy()
                        record = stage['pool_refresh'][old]
                        assert len(ids) == len(np.unique(ids)) == 512
                        assert 0 <= ids.min() <= ids.max() < len(labels[old])
                        template = labels[old][original_ids[old]]
                        assert np.array_equal(labels[old][ids], template)
                        assert array_hash(ids) == record['ids_hash']
                        assert array_hash(template) == record['labels_hash']
                        assert record['independent_reader_agreement']
                        if condition == 'fixed_fresh_pool' and old in fixed_ids:
                            assert np.array_equal(ids, fixed_ids[old])
                        fixed_ids[old] = ids
                        perm = np.random.default_rng([plan['seed'], index,
                            plan['tasks'].index(old), 217]).permutation(512)[:192]
                        seen_ids.setdefault(old, set()).update(ids[perm].tolist())
                    assert all(c['max_abs_logit_error'] <= 1e-5 and c['changed_predictions'] == 0
                        for world in stage['export_checks'].values() for c in world.values())
                previous = state
            for task in plan['pilot_tasks']:
                assert run['dataset_audits'][task]['class_weights'] == ref['dataset_audits'][task]['class_weights']
                assert run['dataset_audits'][task]['validation_size'] == ref['dataset_audits'][task]['validation_size']
            audits[f'{condition}_{order}'] = dict(independent_native_max_error=maximum,
                labels_weights_schedule_unchanged=True, arrived_task_training_pools_only=True)
            union[f'{condition}_{order}'] = {t: len(v) for t, v in seen_ids.items()}
    contrasts = {}
    for left, right in [('stage_fresh_pool', 'original512'), ('fixed_fresh_pool', 'original512'),
                        ('stage_fresh_pool', 'fixed_fresh_pool')]:
        name = left + '_minus_' + right
        effects = {o: 100 * (results[left][o] - results[right][o]) for o in plan['orders']}
        task_effects = {t: float(np.mean([100 * (task_scores[left][o][t]['BAcc'] -
            task_scores[right][o][t]['BAcc']) for o in plan['orders']])) for t in plan['pilot_tasks']}
        contrasts[name] = dict(paired_effects_pp=effects, mean_pp=float(np.mean(list(effects.values()))),
            task_mean_effects_pp=task_effects,
            leave_one_task_out_pp={t: float(np.mean([v for k, v in task_effects.items() if k != t])) for t in task_effects})
    report = dict(status='passed', plan_sha256=sha256(root / 'plan.json'), results=results,
        contrasts=contrasts, audits=audits, unique_training_rows_across_stages=union,
        scientific_runs=4, scientific_optimizer_updates=432, training_CE_examples=6912,
        new_source_steps=0, official_test_read=False, novelty_established=False,
        overall_goal_achieved=False, scope=plan['scope'])
    with (root / 'comparison.json').open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'results', 'contrasts']}, indent=2))


if __name__ == '__main__':
    main()
