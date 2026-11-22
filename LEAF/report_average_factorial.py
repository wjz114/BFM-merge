"""Paired 2x2x2 comparisons of granularity, CE weighting and fixed averaging."""
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = ROOT / 'runs/aligned_merge_v1'
    new = ROOT / 'runs/paper_merge_v2'
    out = new / 'analysis'
    out.mkdir(exist_ok=True)
    scores, tasks, audits, pending = {}, {}, {}, []
    for seed, layout, balanced, averaged in itertools.product(range(3), ['block', 'tensor'], [False, True], [False, True]):
        name = (layout + '_balanced' + ('_avg' if averaged else '_ce') if balanced
                else layout + '_ce' + ('_avg' if averaged else ''))
        folder = (new if averaged and not balanced else old) / f'seed{seed}'
        path = folder / name / 'run.json'
        if not path.exists():
            pending.append(str(path.relative_to(ROOT)))
            continue
        run = json.loads(path.read_text())
        assert run.get('granularity', 'block') == layout
        assert run.get('class_balanced_ce', False) == balanced
        if averaged:
            source = old / f'seed{seed}' / (name.replace('_avg', '_ce') if balanced else name.removesuffix('_avg'))
            history = np.load(source / 'coefficient_history.npz')
            rows = [history['step'].tolist().index(i) for i in [800,960,1120,1280,1440,1600]]
            expected = history['alpha'][rows].astype(np.float64).mean(0).astype(np.float32)
            assert np.array_equal(expected, np.load(path.parent / 'best_alpha.npy'))
            assert run['source_run_sha256'] == digest(source / 'run.json')
            assert run['source_history_sha256'] == digest(source / 'coefficient_history.npz')
            assert run['alpha_updates_actual'] == 0 and run['source_optimizer_updates'] == 1600
            assert run['averaging_steps'] == [800,960,1120,1280,1440,1600]
            assert not run['engineering_subset']
            actual, original = np.load(path.parent / 'calibration_indices.npz'), np.load(source / 'calibration_indices.npz')
            assert actual.files == original.files and all(np.array_equal(actual[k], original[k]) for k in actual.files)
            audits[f'seed{seed}/{name}'] = {'fixed_mean_exact': True, 'source_and_calibration_match': True,
                'max_export_error': max(run['export_logit_max_abs'].values()),
                'run_sha256': digest(path), 'checkpoint_sha256': digest(path.parent / 'merged.ckpt')}
        per_task = {task:r['BAcc'] for task,r in run['best_val'].items()}
        key = ('validation', seed, layout, balanced, averaged)
        tasks[key], scores[key] = per_task, statistics.mean(per_task.values())
        official = folder / 'official_l2' / f'{name}.csv'
        if official.exists():
            metadata = json.loads(official.with_suffix('.json').read_text())
            assert metadata['status'] == 'complete' and metadata['seed'] == 42 and metadata['batch_size'] == 256
            assert set(metadata['completed_datasets']) == set(run['datasets'])
            per_task = {r['dataset']:float(r['level2_bAcc']) for r in csv.DictReader(official.open())
                        if r['dataset'] in run['datasets']}
            assert set(per_task)==set(run['datasets']) and len(per_task)==16
            key = ('official_l2', seed, layout, balanced, averaged)
            tasks[key], scores[key] = per_task, statistics.mean(per_task.values())
    contrasts = []
    for protocol, seed, layout in itertools.product(['validation','official_l2'], range(3), ['block','tensor']):
        cells = {(b,a):(protocol,seed,layout,b,a) for b,a in itertools.product([False,True],repeat=2)}
        if not all(k in scores for k in cells.values()):
            continue
        def contrast(name, terms):
            delta = {ds:100*sum(coefficient*tasks[cells[cell]][ds] for cell,coefficient in terms)
                     for ds in tasks[cells[(False,False)]]}
            contrasts.append({'protocol':protocol,'seed':seed,'layout':layout,'contrast':name,
                'macro_delta_pp':statistics.mean(delta.values()),'per_task_delta_pp':delta})
        contrast('balance_without_average', [((True,False),1),((False,False),-1)])
        contrast('balance_with_average', [((True,True),1),((False,True),-1)])
        contrast('average_ordinary_ce', [((False,True),1),((False,False),-1)])
        contrast('average_balanced_ce', [((True,True),1),((True,False),-1)])
        contrast('balance_x_average', [((True,True),1),((False,True),-1),((True,False),-1),((False,False),1)])
    for protocol, seed, balanced, averaged in itertools.product(['validation','official_l2'],range(3),[False,True],[False,True]):
        tensor_key, block_key = (protocol,seed,'tensor',balanced,averaged), (protocol,seed,'block',balanced,averaged)
        if tensor_key not in scores or block_key not in scores:
            continue
        delta = {ds:100*(value-tasks[block_key][ds]) for ds,value in tasks[tensor_key].items()}
        contrasts.append({'protocol':protocol,'seed':seed,'layout':'tensor_minus_block',
            'contrast':('balanced' if balanced else 'ordinary')+('_average' if averaged else '_selected'),
            'macro_delta_pp':statistics.mean(delta.values()),'per_task_delta_pp':delta})
    summary = []
    for row in contrasts:
        delta = row['per_task_delta_pp']
        row['worst_task'] = min(delta,key=delta.get)
        row['worst_task_delta_pp'] = min(delta.values())
        row['tasks_down'] = sum(v<0 for v in delta.values())
    for group in sorted({(r['protocol'],r['layout'],r['contrast']) for r in contrasts}):
        rows = [r for r in contrasts if (r['protocol'],r['layout'],r['contrast'])==group]
        values = [r['macro_delta_pp'] for r in rows]
        summary.append(dict(zip(['protocol','layout','contrast'],group), n_seeds=len(values),
            seeds=[r['seed'] for r in rows], mean_pp=statistics.mean(values),
            sample_sd_pp=statistics.stdev(values) if len(values)>1 else None,
            positive_seeds=sum(v>0 for v in values), values_pp=values))
    result = {'pending_exports':pending,'complete_exports':not pending,'audits':audits,
              'contrasts':contrasts,'summary':summary,
              'cells':[dict(zip(['protocol','seed','layout','balanced','averaged'],key),macro_bacc=value)
                       for key,value in scores.items()],
              'interpretation':'Paired development evidence; official tests already exposed. Small seeds are not independent outer confirmation. No significance threshold imposed.'}
    (out / 'average_factorial.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pending_exports':pending,'summary':summary},indent=2))


if __name__ == '__main__':
    main()
