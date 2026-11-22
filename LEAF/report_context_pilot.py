"""Report every fixed cell of the matched teacher/context pilot."""
import json
from pathlib import Path
from statistics import mean

import torch

from d_downstream_qformer_ce import sha256
from report_ri_geometry import describe


def main():
    root = Path('runs/paper_merge_v2')
    path = root/'seed0/context_pilot_evaluation/run.json'
    data = json.loads(path.read_text())
    adaptations = {c: json.loads((root/f'seed0/context_pilot_{c}/MI_Cho2017/run.json').read_text())['context_adaptation']
                   for c in ['composed', 'direct']}
    left, right = adaptations.values()
    for key in ['source_sha256', 'plan', 'updates', 'other_domain_update_counts', 'code_sha256']:
        assert left[key] == right[key], key
    assert left['updates'] == 2500
    comparisons = []
    for split in ['calibration', 'validation']:
        for context in ['base', 'target_expert']:
            for scale in data['plan']['scales']:
                cells = {bank: {row['target']: row['splits'][split] for row in data['records']
                                if row['bank'] == bank and row['context'] == context and row['scale'] == scale}
                         for bank in data['plan']['checkpoints']}
                assert all(len(rows) == 5 for rows in cells.values())
                for treatment, control in [('composed', 'direct'), ('composed', 'ri'), ('direct', 'ri'),
                                            ('composed', 'original'), ('direct', 'original')]:
                    per_task = {task: {'bacc_change_pp': 100*(values['BAcc']-cells[control][task]['BAcc']),
                                      'kl_change': values['kl_from_context']-cells[control][task]['kl_from_context']}
                                for task, values in cells[treatment].items()}
                    comparisons.append({'split': split, 'context': context, 'scale': scale,
                                        'treatment': treatment, 'control': control, 'per_task': per_task,
                                        'macro_bacc_change_pp': mean(v['bacc_change_pp'] for v in per_task.values()),
                                        'macro_kl_change': mean(v['kl_change'] for v in per_task.values()),
                                        'improved_tasks': sum(v['bacc_change_pp']>0 for v in per_task.values()),
                                        'declined_tasks': sum(v['bacc_change_pp']<0 for v in per_task.values())})
    primary = next(row for row in comparisons if row['split']=='validation' and row['context']=='target_expert'
                   and row['scale']==1 and row['treatment']=='composed' and row['control']=='direct')
    torch.set_num_threads(4)
    source = json.loads((Path(data['plan']['source'])/'run.json').read_text())
    assert source['granularity'] == 'tensor'
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    original = torch.load(data['plan']['checkpoints']['original'], map_location='cpu', weights_only=True)
    geometry = {}
    for bank in ['ri', 'composed', 'direct']:
        adapted = torch.load(data['plan']['checkpoints'][bank], map_location='cpu', weights_only=True)
        sums = [0.0]*4
        for name in source['groups']:
            u = original[name].double()-base[name].double()
            v = adapted[name].double()-base[name].double()
            for i, value in enumerate([u.square().sum(), v.square().sum(), (u*v).sum(), (v-u).square().sum()]):
                sums[i] += float(value)
        geometry[bank] = describe(sums)
    result = {'primary': primary, 'all_comparisons': comparisons, 'own_retention': data['own'],
              'task_vector_geometry': geometry,
              'geometry_scope': 'Coordinate-dependent Euclidean summaries, not a norm-matched causal control.',
              'cost': {c: {'fit_seconds': row['seconds']['fit'], 'total_seconds': row['seconds']['expert_total'],
                           'updates': row['updates'], 'examples_processed': row['updates']*64}
                       for c, row in adaptations.items()},
              'source_sha256': sha256(path), 'code_sha256': sha256(Path(__file__)),
              'scope': 'One deliberately selected source expert; five pre-existing MI target domains. Development-exposed full validation. The matched contrast changes parameter context only. RI contrasts also change teacher. No16-task adapted-bank merge or independent confirmation.'}
    (root/'analysis/context_pilot_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'primary': primary, 'own_validation': {k:v['validation'] for k,v in data['own'].items()}}, indent=2))


if __name__ == '__main__':
    main()
