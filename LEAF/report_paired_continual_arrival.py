"""Summarize within-seed contrasts on identical validation EEG."""
import csv
import json
from statistics import mean

import torch
from paired_continual_arrival import DEST, ORDER

HEADS = ['teacher','expert_independent','merged_independent','merged_shared']


def main():
    rows, lines = [], [
        '# 固定顺序 43 的接入时成对实验','',
        'OpenBMI → FACED → BCIC IV-2a。每个目标任务在刚接入时评估，使用完整开发验证划分。'
        '三套专家分别来自 seed 0、1、2。每一套专家均使用同一组标签、类别描述、嵌套校准行、验证行和固定合并采样随机数。'
        '32 条预算从原系数重建，复验原来的 64 条验证探针。512 条预算从第一位专家重新起步、重新学习系数与共享接口。'
        '两预算沿用每步每任务每子集最多 4 条、12 轮、最后六轮平均。IV-2a 接入时旧记忆最多 1,024 条。','',
        '① 新专家 backbone 上独立拟合当前任务读出；② 在合并后的 backbone 上按相同规则独立拟合；'
        '③ 同一合并 backbone 的实际共享读出。①、②使用相同混合教师目标、分支均值池化、冻结的 MPNet 原型、零初始化、'
        'λ=0.001 和至多 20 步共轭梯度。③从旧共享投影出发，使用新旧数据、相同 λ 和三步校准。'
        '② 是目标任务专用的诊断参照。② 与③之差包含共享目标、初始化、回放和求解方式的综合影响。','',
        '| 任务 | 校准条数 | 验证条数 | 教师 | ① | ② | ③ | ②−①（pp） | ③−②（pp） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    faced_counts = []
    for task in ORDER[1:]:
        ids = None
        for n in (32,512):
            group=[]
            for seed in range(3):
                folder=DEST/f'seed{seed}'/f'n{n}'
                result=json.loads((folder/f'{task}.json').read_text())
                scores=torch.load(folder/f'{task}_scores.pt',weights_only=True)
                if ids is None:
                    ids=scores['validation_ids']
                assert scores['validation_ids']==ids
                assert len(scores['calibration_ids'])==n and set(ids).isdisjoint(scores['calibration_ids'])
                if n==512:
                    prev=torch.load(folder.parent/'n32'/f'{task}_scores.pt',weights_only=True)
                    assert scores['calibration_ids'][:32]==prev['calibration_ids']
                    torch.testing.assert_close(scores['logits']['teacher'],prev['logits']['teacher'])
                    assert torch.equal(scores['y'],prev['y'])
                if task=='EMO_FACED' and n==512:
                    vocab=torch.load(DEST.parent/'stage_b_reference/prototypes/EMO_FACED.pt',weights_only=True)
                    classes=torch.tensor([vocab['labels'].index(k) for k in ('Fear','Sad')])
                    preds=[scores['y']]+[scores['logits'][k].argmax(1) for k in ('merged_independent','merged_shared')]
                    faced_counts.append([100*torch.isin(v,classes).double().mean().item() for v in preds])
                row={k:v for k,v in result.items() if k!='probe64'}
                row['backbone_delta_pp']=100*(row['merged_independent']-row['expert_independent'])
                row['shared_delta_pp']=100*(row['merged_shared']-row['merged_independent'])
                group.append(row)
                rows.append(row)
            vals=[100*mean(r[k] for r in group) for k in HEADS]
            gaps=[mean(r[k] for r in group) for k in ('backbone_delta_pp','shared_delta_pp')]
            lines.append(f'| {task} | {n} | {len(ids)} | '+' | '.join(f'{v:.2f}%' for v in vals)
                         +' | '+' | '.join(f'{v:+.2f}' for v in gaps)+' |')
    with (DEST/'per_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines+=['','FACED 的 512 条校准条件下，Fear 和 Sad 两类在验证集的合计占比：','',
            '| seed | 真实标签 | 合并 backbone＋独立读出 | 实际共享读出 |','|---|---:|---:|---:|']
    lines.extend('| '+str(i)+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |' for i,vals in enumerate(faced_counts))
    lines+=['','结果均为三个 expert seed 的等权均值。当前验证集用于开发；官方测试集未参与。'
             '单 seed 结果与配对差见 [per_seed.csv](per_seed.csv)。各子目录保存逐样本分数、标签和样本 ID。','']
    (DEST/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[-8:]))


if __name__=='__main__':
    main()
