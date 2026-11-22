"""Aggregate the fixed-order development replication, including the original seed 0."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

RUN = Path(__file__).resolve().parent/'runs/continual_v12'
DEST = RUN/'stage_b_replication/order43'
ORDER = ['MI_OpenBMI','EMO_FACED','MI_BCIC_IV2a','EMO_SEED_3_seg4']
ARMS = ['uniform','ordinary','detach','full']


def write_csv(name, rows):
    with (DEST/name).open('w') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    summary, per_task = [], []
    for seed in range(3):
        root = RUN/'stage_b_smoke/43' if seed == 0 else DEST/f'seed{seed}'
        teacher = {}
        for arm in ARMS:
            stages = json.loads((root/arm/'results.json').read_text())
            assert [s['arrival'] for s in stages] == ORDER
            assert [s['updates'] for s in stages] == ([0]*4 if arm=='uniform' else [0,12,24,36])
            final = stages[-1]['eval']
            for i, s in enumerate(stages,1):
                assert s.get('expert_seed',0) == seed and list(s['eval']) == ORDER[:i]
                for task, v in s['eval'].items():
                    assert teacher.setdefault(task,v['teacher_bacc']) == v['teacher_bacc']
                    per_task.append(dict(seed=seed,arm=arm,stage=i,task=task,arrival=s['arrival'],**v))
            gap = mean(s['eval'][t]['teacher_bacc']-s['eval'][t]['bacc'] for s,t in zip(stages,ORDER))
            forget = mean(max(s['eval'][t]['bacc'] for s in stages if t in s['eval'])-final[t]['bacc'] for t in ORDER[:-1])
            group = lambda prefix: mean(v['bacc'] for t,v in final.items() if t.startswith(prefix))
            summary.append(dict(seed=seed,arm=arm,final=group(''),mi=group('MI_'),emotion=group('EMO_'),
                                arrival_gap=gap,forgetting=forget))
    write_csv('per_task.csv',per_task)
    write_csv('summary.csv',summary)
    by_arm = {a:[r for r in summary if r['arm']==a] for a in ARMS}
    names = {'uniform':'均匀合并＋接口校准','ordinary':'普通 CE 张量合并',
             'detach':'切断接口校准梯度','full':'完整方法'}
    lines = ['# 固定顺序的三专家 seed 复验', '',
             '按 seed 0 完整方法的四任务最终平均 BAcc 选定顺序 43（46.49%）。'
             '依次接入 OpenBMI、FACED、BCIC IV-2a、SEED 三分类。只改变专家训练 seed，'
             '固定校准与验证行、合并器采样随机数、优化步数和超参数。seed 0 复用原试验。'
             '每任务 32 条校准样本，固定 64 条分层验证样本；所有数值均为开发验证结果。', '',
             '| 方法 | seed 0 | seed 1 | seed 2 | 四任务均值 ± 样本标准差 |',
             '|---|---:|---:|---:|---:|']
    for a,rows in by_arm.items():
        vals=[100*r['final'] for r in rows]
        lines.append('| '+names[a]+' | '+' | '.join(f'{v:.2f}%' for v in vals)
                     + f' | {mean(vals):.2f}% ± {stdev(vals):.2f} pp |')
    lines += ['', '| 任务 | 同验证样本独立专家 | 完整方法，接入当时 | 完整方法，最终 |',
              '|---|---:|---:|---:|']
    for t in ORDER:
        rows=[r for r in per_task if r['task']==t and r['stage']==4 and r['arm']=='full']
        arrival=[r for r in per_task if r['task']==t==r['arrival'] and r['arm']=='full']
        lines.append(f"| {t} | {100*mean(r['teacher_bacc'] for r in rows):.2f}% | "
                     f"{100*mean(r['bacc'] for r in arrival):.2f}% | {100*mean(r['bacc'] for r in rows):.2f}% |")
    lines += ['', 'seed 0 参与顺序选择，seed 1、2 是新增的复验。三个 seed 共享验证样本，'
              '标准差只表达不同专家训练 seed 的波动。', '',
              '[逐阶段成绩](per_task.csv) · [按 seed 汇总，含 MI、情绪、新任务接入损失和最终遗忘](summary.csv)', '']
    (DEST/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines))



if __name__=='__main__':
    main()
