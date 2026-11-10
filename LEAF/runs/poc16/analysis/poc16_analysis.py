"""Analysis + figures for the 16-source, 3-seed LEAF merging POC (runs/poc16).

Reads only run artifacts; writes figures to runs/poc16/figures/ and numbers + markdown tables to
scratchpad/poc16_analysis.json.
"""
import csv, json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path('/home/jingzhou/BFM-merge/LEAF')
R16, R5 = ROOT / 'runs/poc16', ROOT / 'runs/poc3k'
FIG = R16 / 'figures'; FIG.mkdir(exist_ok=True)
OUT = Path(__file__).with_name('poc16_analysis.json')
SEEDS = [0, 1, 2]
DS = ['MI_OpenBMI', 'MI_BCIC_IV2a', 'MI_ShanghaiU', 'MI_BCIC_Upperlimb', 'MI_HighGamma', 'MI_Cho2017', 'MI_Shin2017A',
      'MI_PhysioNet', 'EMO_FACED', 'EMO_SEED_3_seg4', 'EMO_SEED_4_seg4', 'EMO_SEED_5_seg4', 'EMO_SEED_7_seg4',
      'SSVEP_OpenBMI', 'Workload', 'ADHD_AliMotie']
DS5 = ['MI_OpenBMI', 'MI_Cho2017', 'MI_PhysioNet', 'EMO_SEED_3_seg4', 'SSVEP_OpenBMI']
SHORT = {'MI_OpenBMI': 'OpenBMI', 'MI_BCIC_IV2a': 'BCIC-IV2a', 'MI_ShanghaiU': 'ShanghaiU', 'MI_BCIC_Upperlimb': 'Upperlimb',
         'MI_HighGamma': 'HighGamma', 'MI_Cho2017': 'Cho2017', 'MI_Shin2017A': 'Shin2017A', 'MI_PhysioNet': 'PhysioNet',
         'EMO_FACED': 'FACED', 'EMO_SEED_3_seg4': 'SEED-3', 'EMO_SEED_4_seg4': 'SEED-4', 'EMO_SEED_5_seg4': 'SEED-5',
         'EMO_SEED_7_seg4': 'SEED-7', 'SSVEP_OpenBMI': 'SSVEP', 'Workload': 'Workload', 'ADHD_AliMotie': 'ADHD'}
MODELS = ['Base', 'Joint', 'Uniform', 'Global', 'Block', 'Specialist']
MERGES = ['Uniform', 'Global', 'Block']
K = len(DS)
A, T = {}, {}   # numbers, markdown tables


def mean(x): return float(np.mean(x))
def sd(x): return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
def f4(m, s): return f'{m:.4f} ± {s:.4f}'
def z(m, nd=2): return 0.0 if round(m, nd) == 0 else m
def pp(m, s=None): return f'{z(m):+.2f}' if s is None else f'{z(m):+.2f} ± {s:.2f}'


def md(header, rows, align=None):
    align = align or ['---'] + ['---:'] * (len(header) - 1)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ---------------------------------------------------------------- load official (c_direct_inference) tables
def load_direct(run, s, ds_list):
    f = run / f'seed{s}' / ('direct_results_16sources.csv' if (run == R16 and s == 0) else 'direct_results.csv')
    rows = [r for r in csv.DictReader(open(f)) if r['role'] == 'source']
    out = {}
    for r in rows:
        out[(r['model'], r['dataset'], int(r['level']))] = float(r['BAcc'])
    for m in MODELS:
        for lv in (0, 1, 2):
            assert all((m, d, lv) in out for d in ds_list), (run, s, m, lv)
    assert set(d for (_, d, _) in out) == set(ds_list) | {'SOURCE_MACRO'}, (run, s)
    for m in MODELS:   # the stored macro row agrees with the mean of the rounded per-dataset values
        for lv in (0, 1, 2):
            assert abs(out[(m, 'SOURCE_MACRO', lv)] - np.mean([out[(m, d, lv)] for d in ds_list])) < 2e-4
    return out


OFF = {s: load_direct(R16, s, DS) for s in SEEDS}
OFF5 = {s: load_direct(R5, s, DS5) for s in SEEDS}


def macro(tab, m, lv=2, ds_list=DS):
    if ds_list is DS or (tab is not None and len(ds_list) == len({d for (_, d, _) in tab}) - 1):
        return tab[(m, 'SOURCE_MACRO', lv)]
    return mean([tab[(m, d, lv)] for d in ds_list])


# ---------------------------------------------------------------- A. main official comparison
A['main'] = {}
for m in MODELS:
    per = [macro(OFF[s], m) for s in SEEDS]
    dj = [100 * (macro(OFF[s], m) - macro(OFF[s], 'Joint')) for s in SEEDS]
    db = [100 * (macro(OFF[s], m) - macro(OFF[s], 'Base')) for s in SEEDS]
    A['main'][m] = dict(per_seed=per, mean=mean(per), std=sd(per), dJ=mean(dj), dJ_std=sd(dj), dJ_per_seed=dj,
                        dB=mean(db), dB_std=sd(db))
LABEL = {'Specialist': 'Specialist (16 experts, reference)'}
T['main'] = md(['Model', 'Source macro BAcc', 'seed 0', 'seed 1', 'seed 2', 'Δ vs Joint (pp)', 'Δ vs Base (pp)'],
               [[LABEL.get(m, m), f4(A['main'][m]['mean'], A['main'][m]['std']),
                 *[f'{v:.4f}' for v in A['main'][m]['per_seed']],
                 pp(A['main'][m]['dJ'], A['main'][m]['dJ_std']), pp(A['main'][m]['dB'], A['main'][m]['dB_std'])]
                for m in MODELS])
# recovery of Joint's gain over Base
A['recovery'] = {m: mean([(macro(OFF[s], m) - macro(OFF[s], 'Base')) / (macro(OFF[s], 'Joint') - macro(OFF[s], 'Base'))
                          for s in SEEDS]) for m in MERGES + ['Specialist']}

# all levels
LVN = {0: 'Level 0 (Default)', 1: 'Level 1 (Task)', 2: 'Level 2 (Task+Target, main)'}
A['levels'] = {m: {lv: dict(mean=mean([macro(OFF[s], m, lv) for s in SEEDS]), std=sd([macro(OFF[s], m, lv) for s in SEEDS]),
                            dJ=mean([100 * (macro(OFF[s], m, lv) - macro(OFF[s], 'Joint', lv)) for s in SEEDS]),
                            dJ_std=sd([100 * (macro(OFF[s], m, lv) - macro(OFF[s], 'Joint', lv)) for s in SEEDS]))
                   for lv in (0, 1, 2)} for m in MODELS}
T['levels'] = md(['Model'] + [LVN[lv] for lv in (0, 1, 2)] + [f'Δ vs Joint, L{lv} (pp)' for lv in (0, 1, 2)],
                 [[m] + [f4(A['levels'][m][lv]['mean'], A['levels'][m][lv]['std']) for lv in (0, 1, 2)]
                  + [pp(A['levels'][m][lv]['dJ'], A['levels'][m][lv]['dJ_std']) for lv in (0, 1, 2)] for m in MODELS])
A['level_rank'] = {lv: sorted(MODELS, key=lambda m: -A['levels'][m][lv]['mean']) for lv in (0, 1, 2)}
A['rank_per_seed'] = {s: sorted(MODELS, key=lambda m: -macro(OFF[s], m)) for s in SEEDS}

# paired comparisons
PAIRS = [('Block', 'Global'), ('Global', 'Uniform'), ('Uniform', 'Base'), ('Global', 'Base'), ('Block', 'Base'),
         ('Block', 'Joint'), ('Joint', 'Base'), ('Specialist', 'Joint'), ('Specialist', 'Block')]
A['pairs'] = []
rows = []
for a, b in PAIRS:
    d = [100 * (macro(OFF[s], a) - macro(OFF[s], b)) for s in SEEDS]
    wins = [sum(OFF[s][(a, x, 2)] > OFF[s][(b, x, 2)] for x in DS) for s in SEEDS]
    dm = {x: mean([OFF[s][(a, x, 2)] - OFF[s][(b, x, 2)] for s in SEEDS]) for x in DS}
    wins_mean = sum(v > 0 for v in dm.values())
    A['pairs'].append(dict(a=a, b=b, per_seed=d, mean=mean(d), std=sd(d), wins=wins, wins_seedmean=wins_mean,
                           same_sign=all(np.sign(v) == np.sign(d[0]) for v in d)))
    rows.append([f'{a} − {b}', *[f'{v:+.2f}' for v in d], pp(mean(d), sd(d)), '/'.join(map(str, wins)) + ' of 16',
                 f'{wins_mean} of 16'])
T['pairs'] = md(['Comparison', 'seed 0', 'seed 1', 'seed 2', 'Mean ± std', 'Datasets improved (per seed)',
                 'Datasets improved (seed mean)'], rows)

# ---------------------------------------------------------------- B. per dataset
A['perds'] = {}
for d in DS:
    A['perds'][d] = {}
    for m in MODELS:
        v = [OFF[s][(m, d, 2)] for s in SEEDS]
        dj = [100 * (OFF[s][(m, d, 2)] - OFF[s][('Joint', d, 2)]) for s in SEEDS]
        A['perds'][d][m] = dict(mean=mean(v), std=sd(v), dJ=mean(dj), dJ_std=sd(dj), per_seed=v)
T['perds'] = md(['Dataset', 'Classes', *MODELS],
                [[d, None, *[f"{A['perds'][d][m]['mean']:.4f}" + ('' if m == 'Base' else f" ± {A['perds'][d][m]['std']:.4f}")
                             for m in MODELS]] for d in DS]
                + [['**Source macro**', '', *[f"**{A['main'][m]['mean']:.4f}**" for m in MODELS]]])
T['delta'] = md(['Dataset', *[m for m in MODELS if m != 'Joint']],
                [[d, *[pp(A['perds'][d][m]['dJ'], A['perds'][d][m]['dJ_std']) for m in MODELS if m != 'Joint']] for d in DS]
                + [['**Source macro**', *[f"**{pp(A['main'][m]['dJ'], A['main'][m]['dJ_std'])}**" for m in MODELS if m != 'Joint']]])

# win / loss counts per seed
A['counts'] = {}
for m in ['Base'] + MERGES + ['Specialist']:
    A['counts'][m] = dict(gt_joint=[sum(OFF[s][(m, d, 2)] > OFF[s][('Joint', d, 2)] for d in DS) for s in SEEDS],
                          lt_base=[sum(OFF[s][(m, d, 2)] < OFF[s][('Base', d, 2)] for d in DS) for s in SEEDS] if m != 'Base' else None,
                          lt_base_seedmean=[SHORT[d] for d in DS if m != 'Base' and A['perds'][d][m]['mean'] < A['perds'][d]['Base']['mean']])
A['joint_lt_base'] = [SHORT[d] for d in DS if A['perds'][d]['Joint']['mean'] < A['perds'][d]['Base']['mean']]
T['counts'] = md(['Model', 'Datasets above Joint (seed 0/1/2)', 'Datasets below Base (seed 0/1/2)', 'Below Base on the seed mean'],
                 [[m, '/'.join(map(str, A['counts'][m]['gt_joint'])),
                   '–' if m == 'Base' else '/'.join(map(str, A['counts'][m]['lt_base'])),
                   '–' if m == 'Base' else (', '.join(A['counts'][m]['lt_base_seedmean']) or 'none')]
                  for m in ['Base'] + MERGES + ['Specialist']], ['---', '---:', '---:', '---'])

# gap decomposition: macro gap = (1/K) Σ_d (Joint_d − M_d)
A['gap'] = {}
for m in MERGES:
    tot = mean([macro(OFF[s], 'Joint') - macro(OFF[s], m) for s in SEEDS])
    contrib = {d: mean([OFF[s][('Joint', d, 2)] - OFF[s][(m, d, 2)] for s in SEEDS]) / K for d in DS}
    A['gap'][m] = dict(total_pp=100 * tot, contrib_pp={d: 100 * v for d, v in contrib.items()},
                       share={d: v / tot for d, v in contrib.items()})
order = sorted(DS, key=lambda d: -A['gap']['Block']['contrib_pp'][d])
cum = np.cumsum([A['gap']['Block']['share'][d] for d in order])
T['gap'] = md(['Dataset (sorted by Block gap)', 'Joint − Block (pp)', 'Contribution to macro gap (pp)', 'Share of Block macro gap',
               'Cumulative share', 'Share of Global gap', 'Share of Uniform gap'],
              [[d, pp(-A['perds'][d]['Block']['dJ'], A['perds'][d]['Block']['dJ_std']), f"{z(A['gap']['Block']['contrib_pp'][d]):+.2f}",
                f"{z(100 * A['gap']['Block']['share'][d], 1):+.1f} %", f'{100 * c:.1f} %', f"{z(100 * A['gap']['Global']['share'][d], 1):+.1f} %",
                f"{z(100 * A['gap']['Uniform']['share'][d], 1):+.1f} %"] for d, c in zip(order, cum)])
A['gap_order'] = order

# retention of the specialist's gain over Base, on datasets where the specialist gains >= 2 pp
A['retention'] = {}
rows = []
for d in DS:
    sb = A['perds'][d]['Specialist']['mean'] - A['perds'][d]['Base']['mean']
    jb = A['perds'][d]['Joint']['mean'] - A['perds'][d]['Base']['mean']
    if sb < 0.05:
        continue
    r = {m: (A['perds'][d][m]['mean'] - A['perds'][d]['Base']['mean']) / sb for m in ['Joint'] + MERGES}
    A['retention'][d] = dict(spec_gain_pp=100 * sb, **r)
    rows.append([d, f'{100 * sb:+.2f}', *[f'{100 * r[m]:.0f} %' for m in ['Joint'] + MERGES]])
T['retention'] = md(['Dataset', 'Specialist − Base (pp)', 'Joint', 'Uniform', 'Global', 'Block'], rows)
A['retention_excluded'] = [d for d in DS if d not in A['retention']]

# ---------------------------------------------------------------- C. supplementary protocol (eval_poc.py)
def load_supp(run, s):
    rows = list(csv.DictReader(open(run / f'seed{s}' / 'results.csv')))
    return {(r['model'], r['dataset']): r for r in rows}


SUP = {s: load_supp(R16, s) for s in SEEDS}
for s in SEEDS:
    ds_seen = {d for (_, d) in SUP[s]}
    assert ds_seen == set(DS) | {'SOURCE_MACRO', 'MI_Weibo2014'}, ds_seen
A['supp'] = {}
for m in MODELS:
    e = {}
    for met in ['BAcc', 'Kappa', 'Auroc', 'wF1', 'Loss']:
        v = [mean([float(SUP[s][(m, d)][met]) for d in DS]) for s in SEEDS]
        e[met] = (mean(v), sd(v))
    v = [mean([float(SUP[s][(m, d)]['BAcc']) for d in DS]) - mean([float(SUP[s][('Joint', d)]['BAcc']) for d in DS]) for s in SEEDS]
    e['dJ'] = (100 * mean(v), 100 * sd(v))
    if (m, 'MI_Weibo2014') in SUP[0]:
        for met in ['BAcc', 'Kappa', 'Auroc']:
            w = [float(SUP[s][(m, 'MI_Weibo2014')][met]) for s in SEEDS]
            e['weibo_' + met] = (mean(w), sd(w))
        w = [100 * (float(SUP[s][(m, 'MI_Weibo2014')]['BAcc']) - float(SUP[s][('Joint', 'MI_Weibo2014')]['BAcc'])) for s in SEEDS]
        e['weibo_dJ'] = (mean(w), sd(w))
        e['weibo_per_seed'] = [float(SUP[s][(m, 'MI_Weibo2014')]['BAcc']) for s in SEEDS]
        e['weibo_n'] = SUP[0][(m, 'MI_Weibo2014')]['n']
    A['supp'][m] = e
T['supp'] = md(['Model', 'BAcc', 'Kappa', 'AUROC', 'wF1', 'CE', 'Δ BAcc vs Joint (pp)'],
               [[m, *[f4(*A['supp'][m][k]) for k in ['BAcc', 'Kappa', 'Auroc', 'wF1', 'Loss']], pp(*A['supp'][m]['dJ'])] for m in MODELS])
T['weibo'] = md(['Model', 'Weibo BAcc', 'seed 0 / 1 / 2', 'Kappa', 'AUROC', 'Δ BAcc vs Joint (pp)'],
                [[m, f4(*A['supp'][m]['weibo_BAcc']), ' / '.join(f'{v:.4f}' for v in A['supp'][m]['weibo_per_seed']),
                  f4(*A['supp'][m]['weibo_Kappa']), f4(*A['supp'][m]['weibo_Auroc']), pp(*A['supp'][m]['weibo_dJ'])]
                 for m in MODELS if 'weibo_BAcc' in A['supp'][m]])
# per-dataset agreement between the two protocols
off_sup = [(A['perds'][d][m]['mean'], mean([float(SUP[s][(m, d)]['BAcc']) for s in SEEDS])) for d in DS for m in MODELS]
A['protocol_max_absdiff'] = float(max(abs(a - b) for a, b in off_sup))
A['protocol_corr'] = float(np.corrcoef(*zip(*off_sup))[0, 1])

# ---------------------------------------------------------------- D. validation / optimisation
def runj(p): return json.load(open(p))


A['val'] = {}
rows = []
for s in SEEDS:
    j = runj(R16 / f'seed{s}/joint/run.json')
    ent = dict(joint=dict(best=j['best_val_macro_BAcc'], best_round=j['best_round'], rounds=j['rounds'], seconds=j['seconds']['total']))
    for m in ['uniform', 'global', 'block']:
        r = runj(R16 / f'seed{s}/merge_{m}/run.json')
        h = runj(R16 / f'seed{s}/merge_{m}/history.json')
        ent[m] = dict(best=r['best_val_macro_BAcc'], best_ce=r['best_val_macro_Loss'], best_step=r['best_step'],
                      folded=r['folded_val_macro_BAcc'], steps=[x['step'] for x in h], curve=[x['val_macro_BAcc'] for x in h],
                      curve_ce=[x['val_macro_Loss'] for x in h], seconds=r['seconds']['total'],
                      tv_l2=r['task_vector_l2'] if isinstance(r['task_vector_l2'], dict) else eval(r['task_vector_l2']))
        if m != 'uniform':
            met = np.load(R16 / f'seed{s}/merge_{m}/metrics.npz')
            ent[m]['step_loss'] = met['step_loss'].tolist()
    spec = []
    for d in DS:
        e = runj(R16 / f'seed{s}/experts/{d}/run.json')
        bv = e['best_val'] if isinstance(e['best_val'], dict) else eval(e['best_val'])
        spec.append(bv[d]['BAcc'])
        ent.setdefault('expert_best_round', {})[d] = e['best_round']
    ent['specialist'] = dict(best=mean(spec))
    A['val'][s] = ent
for key, name in [('joint', 'Joint'), ('uniform', 'Uniform'), ('global', 'Global'), ('block', 'Block'), ('specialist', 'Specialist')]:
    v = [A['val'][s][key]['best'] for s in SEEDS]
    test = [macro(OFF[s], name) for s in SEEDS]
    supp_test = [mean([float(SUP[s][(name, d)]['BAcc']) for d in DS]) for s in SEEDS]
    step = '–'
    if key in ('global', 'block'):
        step = ' / '.join(str(A['val'][s][key]['best_step']) for s in SEEDS) + ' of 1600'
    elif key == 'joint':
        step = ' / '.join(str(A['val'][s]['joint']['best_round']) for s in SEEDS) + ' of 3000 rounds'
    elif key == 'uniform':
        step = '0 (no optimisation)'
    rows.append([name, f4(mean(v), sd(v)), step, f4(mean(test), sd(test)), f4(mean(supp_test), sd(supp_test))])
T['val'] = md(['Model', 'Best val macro BAcc', 'Selected step', 'Official test (L2)', 'Supplementary test'], rows,
              ['---', '---:', '---:', '---:', '---:'])
A['folded_match'] = max(abs(A['val'][s][m]['folded'] - A['val'][s][m]['best']) for s in SEEDS for m in ['uniform', 'global', 'block'])
# val gain over step 0 and fraction of the final gain reached at the half-way eval
A['val_gain'] = {m: [100 * (A['val'][s][m]['best'] - A['val'][s][m]['curve'][0]) for s in SEEDS] for m in ['global', 'block']}
A['val_last_minus_prev'] = {m: [100 * (A['val'][s][m]['curve'][-1] - A['val'][s][m]['curve'][-2]) for s in SEEDS] for m in ['global', 'block']}
A['val_ce_last_minus_prev'] = {m: [A['val'][s][m]['curve_ce'][-1] - A['val'][s][m]['curve_ce'][-2] for s in SEEDS] for m in ['global', 'block']}
A['expert_best_round'] = {d: [A['val'][s]['expert_best_round'][d] for s in SEEDS] for d in DS}
A['joint_val_minus_block_val'] = [100 * (A['val'][s]['joint']['best'] - A['val'][s]['block']['best']) for s in SEEDS]
T['expert_rounds'] = md(['Dataset', 'Selected round (seed 0/1/2, of 3000)'],
                        [[d, ' / '.join(map(str, A['expert_best_round'][d]))] for d in DS], ['---', '---:'])
A['merge_seconds'] = {m: mean([A['val'][s][m]['seconds'] for s in SEEDS]) for m in ['uniform', 'global', 'block']}
A['joint_seconds'] = mean([A['val'][s]['joint']['seconds'] for s in SEEDS])

# ---------------------------------------------------------------- E. coefficients
groups = runj(R16 / 'seed0/merge_block/run.json')['groups']
groups = groups if isinstance(groups, list) else eval(groups)
G = len(groups)
assert G == 33
GA = np.stack([np.load(R16 / f'seed{s}/merge_global/best_alpha.npy')[:, 0] for s in SEEDS])      # (3, K)
BA = np.stack([np.load(R16 / f'seed{s}/merge_block/best_alpha.npy') for s in SEEDS])             # (3, K, G)
for s in SEEDS:   # expert order in the saved alpha matches DS
    hdr = next(csv.reader(open(R16 / f'seed{s}/merge_block/alpha.csv')))
    assert hdr[1:] == DS, hdr
    gh = [r[0] for r in csv.reader(open(R16 / f'seed{s}/merge_block/alpha.csv'))][1:]
    assert gh == groups
A['global_alpha'] = {d: GA[:, k].tolist() for k, d in enumerate(DS)}
A['global_sum'] = GA.sum(1).tolist()
T['global_alpha'] = md(['Expert', 'α seed 0', 'α seed 1', 'α seed 2', 'Mean ± std', 'Block: mean α over 33 groups (seed mean)',
                        'Task-vector L2 (seed mean)'],
                       [[d, *[f'{v:+.3f}' for v in GA[:, k]], f'{GA[:, k].mean():+.3f} ± {GA[:, k].std(ddof=1):.3f}',
                         f'{BA[:, k, :].mean():+.3f}', f"{mean([A['val'][s]['block']['tv_l2'][d] for s in SEEDS]):.1f}"]
                        for k, d in enumerate(DS)]
                       + [['**Σ over experts**', *[f'**{v:.3f}**' for v in GA.sum(1)], f'**{GA.sum(1).mean():.3f} ± {GA.sum(1).std(ddof=1):.3f}**',
                           f'**{BA.sum(1).mean():.3f}** (mean of per-group sums)', '']])
REG = {'tower.input': 'EEG input', 'tower.mask.outer': 'mask outer'}


def region(g):
    if g.startswith('tower.mask.layer'): return 'mask layers 00–11'
    if g.startswith('tower.next.layer'): return 'next layers 00–11'
    if g.startswith('qformer.layer'): return 'Q-Former layers 00–03'
    return g


REGIONS = list(dict.fromkeys(region(g) for g in groups))
S = BA.sum(1)                       # (3, G) per-group coefficient sums
A['block_sum'] = {g: S[:, i].tolist() for i, g in enumerate(groups)}
rows = []
A['regions'] = {}
for rg in REGIONS:
    idx = [i for i, g in enumerate(groups) if region(g) == rg]
    sub = BA[:, :, idx]
    sums = S[:, idx]
    e = dict(n_groups=len(idx), sum_mean=float(sums.mean()), sum_min=float(sums.min()), sum_max=float(sums.max()),
             sum_seed_std=float(sums.mean(1).std(ddof=1)), frac_neg=float((sub < 0).mean()), max_abs=float(np.abs(sub).max()))
    A['regions'][rg] = e
    rows.append([rg, len(idx), f"{e['sum_mean']:.2f} ± {e['sum_seed_std']:.2f}", f"{e['sum_min']:.2f} … {e['sum_max']:.2f}",
                 f"{100 * e['frac_neg']:.0f} %", f"{e['max_abs']:.2f}"])
T['regions'] = md(['Region', 'Groups', 'Mean Σ_k α (± seed std)', 'Range of Σ_k α (all groups, all seeds)', 'Negative α', 'max \\|α\\|'], rows,
                  ['---', '---:', '---:', '---:', '---:', '---:'])
# extremes, consistent across seeds
A['top_sum_groups'] = [[groups[i] for i in np.argsort(-S[s])[:4]] for s in SEEDS]
A['bottom_sum_groups'] = [[groups[i] for i in np.argsort(S[s])[:3]] for s in SEEDS]
rows = []
for g in ['qformer.output', 'qformer.condition', 'tower.mask.layer11', 'tower.input', 'tower.next.outer']:
    i = groups.index(g)
    sub = BA[:, :, i]
    top = [DS[k] for k in np.argsort(-sub.mean(0))[:3]]
    rows.append([g, ' / '.join(f'{v:.2f}' for v in S[:, i]), ', '.join(f'{SHORT[d]} {sub.mean(0)[DS.index(d)]:+.2f}' for d in top),
                 f'{sub.min():+.2f}'])
T['extreme_groups'] = md(['Group', 'Σ_k α (seed 0/1/2)', 'Largest α (seed mean)', 'Most negative α (any seed)'], rows,
                         ['---', '---:', '---', '---:'])
# cross-seed consistency
def corr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
pairs_s = [(0, 1), (0, 2), (1, 2)]
A['corr_alpha'] = [corr(BA[a], BA[b]) for a, b in pairs_s]
A['corr_sum'] = [corr(S[a], S[b]) for a, b in pairs_s]
A['corr_alpha_centered'] = [corr(BA[a] - BA[a].mean(0, keepdims=True), BA[b] - BA[b].mean(0, keepdims=True)) for a, b in pairs_s]
A['corr_global'] = [corr(GA[a], GA[b]) for a, b in pairs_s]
arg = BA.argmax(1)                  # (3, G) dominant expert per group
A['argmax_agree_all3'] = int(sum(len(set(arg[:, i])) == 1 for i in range(G)))
dom = {}
for i in range(G):
    if len(set(arg[:, i])) == 1:
        dom.setdefault(DS[arg[0, i]], []).append(groups[i])
A['dominant_consistent'] = dom
mean_arg = BA.mean(0).argmax(0)
A['dominant_seedmean_counts'] = {d: int((mean_arg == k).sum()) for k, d in enumerate(DS) if (mean_arg == k).sum()}
A['frac_neg_all'] = float((BA < 0).mean())
A['alpha_range'] = [float(BA.min()), float(BA.max())]
# own-expert weight vs per-dataset outcome
own = BA.mean((0, 2))
gain = np.array([A['perds'][d]['Block']['mean'] - A['perds'][d]['Base']['mean'] for d in DS])
dj = np.array([A['perds'][d]['Block']['dJ'] for d in DS])
tv = np.array([mean([A['val'][s]['block']['tv_l2'][d] for s in SEEDS]) for d in DS])
from scipy.stats import spearmanr
A['spearman_ownalpha_gain'] = float(spearmanr(own, gain).correlation)
A['spearman_ownalpha_dJ'] = float(spearmanr(own, dj).correlation)
A['spearman_alpha_tvnorm'] = float(spearmanr(own, tv).correlation)
A['spearman_globalalpha_tvnorm'] = float(spearmanr(GA.mean(0), tv).correlation)
A['tv_l2'] = {d: float(v) for d, v in zip(DS, tv)}

# ---------------------------------------------------------------- F. K = 5 vs K = 16 on the 5 shared datasets
A['k5k16'] = {}
rows = []
for m in MODELS:
    v5 = [macro(OFF5[s], m, 2, DS5) for s in SEEDS]
    v16 = [macro(OFF[s], m, 2, DS5) for s in SEEDS]
    d = [100 * (b - a) for a, b in zip(v5, v16)]
    A['k5k16'][m] = dict(k5=(mean(v5), sd(v5)), k16=(mean(v16), sd(v16)), diff=(mean(d), sd(d)), k5_per=v5, k16_per=v16)
    rows.append([m, f4(mean(v5), sd(v5)), f4(mean(v16), sd(v16)), pp(mean(d), sd(d))])
T['k5k16'] = md(['Model', 'K = 5 (runs/poc3k)', 'K = 16 (runs/poc16)', 'K16 − K5 (pp, paired by seed)'], rows)
A['k5k16_gapJ'] = {m: dict(k5=mean([100 * (macro(OFF5[s], m, 2, DS5) - macro(OFF5[s], 'Joint', 2, DS5)) for s in SEEDS]),
                           k16=mean([100 * (macro(OFF[s], m, 2, DS5) - macro(OFF[s], 'Joint', 2, DS5)) for s in SEEDS]))
                   for m in MERGES + ['Base', 'Specialist']}
rows = []
for d in DS5:
    r = [d]
    for m in ['Joint', 'Uniform', 'Global', 'Block']:
        a = mean([OFF5[s][(m, d, 2)] for s in SEEDS]); b = mean([OFF[s][(m, d, 2)] for s in SEEDS])
        r.append(f'{a:.4f} → {b:.4f} ({100 * (b - a):+.2f})')
    rows.append(r)
T['k5k16_ds'] = md(['Dataset', 'Joint K5 → K16', 'Uniform K5 → K16', 'Global K5 → K16', 'Block K5 → K16'], rows,
                   ['---', '---:', '---:', '---:', '---:'])
A['k5k16_ds'] = {d: {m: 100 * (mean([OFF[s][(m, d, 2)] for s in SEEDS]) - mean([OFF5[s][(m, d, 2)] for s in SEEDS]))
                     for m in ['Joint', 'Uniform', 'Global', 'Block']} for d in DS5}
# K16 gap to Joint on the 5 shared vs the 11 added datasets
DS11 = [d for d in DS if d not in DS5]
A['gap_shared_vs_added'] = {m: dict(shared=mean([100 * (macro(OFF[s], m, 2, DS5) - macro(OFF[s], 'Joint', 2, DS5)) for s in SEEDS]),
                                    added=mean([100 * (macro(OFF[s], m, 2, DS11) - macro(OFF[s], 'Joint', 2, DS11)) for s in SEEDS]))
                            for m in ['Base'] + MERGES + ['Specialist']}
# task-family summary (official L2)
FAM = {'MI (8)': [d for d in DS if d.startswith('MI_')], 'Emotion (5)': [d for d in DS if d.startswith('EMO_')],
       'SSVEP (1)': ['SSVEP_OpenBMI'], 'Workload + ADHD (2)': ['Workload', 'ADHD_AliMotie']}
A['family'] = {f: {m: dict(mean=mean([macro(OFF[s], m, 2, v) for s in SEEDS]), std=sd([macro(OFF[s], m, 2, v) for s in SEEDS]),
                           dJ=mean([100 * (macro(OFF[s], m, 2, v) - macro(OFF[s], 'Joint', 2, v)) for s in SEEDS]))
                   for m in MODELS} for f, v in FAM.items()}
T['family'] = md(['Family', *MODELS, 'Block − Joint (pp)'],
                 [[f, *[f"{A['family'][f][m]['mean']:.4f}" for m in MODELS], pp(A['family'][f]['Block']['dJ'])] for f in FAM])
# full K5 macro reference (official) and gap on the full K16 sources
A['k5_full'] = {m: mean([macro(OFF5[s], m, 2, DS5) for s in SEEDS]) for m in MODELS}

# ---------------------------------------------------------------- setup table
jr = runj(R16 / 'seed0/joint/run.json')
spl = jr['split_sizes'] if isinstance(jr['split_sizes'], dict) else eval(jr['split_sizes'])
lab = jr['labels'] if isinstance(jr['labels'], dict) else eval(jr['labels'])
ins = jr['instruction'] if isinstance(jr['instruction'], dict) else eval(jr['instruction'])
ntest = {}
for r in csv.DictReader(open(R16 / 'seed1/direct_results.csv')):
    if r['model'] == 'Base' and r['level'] == '2' and r['dataset'] in DS:
        ntest[r['dataset']] = int(r['n'])
T['setup'] = md(['Dataset', 'Classes', 'Train', 'Val', 'Test', 'Task instruction (training)'],
                [[d, len(lab[d]), spl[d]['train'], spl[d]['validation'], ntest[d], ins[d]] for d in DS],
                ['---', '---:', '---:', '---:', '---:', '---'])
A['n_classes'] = {d: len(lab[d]) for d in DS}
A['chance'] = {d: 1 / len(lab[d]) for d in DS}
# fill classes into the per-dataset table
T['perds'] = T['perds'].replace('| None |', '| CLS |')
lines = T['perds'].split('\n')
for i, d in enumerate(DS):
    lines[2 + i] = lines[2 + i].replace('| CLS |', f'| {len(lab[d])} |', 1)
T['perds'] = '\n'.join(lines)

# ================================================================= figures
INK, INK2, MUTED, GRID, BASEL = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COL = {'Block': '#2a78d6', 'Global': '#eb6834', 'Uniform': '#1baf7a', 'Joint': INK, 'Base': MUTED, 'Specialist': INK2}
LS = {'Block': '-', 'Global': '-', 'Uniform': '-', 'Joint': '-', 'Base': ':', 'Specialist': '--'}
MK = {'Block': 'o', 'Global': 's', 'Uniform': '^', 'Joint': 'D', 'Base': 'v', 'Specialist': 'P'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': BASEL, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'savefig.facecolor': '#fcfcfb', 'font.family': 'DejaVu Sans'})


def declutter(ys, gap):
    order = np.argsort(ys); out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out


def grid(ax, axis='both'):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


# Fig 1: main comparison, seed dots + mean
fig, ax = plt.subplots(figsize=(6.4, 2.9))
order_m = sorted(MODELS, key=lambda m: A['main'][m]['mean'])
for y, m in enumerate(order_m):
    e = A['main'][m]
    ax.plot([min(e['per_seed']), max(e['per_seed'])], [y, y], color=COL[m], lw=2, solid_capstyle='round', zorder=2)
    ax.scatter(e['per_seed'], [y] * 3, s=30, color=COL[m], edgecolor='#fcfcfb', linewidth=1.2, zorder=3, marker=MK[m])
    ax.plot([e['mean']] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.2, zorder=4)
    txt = f"{e['mean']:.4f}" + ('' if m == 'Base' else f" ± {e['std']:.4f}") + ('' if m == 'Joint' else f"   ({e['dJ']:+.2f} pp vs Joint)")
    ax.text(max(e['per_seed']) + 0.003, y, txt, va='center', ha='left', fontsize=7.5, color=INK)
ax.set_yticks(range(len(order_m))); ax.set_yticklabels(order_m, color=INK)
ax.set_xlim(0.515, 0.69); grid(ax, 'x')
ax.set_xlabel('Source macro BAcc over 16 datasets (official test, level 2)')
ax.set_title('Fig. 1  Main comparison: 3 seeds (dots), mean (black tick)', loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig1_main.png', dpi=200); plt.close(fig)

# Fig 2: per-dataset Δ vs Joint heatmap
cols = ['Base', 'Uniform', 'Global', 'Block', 'Specialist']
M = np.array([[A['perds'][d][m]['dJ'] for m in cols] for d in DS])
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
lim = 20
fig, ax = plt.subplots(figsize=(5.6, 6.0))
im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
for i in range(len(DS)):
    for j in range(len(cols)):
        v = M[i, j]
        ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=7, color='#ffffff' if abs(v) > 13 else INK)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, color=INK)
ax.set_yticks(range(len(DS))); ax.set_yticklabels([SHORT[d] for d in DS], color=INK)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for y in (7.5, 12.5, 13.5): ax.axhline(y, color='#fcfcfb', lw=2)
for x in np.arange(0.5, len(cols) - 0.5): ax.axvline(x, color='#fcfcfb', lw=2)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label('Δ BAcc vs Joint (pp); red = below Joint', fontsize=7)
cb.outline.set_visible(False)
ax.set_title('Fig. 2  Per-dataset Δ vs Joint (official L2, mean of 3 seeds)', loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig2_delta_heatmap.png', dpi=200); plt.close(fig)

# Fig 3: instruction levels
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for m in MODELS:
    y = [A['levels'][m][lv]['mean'] for lv in (0, 1, 2)]
    e = [A['levels'][m][lv]['std'] for lv in (0, 1, 2)]
    ax.errorbar([0, 1, 2], y, yerr=e, color=COL[m], ls=LS[m], marker=MK[m], ms=5, lw=2, capsize=2, label=m,
                markeredgecolor='#fcfcfb', markeredgewidth=0.8)
ys = declutter([A['levels'][m][2]['mean'] for m in MODELS], 0.0045)
for m, y in zip(MODELS, ys):
    ax.text(2.08, y, f"{m} {A['levels'][m][2]['mean']:.4f}", va='center', fontsize=7.5, color=INK)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['L0 Default', 'L1 Task', 'L2 Task+Target'])
ax.set_xlim(-0.15, 2.75); grid(ax, 'y')
ax.set_ylabel('Source macro BAcc (16 datasets)')
ax.legend(frameon=False, fontsize=7, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.13), handlelength=2.6)
ax.set_title('Fig. 3  Official test across instruction levels (mean ± std)', loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig3_levels.png', dpi=200); plt.close(fig)

# Fig 4: validation curves of the merges
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
seed_ls = {0: '-', 1: '--', 2: ':'}
for m, name in [('global', 'Global'), ('block', 'Block')]:
    for s in SEEDS:
        e = A['val'][s][m]
        axs[0].plot(e['steps'], e['curve'], color=COL[name], ls=seed_ls[s], lw=1.6, marker=MK[name], ms=3.5,
                    label=f'{name} seed {s}')
        axs[0].plot([e['best_step']], [e['best']], marker='*', ms=10, color=COL[name], markeredgecolor='#fcfcfb', zorder=5)
        axs[1].plot(e['steps'], e['curve_ce'], color=COL[name], ls=seed_ls[s], lw=1.6, marker=MK[name], ms=3.5)
jv = mean([A['val'][s]['joint']['best'] for s in SEEDS])
axs[0].axhline(jv, color=INK, lw=1.2, ls='-.')
axs[0].text(20, jv - 0.004, f'Joint best val {jv:.4f}', fontsize=7, color=INK, va='top')
uv = mean([A['val'][s]['uniform']['best'] for s in SEEDS])
axs[0].annotate(f'step 0 = Uniform\n(mean {uv:.4f})', xy=(0, uv), xytext=(300, 0.571), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
axs[0].set_ylim(0.565, 0.685)
axs[0].set_ylabel('Val macro BAcc'); axs[1].set_ylabel('Val macro CE')
for ax in axs: ax.set_xlabel('Coefficient step'); grid(ax)
axs[0].set_title('Fig. 4a  Validation BAcc (★ = selected)', loc='left'); axs[1].set_title('Fig. 4b  Validation CE', loc='left')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / 'fig4_val_curves.png', dpi=200); plt.close(fig)

# Fig 5: block coefficients (seed mean) + per-group sums
BM = BA.mean(0).T                    # (G, K)
fig = plt.figure(figsize=(9.2, 7.8))
ax = fig.add_axes([0.17, 0.19, 0.52, 0.76]); ax2 = fig.add_axes([0.76, 0.19, 0.2, 0.76], sharey=ax)
lim = float(np.abs(BM).max())
im = ax.imshow(BM, cmap=LinearSegmentedColormap.from_list('div2', ['#e34948', '#f0efec', '#2a78d6']), norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
ax.set_xticks(range(K)); ax.set_xticklabels([SHORT[d] for d in DS], rotation=60, ha='right', color=INK)
ax.set_yticks(range(G)); ax.set_yticklabels(groups, fontsize=6.5, color=INK); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for y in (1.5, 13.5, 14.5, 26.5, 27.5, 31.5): ax.axhline(y, color='#fcfcfb', lw=2)
cax = fig.add_axes([0.17, 0.065, 0.52, 0.012])
cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
cb.set_label('α_{k,g}, mean of 3 seeds (init 1/16 = 0.0625; red = negative)', fontsize=7)
for s in SEEDS:
    ax2.plot(S[s], range(G), color=COL['Block'], ls=seed_ls[s], lw=1.4, marker='o', ms=2.5, label=f'seed {s}')
ax2.axvline(1.0, color=MUTED, lw=1, ls=':')
ax2.axvline(GA.sum(1).mean(), color=COL['Global'], lw=1.2, ls='-.', label=f'Global Σα ({GA.sum(1).mean():.2f})')
ax2.plot([], [], color=MUTED, lw=1, ls=':', label='Σα = 1')
grid(ax2, 'x'); ax2.set_xlabel('Σ_k α_{k,g} per group'); plt.setp(ax2.get_yticklabels(), visible=False)
ax2.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.45, -0.07), ncol=2)
ax.set_title('Fig. 5a  Block coefficients α (groups × experts)', loc='left'); ax2.set_title('Fig. 5b  Per-group sum', loc='left')
fig.savefig(FIG / 'fig5_block_alpha.png', dpi=200); plt.close(fig)

# Fig 6: K = 5 vs K = 16 on the 5 shared datasets
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for m in ['Specialist', 'Joint', 'Block', 'Global', 'Uniform', 'Base']:
    e = A['k5k16'][m]
    ax.errorbar([0, 1], [e['k5'][0], e['k16'][0]], yerr=[e['k5'][1], e['k16'][1]], color=COL[m], ls=LS[m], marker=MK[m],
                ms=6, lw=2, capsize=2, label=m, markeredgecolor='#fcfcfb', markeredgewidth=0.8)

ms6 = ['Specialist', 'Joint', 'Block', 'Global', 'Uniform', 'Base']
ys = declutter([A['k5k16'][m]['k16'][0] for m in ms6], 0.0055)
for m, y in zip(ms6, ys):
    ax.text(1.07, y, f"{m} {A['k5k16'][m]['k16'][0]:.4f} ({z(A['k5k16'][m]['diff'][0]):+.2f} pp)", va='center', fontsize=7.5, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(['K = 5\n(runs/poc3k)', 'K = 16\n(runs/poc16)'])
ax.set_xlim(-0.2, 2.0); grid(ax, 'y')
ax.set_ylabel('Macro BAcc on the 5 shared datasets')
ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))
ax.set_title('Fig. 6  Same 5 experts, 11 more added (official L2)', loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig6_k5_vs_k16.png', dpi=200); plt.close(fig)

# Fig 7: gap decomposition for Block
fig, ax = plt.subplots(figsize=(5.6, 4.0))
vals = [A['gap']['Block']['contrib_pp'][d] for d in order]
errs = [A['perds'][d]['Block']['dJ_std'] / K for d in order]
ax.barh(range(K), vals, xerr=errs, color=COL['Block'], height=0.7, error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
ax.axvline(0, color=BASEL, lw=1)
for i, (d, v, er) in enumerate(zip(order, vals, errs)):
    sh = 100 * A['gap']['Block']['share'][d]
    ax.text(max(v, 0) + er + 0.04, i, f"{z(v):+.2f} pp ({'0' if abs(sh) < 0.5 else f'{sh:.0f}'} %)", va='center', fontsize=7, color=INK)
ax.set_yticks(range(K)); ax.set_yticklabels([SHORT[d] for d in order], color=INK); ax.invert_yaxis()
ax.set_xlim(-0.35, 1.6); grid(ax, 'x')
ax.set_xlabel('Contribution to the macro gap Joint − Block (pp) = per-dataset gap / 16')
ax.set_title(f"Fig. 7  Where Block's {A['gap']['Block']['total_pp']:.2f} pp gap to Joint comes from", loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig7_gap_decomposition.png', dpi=200); plt.close(fig)

json.dump(dict(numbers=A, tables=T), open(OUT, 'w'), indent=1, default=float)
print('wrote', OUT, 'and', sorted(p.name for p in FIG.glob('*.png')))
