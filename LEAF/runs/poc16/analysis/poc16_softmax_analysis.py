"""Analysis + figures for the softmax round on the 16-source, 3-seed LEAF merging POC (runs/poc16).

Softmax-Global / Softmax-Block were merged from each seed's existing 16 experts with the same calibration
subsets, budget and selection rule as the unconstrained Global / Block merges (runs/poc16/chain_softmax.sh).
Reads only run artifacts; writes figures 8-11 to runs/poc16/figures/ and numbers + markdown tables to
poc16_softmax_analysis.json next to this script.
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path('/home/jingzhou/BFM-merge/LEAF')
R16, R5, R1K = ROOT / 'runs/poc16', ROOT / 'runs/poc3k', ROOT / 'runs/poc'
FIG = R16 / 'figures'; FIG.mkdir(exist_ok=True)
OUT = Path(__file__).with_name('poc16_softmax_analysis.json')
SEEDS = [0, 1, 2]
DS = ['MI_OpenBMI', 'MI_BCIC_IV2a', 'MI_ShanghaiU', 'MI_BCIC_Upperlimb', 'MI_HighGamma', 'MI_Cho2017', 'MI_Shin2017A',
      'MI_PhysioNet', 'EMO_FACED', 'EMO_SEED_3_seg4', 'EMO_SEED_4_seg4', 'EMO_SEED_5_seg4', 'EMO_SEED_7_seg4',
      'SSVEP_OpenBMI', 'Workload', 'ADHD_AliMotie']
DS5 = ['MI_OpenBMI', 'MI_Cho2017', 'MI_PhysioNet', 'EMO_SEED_3_seg4', 'SSVEP_OpenBMI']
SHORT = {'MI_OpenBMI': 'OpenBMI', 'MI_BCIC_IV2a': 'BCIC-IV2a', 'MI_ShanghaiU': 'ShanghaiU', 'MI_BCIC_Upperlimb': 'Upperlimb',
         'MI_HighGamma': 'HighGamma', 'MI_Cho2017': 'Cho2017', 'MI_Shin2017A': 'Shin2017A', 'MI_PhysioNet': 'PhysioNet',
         'EMO_FACED': 'FACED', 'EMO_SEED_3_seg4': 'SEED-3', 'EMO_SEED_4_seg4': 'SEED-4', 'EMO_SEED_5_seg4': 'SEED-5',
         'EMO_SEED_7_seg4': 'SEED-7', 'SSVEP_OpenBMI': 'SSVEP', 'Workload': 'Workload', 'ADHD_AliMotie': 'ADHD'}
MODELS = ['Base', 'Joint', 'Uniform', 'Global', 'Softmax-Global', 'Block', 'Softmax-Block', 'Specialist']
MERGES = ['Uniform', 'Global', 'Softmax-Global', 'Block', 'Softmax-Block']
DIR = {'Global': 'global', 'Block': 'block', 'Softmax-Global': 'softmax_global', 'Softmax-Block': 'softmax_block',
       'Uniform': 'uniform'}
K = len(DS)
A, T = {}, {}


def mean(x): return float(np.mean(x))
def sd(x): return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
def f4(m, s): return f'{m:.4f} ± {s:.4f}'
def z(m, nd=2): return 0.0 if round(m, nd) == 0 else m
def pp(m, s=None): return f'{z(m):+.2f}' if s is None else f'{z(m):+.2f} ± {s:.2f}'
def runj(p): return json.loads(Path(p).read_text())


def md(header, rows, align=None):
    align = align or ['---'] + ['---:'] * (len(header) - 1)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ---------------------------------------------------------------- official tables (with softmax rows)
def load_direct(s):
    out = {}
    for r in csv.DictReader(open(R16 / f'seed{s}/direct_results_softmax.csv')):
        assert r['role'] == 'source', r
        out[(r['model'], r['dataset'], int(r['level']))] = float(r['BAcc'])
    assert set(d for (_, d, _) in out) == set(DS) | {'SOURCE_MACRO'}
    for m in MODELS:
        for lv in (0, 1, 2):
            assert all((m, d, lv) in out for d in DS), (s, m, lv)
            assert abs(out[(m, 'SOURCE_MACRO', lv)] - np.mean([out[(m, d, lv)] for d in DS])) < 2e-4
    # the non-softmax rows are the rows of the main 3-seed analysis
    ref = R16 / f'seed{s}' / ('direct_results_16sources.csv' if s == 0 else 'direct_results.csv')
    for r in csv.DictReader(open(ref)):
        assert out[(r['model'], r['dataset'], int(r['level']))] == float(r['BAcc'])
    return out


OFF = {s: load_direct(s) for s in SEEDS}
def macro(s, m, lv=2): return OFF[s][(m, 'SOURCE_MACRO', lv)]


A['main'] = {}
for m in MODELS:
    per = [macro(s, m) for s in SEEDS]
    dj = [100 * (macro(s, m) - macro(s, 'Joint')) for s in SEEDS]
    db = [100 * (macro(s, m) - macro(s, 'Base')) for s in SEEDS]
    A['main'][m] = dict(per_seed=per, mean=mean(per), std=sd(per), dJ=mean(dj), dJ_std=sd(dj), dB=mean(db), dB_std=sd(db))
A['recovery'] = {m: mean([(macro(s, m) - macro(s, 'Base')) / (macro(s, 'Joint') - macro(s, 'Base')) for s in SEEDS])
                 for m in MERGES}
A['recovery_per_seed'] = {m: [(macro(s, m) - macro(s, 'Base')) / (macro(s, 'Joint') - macro(s, 'Base')) for s in SEEDS]
                          for m in MERGES}
A['rank_per_seed'] = {s: sorted(MODELS, key=lambda m: -macro(s, m)) for s in SEEDS}
LABEL = {'Specialist': 'Specialist (16 experts, reference)'}
T['sm_main'] = md(['Model', 'Source macro BAcc', 'seed 0', 'seed 1', 'seed 2', 'Δ vs Joint (pp)', 'Δ vs Base (pp)',
                   'Share of Joint − Base recovered'],
                  [[('**' + m + '**') if m.startswith('Softmax') else LABEL.get(m, m), f4(A['main'][m]['mean'], A['main'][m]['std']),
                    *[f'{v:.4f}' for v in A['main'][m]['per_seed']], pp(A['main'][m]['dJ'], A['main'][m]['dJ_std']),
                    pp(A['main'][m]['dB'], A['main'][m]['dB_std']),
                    f"{100 * A['recovery'][m]:.0f} %" if m in MERGES else '']
                   for m in sorted(MODELS, key=lambda m: -A['main'][m]['mean'])])

# levels
LVN = {0: 'L0 Default', 1: 'L1 Task', 2: 'L2 Task+Target (main)'}
A['levels'] = {m: {lv: dict(mean=mean([macro(s, m, lv) for s in SEEDS]), std=sd([macro(s, m, lv) for s in SEEDS]),
                            dJ=mean([100 * (macro(s, m, lv) - macro(s, 'Joint', lv)) for s in SEEDS]))
                   for lv in (0, 1, 2)} for m in MODELS}
T['sm_levels'] = md(['Model'] + [LVN[lv] for lv in (0, 1, 2)] + [f'Δ vs Joint, L{lv} (pp)' for lv in (0, 1, 2)],
                    [[m] + [f4(A['levels'][m][lv]['mean'], A['levels'][m][lv]['std']) for lv in (0, 1, 2)]
                     + [pp(A['levels'][m][lv]['dJ']) for lv in (0, 1, 2)]
                     for m in ['Joint', 'Block', 'Softmax-Block', 'Softmax-Global', 'Global', 'Base', 'Uniform']])

# paired comparisons (L2), and the two constraint effects at every level
PAIRS = [('Softmax-Block', 'Block'), ('Softmax-Global', 'Global'), ('Softmax-Block', 'Softmax-Global'),
         ('Block', 'Global'), ('Softmax-Block', 'Base'), ('Softmax-Global', 'Base'), ('Softmax-Global', 'Uniform'),
         ('Softmax-Block', 'Joint')]
A['pairs'] = {}
rows = []
for a, b in PAIRS:
    d = [100 * (macro(s, a) - macro(s, b)) for s in SEEDS]
    wins = [sum(OFF[s][(a, x, 2)] > OFF[s][(b, x, 2)] for x in DS) for s in SEEDS]
    dm = {x: mean([OFF[s][(a, x, 2)] - OFF[s][(b, x, 2)] for s in SEEDS]) for x in DS}
    A['pairs'][f'{a}|{b}'] = dict(per_seed=d, mean=mean(d), std=sd(d), wins=wins, wins_seedmean=sum(v > 0 for v in dm.values()),
                                  same_sign=all(np.sign(v) == np.sign(d[0]) for v in d))
    rows.append([f'{a} − {b}', *[f'{z(v):+.2f}' for v in d], pp(mean(d), sd(d)), '/'.join(map(str, wins)) + ' of 16',
                 f'{sum(v > 0 for v in dm.values())} of 16'])
T['sm_pairs'] = md(['Comparison (L2)', 'seed 0', 'seed 1', 'seed 2', 'Mean ± std', 'Datasets improved (per seed)',
                    'Datasets improved (seed mean)'], rows)
A['constraint_levels'] = {}
rows = []
for a, b in [('Softmax-Block', 'Block'), ('Softmax-Global', 'Global'), ('Softmax-Block', 'Softmax-Global'), ('Block', 'Global')]:
    e = {}
    for lv in (0, 1, 2):
        d = [100 * (macro(s, a, lv) - macro(s, b, lv)) for s in SEEDS]
        e[lv] = dict(mean=mean(d), std=sd(d), per_seed=d)
    A['constraint_levels'][f'{a}|{b}'] = e
    rows.append([f'{a} − {b}', *[pp(e[lv]['mean'], e[lv]['std']) for lv in (0, 1, 2)]])
T['sm_constraint_levels'] = md(['Comparison (pp)', 'L0 Default', 'L1 Task', 'L2 Task+Target'], rows)

# per dataset (L2, seed mean)
A['perds'] = {}
rows = []
for d in DS:
    v = {m: [OFF[s][(m, d, 2)] for s in SEEDS] for m in MODELS}
    sbb = [100 * (OFF[s][('Softmax-Block', d, 2)] - OFF[s][('Block', d, 2)]) for s in SEEDS]
    sgg = [100 * (OFF[s][('Softmax-Global', d, 2)] - OFF[s][('Global', d, 2)]) for s in SEEDS]
    A['perds'][d] = dict(**{m: mean(v[m]) for m in MODELS}, sbb=mean(sbb), sbb_std=sd(sbb), sgg=mean(sgg), sgg_std=sd(sgg),
                         sbb_same_sign=all(np.sign(x) == np.sign(sbb[0]) for x in sbb))
    e = A['perds'][d]
    rows.append([d, f"{e['Base']:.4f}", f"{e['Joint']:.4f}", f"{e['Block']:.4f}", f"{e['Softmax-Block']:.4f}",
                 pp(e['sbb'], e['sbb_std']), f"{e['Global']:.4f}", f"{e['Softmax-Global']:.4f}", pp(e['sgg'], e['sgg_std'])])
tot_sbb = sum(A['perds'][d]['sbb'] for d in DS) / K
tot_sgg = sum(A['perds'][d]['sgg'] for d in DS) / K
rows.append(['**Macro**', *[f"{A['main'][m]['mean']:.4f}" for m in ['Base', 'Joint', 'Block', 'Softmax-Block']],
             pp(tot_sbb), *[f"{A['main'][m]['mean']:.4f}" for m in ['Global', 'Softmax-Global']], pp(tot_sgg)])
T['sm_perds'] = md(['Dataset', 'Base', 'Joint', 'Block', 'Softmax-Block', 'SB − B (pp)', 'Global', 'Softmax-Global',
                    'SG − G (pp)'], rows)
neg = sorted(DS, key=lambda d: A['perds'][d]['sbb'])
A['sbb_total'] = tot_sbb
A['sbb_share'] = {d: (A['perds'][d]['sbb'] / K) / tot_sbb for d in DS}
A['sbb_top4'] = neg[:4]
A['sbb_top4_share'] = sum(A['sbb_share'][d] for d in neg[:4])
A['sbb_counts'] = dict(sb_gt_b=sum(A['perds'][d]['sbb'] > 0 for d in DS), sg_gt_g=sum(A['perds'][d]['sgg'] > 0 for d in DS),
                       sb_gt_base=sum(A['perds'][d]['Softmax-Block'] > A['perds'][d]['Base'] for d in DS),
                       b_gt_base=sum(A['perds'][d]['Block'] > A['perds'][d]['Base'] for d in DS))

# ---------------------------------------------------------------- supplementary protocol + Weibo
def load_supp(s):
    out = {}
    for f in ('results.csv', 'results_softmax.csv'):
        for r in csv.DictReader(open(R16 / f'seed{s}' / f)):
            key = (r['model'], r['dataset'])
            vals = {k: float(r[k]) for k in ('BAcc', 'Kappa', 'Auroc', 'wF1', 'Loss')}
            if key in out:   # Base and Joint were re-evaluated in the softmax round: must agree exactly
                assert out[key] == vals, key
            out[key] = vals
    return out


SUP = {s: load_supp(s) for s in SEEDS}
MET = [('BAcc', 'BAcc'), ('Kappa', 'Kappa'), ('Auroc', 'AUROC'), ('wF1', 'wF1'), ('Loss', 'CE')]
A['supp'] = {m: {k: dict(mean=mean([SUP[s][(m, 'SOURCE_MACRO')][k] for s in SEEDS]),
                         std=sd([SUP[s][(m, 'SOURCE_MACRO')][k] for s in SEEDS])) for k, _ in MET} for m in MODELS}
T['sm_supp'] = md(['Model', *[n for _, n in MET]],
                  [[m, *[f4(A['supp'][m][k]['mean'], A['supp'][m][k]['std']) for k, _ in MET]]
                   for m in sorted(MODELS, key=lambda m: -A['supp'][m]['BAcc']['mean'])])
WM = [m for m in MODELS if m != 'Specialist']
A['weibo'] = {m: {k: dict(mean=mean([SUP[s][(m, 'MI_Weibo2014')][k] for s in SEEDS]),
                          std=sd([SUP[s][(m, 'MI_Weibo2014')][k] for s in SEEDS])) for k, _ in MET} for m in WM}
T['sm_weibo'] = md(['Model', 'BAcc', 'Kappa', 'AUROC', 'CE'],
                   [[m, *[f4(A['weibo'][m][k]['mean'], A['weibo'][m][k]['std']) for k in ('BAcc', 'Kappa', 'Auroc', 'Loss')]]
                    for m in sorted(WM, key=lambda m: -A['weibo'][m]['BAcc']['mean'])])

# ---------------------------------------------------------------- validation / selection
A['val'] = {}
rows = []
for m in ['Global', 'Softmax-Global', 'Block', 'Softmax-Block']:
    A['val'][m] = {}
    for s in SEEDS:
        d = R16 / f'seed{s}' / f'merge_{DIR[m]}'
        h, r = runj(d / 'history.json'), runj(d / 'run.json')
        curve = [e['val_macro_BAcc'] for e in h]
        assert abs(r['folded_val_macro_BAcc'] - r['best_val_macro_BAcc']) < 1e-9
        A['val'][m][s] = dict(steps=[e['step'] for e in h], curve=curve, curve_ce=[e['val_macro_Loss'] for e in h],
                              best_step=r['best_step'], best=r['best_val_macro_BAcc'], last=curve[-1],
                              share160=(curve[1] - curve[0]) / (max(curve) - curve[0]),
                              gain_after160=100 * (max(curve) - curve[1]),
                              rising_tail=bool(curve[-1] >= curve[-3] and curve[-2] >= curve[-4]))
    e = A['val'][m]
    rows.append([m, ' / '.join(str(e[s]['best_step']) for s in SEEDS), ' / '.join(f"{e[s]['best']:.4f}" for s in SEEDS),
                 ' / '.join(f"{100 * (e[s]['curve'][1] - e[s]['curve'][0]):+.2f}" for s in SEEDS),
                 ' / '.join(f"{100 * e[s]['share160']:.0f} %" for s in SEEDS),
                 ' / '.join(f"{e[s]['gain_after160']:+.2f}" for s in SEEDS)])
T['sm_val'] = md(['Merge', 'Selected step (s0/s1/s2)', 'Selected val BAcc', 'Gain at step 160 (pp)',
                  'Share of final gain at step 160', 'Gain after step 160 (pp)'], rows,
                 ['---', '---:', '---:', '---:', '---:', '---:'])
A['val_mean_best'] = {m: mean([A['val'][m][s]['best'] for s in SEEDS]) for m in A['val']}
# monotonicity of Softmax-Block's validation curve
A['sb_val_increments'] = {s: int(sum(b > a for a, b in zip(A['val']['Softmax-Block'][s]['curve'][:-1],
                                                          A['val']['Softmax-Block'][s]['curve'][1:]))) for s in SEEDS}

# ---------------------------------------------------------------- learned weights
names = runj(R16 / 'seed0/merge_block/run.json')['datasets']; assert names == DS
groups = runj(R16 / 'seed0/merge_block/run.json')['groups']; G = len(groups)
SG = np.stack([np.load(R16 / f'seed{s}/merge_softmax_global/best_alpha.npy')[:, 0] for s in SEEDS])   # (3, K)
GA = np.stack([np.load(R16 / f'seed{s}/merge_global/best_alpha.npy')[:, 0] for s in SEEDS])
SB = np.stack([np.load(R16 / f'seed{s}/merge_softmax_block/best_alpha.npy') for s in SEEDS])            # (3, K, G)
BA = np.stack([np.load(R16 / f'seed{s}/merge_block/best_alpha.npy') for s in SEEDS])
assert np.allclose(SG.sum(1), 1, atol=1e-5) and np.allclose(SB.sum(1), 1, atol=1e-5) and (SB > 0).all()
for s in SEEDS:
    r = runj(R16 / f'seed{s}/merge_softmax_block/run.json')
    assert np.allclose(np.array(r['best_alpha']), SB[s])
def corr(a, b): return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
PAIRS_S = [(0, 1), (0, 2), (1, 2)]
A['sg_alpha'] = dict(per_expert={d: SG[:, k].tolist() for k, d in enumerate(DS)}, min=float(SG.min()), max=float(SG.max()),
                     eff=[float(1 / np.sum(a ** 2)) for a in SG], r_seeds=[corr(SG[i], SG[j]) for i, j in PAIRS_S],
                     r_vs_global=[corr(SG[s], GA[s]) for s in SEEDS])
T['sm_sg_alpha'] = md(['Expert', 'Softmax-Global α (s0 / s1 / s2)', 'mean', 'Global α (unconstrained), mean'],
                      [[d, ' / '.join(f'{x:.3f}' for x in SG[:, k]), f'{SG[:, k].mean():.3f}', f'{GA[:, k].mean():+.3f}']
                       for k, d in enumerate(DS)] +
                      [['**Sum**', ' / '.join(f'{x:.3f}' for x in SG.sum(1)), f'{SG.sum(1).mean():.3f}', f'{GA.sum(1).mean():+.3f}']],
                      ['---', '---:', '---:', '---:'])
EFF = 1 / np.sum(SB ** 2, axis=1)       # (3, G) effective number of experts per group
MX = SB.max(1)
top = [[DS[int(SB[s, :, g].argmax())] for s in SEEDS] for g in range(G)]
A['sb_alpha'] = dict(min=float(SB.min()), max=float(SB.max()), eff_mean=float(EFF.mean()), eff_min=float(EFF.min()),
                     eff_max=float(EFF.max()), maxw_mean=float(MX.mean()),
                     top_agree=int(sum(len(set(t)) == 1 for t in top)),
                     r_seeds=[corr(SB[i], SB[j]) for i, j in PAIRS_S], r_block_seeds=[corr(BA[i], BA[j]) for i, j in PAIRS_S],
                     r_vs_block=[corr(SB[s], BA[s]) for s in SEEDS],
                     groups_maxw_gt_half=int(sum(MX.mean(0) > 0.5)))
T['sm_sb_groups'] = md(['Group', 'Effective # experts (s0/s1/s2)', 'Largest weight (s0/s1/s2)', 'Top expert (s0/s1/s2)'],
                       [[g, ' / '.join(f'{x:.1f}' for x in EFF[:, i]), ' / '.join(f'{x:.2f}' for x in MX[:, i]),
                         ' / '.join(SHORT[t] for t in top[i]) + (' ✓' if len(set(top[i])) == 1 else '')]
                        for i, g in enumerate(groups)], ['---', '---:', '---:', '---'])
# weight by expert family and model region
FAM = {'MI (8 experts)': [d for d in DS if d.startswith('MI_')], 'Emotion (5)': [d for d in DS if d.startswith('EMO_')],
       'SSVEP (1)': ['SSVEP_OpenBMI'], 'Workload + ADHD (2)': ['Workload', 'ADHD_AliMotie']}
REG = {'tower input + mask branch (14)': [i for i, g in enumerate(groups) if g == 'tower.input' or g.startswith('tower.mask')],
       'tower next branch (13)': [i for i, g in enumerate(groups) if g.startswith('tower.next')],
       'Q-Former (6)': [i for i, g in enumerate(groups) if g.startswith('qformer')]}
A['sb_family'] = {f: {reg: float(SB[:, [DS.index(d) for d in fs], :][:, :, gi].sum(1).mean()) for reg, gi in REG.items()}
                  for f, fs in FAM.items()}
A['sb_family_uniform'] = {f: len(fs) / K for f, fs in FAM.items()}
T['sm_sb_family'] = md(['Expert family', 'Uniform share', *REG.keys()],
                       [[f, f'{A["sb_family_uniform"][f]:.3f}', *[f'{A["sb_family"][f][r]:.3f}' for r in REG]] for f in FAM])
A['sb_mean_weight'] = {d: float(SB[:, k, :].mean()) for k, d in enumerate(DS)}

# ---------------------------------------------------------------- K = 5 reference (round 2, runs/poc) and Block Σα vs K
def macro_file(path, model, lv=2):
    for r in csv.DictReader(open(path)):
        if r['model'] == model and r['dataset'] == 'SOURCE_MACRO' and r['level'] == str(lv):
            return float(r['BAcc'])
    raise KeyError(model)
def macro_raw(path, lv=2):
    t = [r for r in csv.DictReader(open(path)) if r['dataset'] != 'MEAN']
    assert sorted(r['dataset'] for r in t) == sorted(DS5)
    return mean([float(r[f'level{lv}_bAcc']) for r in t])
k5 = {m: macro_file(R1K / 'direct_results.csv', m) for m in ['Base', 'Joint', 'Global', 'Block']}
k5['Softmax-Global'] = macro_raw(R1K / 'direct_inference/softmax_global.csv')
k5['Softmax-Block'] = macro_raw(R1K / 'direct_inference/softmax_block.csv')
A['k5_round2'] = k5
A['k5_round2_d'] = {'SB-B': 100 * (k5['Softmax-Block'] - k5['Block']), 'SG-G': 100 * (k5['Softmax-Global'] - k5['Global'])}
B3 = np.stack([np.load(R5 / f'seed{s}/merge_block/best_alpha.npy') for s in SEEDS])     # K = 5, same 5 experts as poc16
B1 = np.load(R1K / 'merge_block/best_alpha.npy')
A['block_sum'] = {'poc16 (K=16, 3k)': dict(mean=float(BA.sum(1).mean()), max=float(BA.sum(1).max()), min=float(BA.sum(1).min()),
                                         frac_neg=float((BA < 0).mean())),
                  'poc3k (K=5, 3k)': dict(mean=float(B3.sum(1).mean()), max=float(B3.sum(1).max()), min=float(B3.sum(1).min()),
                                         frac_neg=float((B3 < 0).mean())),
                  'poc (K=5, 1k)': dict(mean=float(B1.sum(0).mean()), max=float(B1.sum(0).max()), min=float(B1.sum(0).min()),
                                       frac_neg=float((B1 < 0).mean()))}
T['sm_k'] = md(['Setting', 'Experts', 'Seeds', 'SB − B (pp)', 'SG − G (pp)', 'Block: mean per-group Σα', 'Block: max per-group Σα',
                'Block: share of α < 0'],
               [['`runs/poc` (round 2)', '5, 1,000 steps', '1', pp(A['k5_round2_d']['SB-B']), pp(A['k5_round2_d']['SG-G']),
                 f"{A['block_sum']['poc (K=5, 1k)']['mean']:.2f}", f"{A['block_sum']['poc (K=5, 1k)']['max']:.2f}",
                 f"{100 * A['block_sum']['poc (K=5, 1k)']['frac_neg']:.0f} %"],
                ['`runs/poc3k`', '5, 3,000 steps', '3', 'not run', 'not run',
                 f"{A['block_sum']['poc3k (K=5, 3k)']['mean']:.2f}", f"{A['block_sum']['poc3k (K=5, 3k)']['max']:.2f}",
                 f"{100 * A['block_sum']['poc3k (K=5, 3k)']['frac_neg']:.0f} %"],
                ['`runs/poc16` (this round)', '16, 3,000 steps', '3', pp(A['pairs']['Softmax-Block|Block']['mean']),
                 pp(A['pairs']['Softmax-Global|Global']['mean']),
                 f"{A['block_sum']['poc16 (K=16, 3k)']['mean']:.2f}", f"{A['block_sum']['poc16 (K=16, 3k)']['max']:.2f}",
                 f"{100 * A['block_sum']['poc16 (K=16, 3k)']['frac_neg']:.0f} %"]],
               ['---', '---', '---:', '---:', '---:', '---:', '---:', '---:'])

# ---------------------------------------------------------------- runtime
A['runtime_s'] = {m: [runj(R16 / f'seed{s}/merge_{DIR[m]}/run.json')['seconds']['total'] for s in SEEDS]
                  for m in ['Softmax-Global', 'Softmax-Block']}

# ================================================================= figures
INK, INK2, MUTED, GRID, BASEL, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
HUE = {'Block': BLUE, 'Softmax-Block': BLUE, 'Global': ORANGE, 'Softmax-Global': ORANGE}
FILLED = {'Block': True, 'Global': True, 'Softmax-Block': False, 'Softmax-Global': False}
MK = {'Block': 'o', 'Softmax-Block': 'o', 'Global': 's', 'Softmax-Global': 's'}
LS = {'Block': '-', 'Global': '-', 'Softmax-Block': '--', 'Softmax-Global': '--'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': BASEL, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans'})
def grid(ax, axis='both'): ax.grid(True, axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
def mstyle(m): return dict(marker=MK[m], color=HUE[m], markerfacecolor=HUE[m] if FILLED[m] else SURF, markeredgecolor=HUE[m],
                           markeredgewidth=1.4)

# Fig 8: (a) L2 macro with seed dots, unconstrained (filled) vs softmax (hollow), Joint/Base references; (b) constraint effect by level
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
rowsy = {'Block': 3, 'Softmax-Block': 2.4, 'Global': 1, 'Softmax-Global': 0.4}
for m, y in rowsy.items():
    e = A['main'][m]
    ax.plot([min(e['per_seed']), max(e['per_seed'])], [y, y], color=HUE[m], lw=1.6, ls=LS[m], zorder=2)
    ax.scatter(e['per_seed'], [y] * 3, s=26, zorder=3, **{k: v for k, v in mstyle(m).items() if k == 'marker'},
               facecolor=HUE[m] if FILLED[m] else SURF, edgecolor=HUE[m], linewidth=1.3)
    ax.plot([e['mean']] * 2, [y - 0.2, y + 0.2], color=INK, lw=1.2, zorder=4)
    ax.text(0.6045, y, f"{m}  {e['mean']:.4f}  ({e['dJ']:+.2f} pp vs Joint)", va='center', fontsize=7.3, color=INK)
for m, ls in [('Joint', '-.'), ('Base', ':')]:
    v = A['main'][m]['mean']
    ax.axvline(v, color=INK if m == 'Joint' else MUTED, lw=1.1, ls=ls, zorder=1)
    ax.text(v + 0.0008, 3.6, f'{m} {v:.4f}', ha='left', va='bottom', fontsize=7, color=INK)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlim(0.535, 0.665); ax.set_ylim(0.0, 3.95); grid(ax, 'x')
ax.set_xlabel('Source macro BAcc, 16 datasets (official test, L2)')
ax.set_title('Fig. 8a  Unconstrained (filled) vs softmax (hollow): 3 seeds + mean', loc='left')
ax = axs[1]
w = 0.34
for j, (key, hue, lab) in enumerate([('Softmax-Global|Global', ORANGE, 'Softmax-Global − Global'),
                                     ('Softmax-Block|Block', BLUE, 'Softmax-Block − Block')]):
    e = A['constraint_levels'][key]
    xs = np.arange(3) + (j - 0.5) * w
    vals = [e[lv]['mean'] for lv in (0, 1, 2)]
    ax.bar(xs, vals, width=w - 0.04, color=hue, label=lab, zorder=2)
    for x, lv in zip(xs, (0, 1, 2)):
        ax.scatter([x] * 3, e[lv]['per_seed'], s=9, color=INK, zorder=3, linewidth=0)
        v = e[lv]['mean']
        ax.text(x, (min(e[lv]['per_seed']) - 0.12) if v < 0 else (max(e[lv]['per_seed']) + 0.08), f'{z(v):+.2f}',
                ha='center', va='top' if v < 0 else 'bottom', fontsize=7, color=INK)
ax.axhline(0, color=BASEL, lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(['L0 Default', 'L1 Task', 'L2 Task+Target'])
ax.set_ylim(-2.6, 0.9); grid(ax, 'y'); ax.set_ylabel('Δ macro BAcc (pp), paired per seed')
ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.set_title('Fig. 8b  Effect of the softmax constraint', loc='left')
fig.tight_layout(); fig.savefig(FIG / 'fig8_softmax_main.png', dpi=200); plt.close(fig)

# Fig 9: per-dataset effect of the constraint (small multiples, one hue each)
order = sorted(DS, key=lambda d: A['perds'][d]['sbb'])
fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.4), sharey=True)
for ax, key, hue, title in [(axs[0], 'sbb', BLUE, 'Fig. 9a  Softmax-Block − Block'),
                            (axs[1], 'sgg', ORANGE, 'Fig. 9b  Softmax-Global − Global')]:
    vals = [A['perds'][d][key] for d in order]
    errs = [A['perds'][d][key + '_std'] for d in order]
    ax.barh(range(K), vals, xerr=errs, color=hue, height=0.68, error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5), zorder=2)
    ax.axvline(0, color=BASEL, lw=1)
    for i, (v, er) in enumerate(zip(vals, errs)):
        ax.text((v - er - 0.3) if v < 0 else (v + er + 0.3), i, f'{z(v):+.1f}', va='center',
                ha='right' if v < 0 else 'left', fontsize=7, color=INK)
    ax.set_xlim(-13, 10); grid(ax, 'x'); ax.set_title(title, loc='left')
axs[0].set_yticks(range(K)); axs[0].set_yticklabels([SHORT[d] for d in order], color=INK); axs[0].invert_yaxis()
fig.supxlabel('Δ BAcc (pp), official L2, sorted by Softmax-Block − Block; bar = seed mean, whisker = seed std', fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig9_softmax_perds.png', dpi=200); plt.close(fig)

# Fig 10: validation curves, one panel per granularity
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=True)
seed_alpha = {0: 1.0, 1: 0.75, 2: 0.5}
for ax, pair, title in [(axs[0], ('Global', 'Softmax-Global'), 'Fig. 10a  Global granularity (16 coefficients)'),
                        (axs[1], ('Block', 'Softmax-Block'), 'Fig. 10b  Block granularity (528 coefficients)')]:
    for m in pair:
        for s in SEEDS:
            e = A['val'][m][s]
            ax.plot(np.array(e['steps']) / 64, e['curve'], ls=LS[m], lw=1.4, ms=3.6, alpha=seed_alpha[s], **mstyle(m))
            ax.plot([e['best_step'] / 64], [e['best']], marker='*', ms=9, color=HUE[m], markeredgecolor=SURF, zorder=5)
    last = [mean([A['val'][m][s]['curve'][-1] for s in SEEDS]) for m in pair]
    lo, hi = sorted(range(2), key=lambda i: last[i])
    ys = list(last)
    if ys[hi] - ys[lo] < 0.004: ys[hi] = ys[lo] + 0.004
    for m, y, y0 in zip(pair, ys, last):
        ax.annotate(f'{m} (25-epoch seed mean {y0:.4f})', xy=(25, y0), xytext=(26.2, y), va='center', fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.set_xlabel('Epochs over the 512-sample calibration set (1 epoch = 64 steps)')
    ax.set_xlim(-0.8, 43); ax.set_xticks([0, 5, 10, 15, 20, 25]); grid(ax); ax.set_title(title, loc='left')
axs[0].set_ylabel('Val macro BAcc (16 sources)')
axs[0].set_ylim(0.575, 0.632)
handles = [Line2D([], [], ls='-', lw=1.4, ms=4, **mstyle('Block'), label='unconstrained (filled, solid)'),
           Line2D([], [], ls='--', lw=1.4, ms=4, **mstyle('Softmax-Block'), label='softmax (hollow, dashed)'),
           Line2D([], [], ls='', marker='*', ms=9, color=INK2, label='selected step'),
           Line2D([], [], color=INK2, alpha=1.0, label='seed 0'), Line2D([], [], color=INK2, alpha=0.75, label='seed 1'),
           Line2D([], [], color=INK2, alpha=0.5, label='seed 2')]
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / 'fig10_softmax_val_curves.png', dpi=200); plt.close(fig)

# Fig 11: Softmax-Block weights (seed mean) + effective number of experts per group
SBM = SB.mean(0).T   # (G, K)
fig = plt.figure(figsize=(9.2, 7.8))
ax = fig.add_axes([0.17, 0.19, 0.52, 0.76]); ax2 = fig.add_axes([0.76, 0.19, 0.2, 0.76], sharey=ax)
seq = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#9dc0ec', '#2a78d6', '#163f73'])
im = ax.imshow(SBM, cmap=seq, vmin=0, vmax=float(np.ceil(SBM.max() * 10) / 10), aspect='auto')
ax.set_xticks(range(K)); ax.set_xticklabels([SHORT[d] for d in DS], rotation=60, ha='right', color=INK)
ax.set_yticks(range(G)); ax.set_yticklabels(groups, fontsize=6.5, color=INK); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for y in (1.5, 13.5, 14.5, 26.5, 27.5, 31.5): ax.axhline(y, color=SURF, lw=2)
for x in (7.5, 12.5, 13.5): ax.axvline(x, color=SURF, lw=2)
cax = fig.add_axes([0.17, 0.065, 0.52, 0.012])
cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
cb.set_label('Softmax-Block weight α_{k,g}, mean of 3 seeds (each row sums to 1; uniform = 1/16 = 0.0625)', fontsize=7)
for s in SEEDS:
    ax2.plot(EFF[s], range(G), color=BLUE, alpha=seed_alpha[s], lw=1.4, marker='o', ms=2.5, label=f'seed {s}')
ax2.axvline(16, color=MUTED, lw=1, ls=':')
ax2.text(15.6, 0, 'uniform = 16', ha='right', va='center', fontsize=6.5, color=INK2)
ax2.set_xlim(0, 16.5)
grid(ax2, 'x'); ax2.set_xlabel('Effective # experts, 1 / Σ_k α²'); plt.setp(ax2.get_yticklabels(), visible=False)
ax2.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.45, -0.07), ncol=3)
ax.set_title('Fig. 11a  Softmax-Block weights (groups × experts)', loc='left')
ax2.set_title('Fig. 11b  Concentration', loc='left')
fig.savefig(FIG / 'fig11_softmax_block_weights.png', dpi=200); plt.close(fig)

json.dump(dict(numbers=A, tables=T), open(OUT, 'w'), indent=1, default=float)
print('wrote', OUT, 'and', sorted(p.name for p in FIG.glob('fig*_softmax*.png')))
