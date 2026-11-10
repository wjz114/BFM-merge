"""Evaluate ordinary LEAF checkpoints of the model-merging POC on test data.

Every model (Base, each expert, Joint, Uniform/Global/Block merges) is a plain
LEAF state dict. Only the fixed text readout (instruction + label prototypes)
changes per dataset, exactly as in d_downstream_qformer_ce.py. Source datasets
use their standard test split; external MI datasets are used only here (never
for training, checkpoint selection or alpha updates).

The "Specialist" rows combine each expert on its own dataset. They are a
reference built from several models, not a single deployable model.

Example, from LEAF/:
    python eval_poc.py --base checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt \
        --run-dir runs/poc --datasets MI_OpenBMI,MI_Cho2017 --output runs/poc/results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from d_downstream_qformer_ce import (
    METRIC_KEYS, ROOT, PrototypeClassifier, evaluate, load_leaf, make_loader,
    parse_datasets, prepare_text, resolve_file, sha256,
)
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from load_datasets import load_test_split
from utils import seed_all


def load_weibo():
    xs, ys = [], []
    with h5py.File(DATA_FOLDER / 'MI_Weibo2014.h5', 'r', locking=False) as handle:
        for subject in sorted(handle.keys()):
            xs.append(handle[subject]['X'][:])
            ys.append(handle[subject]['Y'][:])
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64), 'all 10 subjects', \
        'verified 0=Left;1=Right (raw event codes 1/2 minus 1)'


def load_dreyer():
    with h5py.File(DATA_FOLDER / 'MI_Dreyer2023.h5', 'r', locking=False) as handle:
        x, y = handle['test']['X'][:], handle['test']['Y'][:].astype(np.int64)
    # 801 samples: crop to the 800-sample convention of the other 4 s datasets.
    return np.ascontiguousarray(x[:, :, :800]), y, 'test split', \
        'ASSUMED 0=Left;1=Right (HDF5 has no label legend)'


EXTERNAL = {'MI_Weibo2014': load_weibo, 'MI_Dreyer2023': load_dreyer}
EXTERNAL_LABELS = ['Left', 'Right']
DEFAULT_GROUPS = ['base', 'experts', 'joint', 'uniform', 'global', 'block']
MODEL_GROUPS = [*DEFAULT_GROUPS, 'softmax_global', 'softmax_block']
MODEL_NAMES = {'softmax_global': 'Softmax-Global', 'softmax_block': 'Softmax-Block'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--aggregate', type=Path, nargs='+',
                   help='Instead of evaluating: summarize per-seed results.csv files (mean, std over seeds)')
    p.add_argument('--collect-direct', type=Path,
                   help='Instead of evaluating: collect c_direct_inference.py CSVs from this directory')
    p.add_argument('--base', type=Path)
    p.add_argument('--run-dir', type=Path,
                   help='Contains experts/<ds>/best.ckpt, joint/best.ckpt, merge_<method>/merged.ckpt')
    p.add_argument('--datasets', help='Comma-separated source datasets')
    p.add_argument('--models', default=','.join(DEFAULT_GROUPS),
                   help=f'Comma-separated subset of {MODEL_GROUPS}')
    p.add_argument('--external', default='MI_Weibo2014',
                   help=f'Comma-separated external MI datasets from {list(EXTERNAL)}; empty for none')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--embedding-dir', type=Path, default=ROOT / 'text_embeddings')
    p.add_argument('--output', type=Path, required=True, help='New CSV file')
    args = p.parse_args()
    if args.aggregate:
        if len(args.aggregate) < 2:
            p.error('--aggregate needs at least two results.csv files')
        return args
    if args.collect_direct:
        if args.datasets is None:
            p.error('--collect-direct needs --datasets (the source datasets)')
        args.datasets = parse_datasets(args.datasets, p)
        # Only merge groups beyond the default set are optional here; the default set is always collected.
        models = [m.strip() for m in args.models.split(',') if m.strip()]
        if any(m not in MODEL_GROUPS for m in models):
            p.error(f'--models must be a subset of {MODEL_GROUPS}')
        args.extra = [m for m in models if m not in DEFAULT_GROUPS]
        return args
    if args.base is None or args.run_dir is None or args.datasets is None:
        p.error('--base, --run-dir and --datasets are required unless --aggregate/--collect-direct is used')
    args.datasets = parse_datasets(args.datasets, p)
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.external = [e.strip() for e in args.external.split(',') if e.strip()]
    if not args.models or any(m not in MODEL_GROUPS for m in args.models):
        p.error(f'--models must be a subset of {MODEL_GROUPS}')
    if any(e not in EXTERNAL for e in args.external):
        p.error(f'--external must be a subset of {list(EXTERNAL)}')
    if args.gpu != '-1' and not args.gpu.isdigit():
        p.error('--gpu must be one nonnegative GPU index, or -1 for CPU')
    return args


AGG_METRICS = [*METRIC_KEYS, 'dBAcc_vs_Joint_pp']


def write_rows(output, rows):
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()})


def print_macro(rows, suffix=''):
    """Compact view: SOURCE_MACRO BAcc and its paired delta vs Joint, per model (and level)."""
    print(f'{"model":<34}{"level":>6}{"source macro BAcc":>22}{"d vs Joint (pp)":>20}')
    for r in rows:
        if r['dataset'] != 'SOURCE_MACRO':
            continue
        bacc, delta = r[f'BAcc{suffix}'], r[f'dBAcc_vs_Joint_pp{suffix}']
        if suffix:
            bacc = f'{bacc:.4f}±{r["BAcc_std"]:.4f}'
            delta = f'{delta:+.2f}±{r["dBAcc_vs_Joint_pp_std"]:.2f}' if delta != '' else '-'
        else:
            bacc, delta = f'{bacc:.4f}', (f'{delta:+.2f}' if delta != '' else '-')
        print(f'{r["model"]:<34}{str(r.get("level", "")):>6}{bacc:>22}{delta:>20}')


def aggregate(paths, output):
    """Mean and sample std (ddof=1) over seeds; deltas vs Joint are paired within each seed.

    Works for both eval_poc results.csv and --collect-direct tables (which add a level column).
    """
    tables = []
    for path in paths:
        with open(path) as handle:
            tables.append({(r['model'], r['dataset'], r.get('level', '')): r for r in csv.DictReader(handle)})
    keys = list(tables[0])
    if any(list(t) != keys for t in tables):
        raise ValueError('Per-seed tables do not contain the same (model, dataset, level) rows')
    header = list(next(iter(tables[0].values())))
    metrics = [m for m in AGG_METRICS if m in header]
    identity = [c for c in header if c not in metrics and c != 'checkpoint']
    rows = []
    for key in keys:
        first = tables[0][key]
        row = {c: first[c] for c in identity}
        row['n_seeds'] = len(tables)
        for metric in metrics:
            if first[metric] == '':
                row[f'{metric}_mean'] = row[f'{metric}_std'] = ''
                continue
            values = np.array([float(t[key][metric]) for t in tables])
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=1))
        row['BAcc_per_seed'] = ';'.join(t[key]['BAcc'] for t in tables)
        rows.append(row)
    write_rows(output, rows)
    (output.with_suffix('.json')).write_text(json.dumps(
        {'sources': [str(Path(p).resolve()) for p in paths],
         'note': 'mean and sample std (ddof=1) over seeds; dBAcc_vs_Joint_pp is paired within each seed'},
        indent=2) + '\n')
    print(f'Test BAcc mean ± std over {len(tables)} seeds')
    print_macro(rows, suffix='_mean')
    print(f'\nSaved {output} and {output.with_suffix(".json")}')


def collect_direct(directory, sources, output, extra=()):
    """Collect c_direct_inference.py CSVs (one per checkpoint) into one long table.

    Expected files: base.csv, joint.csv, uniform.csv, global.csv, block.csv,
    expert_<dataset>.csv for every source dataset, and <group>.csv for every extra
    merge group (softmax_global, softmax_block). Datasets that are not sources
    are kept as separate rows and excluded from SOURCE_MACRO; a non-source dataset
    that some checkpoint was not tested on is dropped (every source is required).
    """
    names = [('Base', 'base'), *[(f'Expert[{ds}]', f'expert_{ds}') for ds in sources],
             ('Joint', 'joint'), ('Uniform', 'uniform'), ('Global', 'global'), ('Block', 'block'),
             *[(MODEL_NAMES[group], group) for group in extra]]
    missing = [str(directory / f'{stem}.csv') for _, stem in names if not (directory / f'{stem}.csv').is_file()]
    if missing:
        raise FileNotFoundError(f'Missing direct-inference results: {missing}')
    results, levels = {}, None
    for model, stem in names:
        with open(directory / f'{stem}.csv') as handle:
            table = [r for r in csv.DictReader(handle) if r['dataset'] != 'MEAN']
        found = sorted({int(k[5]) for k in table[0] if k.startswith('level') and k.endswith('_bAcc')})
        if levels is None:
            levels = found
        if found != levels:
            raise ValueError(f'{stem}.csv has levels {found}, expected {levels}')
        meta = json.loads((directory / f'{stem}.json').read_text())
        if meta.get('status') != 'complete':
            raise ValueError(f'{stem}.json is not complete')
        results[model] = {r['dataset']: r for r in table}
        results[model]['__checkpoint__'] = meta['checkpoint']
    tested = [d for d in results['Base'] if d != '__checkpoint__']
    datasets = [d for d in tested if all(d in results[model] for model, _ in names)]
    absent = [d for d in sources if d not in datasets]
    if absent:
        raise ValueError(f'Source datasets missing from the direct-inference results: {absent}')
    dropped = [d for d in tested if d not in datasets]
    if dropped:
        print(f'Non-source datasets not tested for every checkpoint, dropped: {dropped}', flush=True)
    rows = []
    single = 'single model'
    for model, _ in names:
        for level in levels:
            per = {}
            for ds in datasets:
                r = results[model][ds]
                per[ds] = (float(r[f'level{level}_bAcc']), float(r[f'level{level}_kappa']))
                rows.append({'model': model, 'kind': single, 'dataset': ds,
                             'role': 'source' if ds in sources else 'not a source', 'level': level,
                             'n': r['n_samples'], 'BAcc': per[ds][0], 'Kappa': per[ds][1],
                             'checkpoint': results[model]['__checkpoint__']})
            rows.append({'model': model, 'kind': single, 'dataset': 'SOURCE_MACRO', 'role': 'source',
                         'level': level, 'n': '', 'BAcc': float(np.mean([per[d][0] for d in sources])),
                         'Kappa': float(np.mean([per[d][1] for d in sources])),
                         'checkpoint': results[model]['__checkpoint__']})
    kind = f'specialist reference: {len(sources)} separate models (not one deployable model)'
    for level in levels:
        per = {}
        for ds in sources:
            r = results[f'Expert[{ds}]'][ds]
            per[ds] = (float(r[f'level{level}_bAcc']), float(r[f'level{level}_kappa']))
            rows.append({'model': 'Specialist', 'kind': kind, 'dataset': ds, 'role': 'source', 'level': level,
                         'n': r['n_samples'], 'BAcc': per[ds][0], 'Kappa': per[ds][1],
                         'checkpoint': results[f'Expert[{ds}]']['__checkpoint__']})
        rows.append({'model': 'Specialist', 'kind': kind, 'dataset': 'SOURCE_MACRO', 'role': 'source',
                     'level': level, 'n': '', 'BAcc': float(np.mean([per[d][0] for d in sources])),
                     'Kappa': float(np.mean([per[d][1] for d in sources])), 'checkpoint': ''})
    joint = {(r['dataset'], r['level']): r['BAcc'] for r in rows if r['model'] == 'Joint'}
    for row in rows:
        ref = joint.get((row['dataset'], row['level']))
        row['dBAcc_vs_Joint_pp'] = 100.0 * (row['BAcc'] - ref) if ref is not None else ''
    write_rows(output, rows)
    output.with_suffix('.json').write_text(json.dumps(
        {'protocol': 'c_direct_inference.py (frozen checkpoint, instruction levels, default seed 42)',
         'directory': str(directory.resolve()), 'sources': sources, 'levels': levels,
         'non_source_datasets': [d for d in datasets if d not in sources]}, indent=2) + '\n')
    print_macro(rows)
    print(f'\nSaved {output} and {output.with_suffix(".json")}')


def main():
    started = time.time()
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new --output')
    if args.aggregate:
        return aggregate(args.aggregate, output)
    if args.collect_direct:
        return collect_direct(args.collect_direct, args.datasets, output, args.extra)
    config = build_model_config(load_yaml(resolve_file(args.config, ROOT)))
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    seed_all(0)
    torch.backends.cudnn.benchmark = False
    run_dir = args.run_dir.resolve()

    checkpoints = []  # (model name, path)
    for group in args.models:
        if group == 'base':
            checkpoints.append(('Base', resolve_file(args.base, ROOT / 'checkpoints')))
        elif group == 'experts':
            checkpoints += [(f'Expert[{ds}]', run_dir / 'experts' / ds / 'best.ckpt') for ds in args.datasets]
        elif group == 'joint':
            checkpoints.append(('Joint', run_dir / 'joint' / 'best.ckpt'))
        else:
            checkpoints.append((MODEL_NAMES.get(group, group.capitalize()),
                                run_dir / f'merge_{group}' / 'merged.ckpt'))
    missing = [str(path) for _, path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing checkpoints: {missing}')

    # ---- fixed readouts and test data, loaded once ----
    evals = []  # (dataset, role, split, mapping, instruction, prototypes, loader)
    for ds in args.datasets:
        labels, text, instruction, prototypes = prepare_text(config, ds, None, args.embedding_dir)
        x, y = load_test_split(ds)
        evals.append((ds, 'source', 'test', 'configs/tasks.yaml order', instruction, prototypes,
                      make_loader(x, y, config, len(labels), args.bs, False)))
        print(f'{ds}: test={len(y)} classes={labels}', flush=True)
    for ds in args.external:
        _, text, instruction, prototypes = prepare_text(config, ds, None, args.embedding_dir,
                                                        labels=EXTERNAL_LABELS)
        x, y, split, mapping = EXTERNAL[ds]()
        evals.append((ds, 'external', split, mapping, instruction, prototypes,
                      make_loader(x, y, config, len(EXTERNAL_LABELS), args.bs, False)))
        print(f'{ds}: {split} n={len(y)} classes={EXTERNAL_LABELS} instruction={text!r}; {mapping}', flush=True)

    results = {}  # model -> dataset -> metrics
    for model_name, path in checkpoints:
        leaf = load_leaf(config, path).to(device)
        results[model_name] = {}
        for ds, role, split, mapping, instruction, prototypes, loader in evals:
            model = PrototypeClassifier(leaf, instruction, prototypes).to(device)
            results[model_name][ds] = evaluate(model, loader, device)
        line = ' '.join(f'{ds}={m["BAcc"]:.4f}' for ds, m in results[model_name].items())
        print(f'{model_name}: {line}', flush=True)
        del leaf
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    sources = [e for e in evals if e[1] == 'source']
    externals = [e for e in evals if e[1] == 'external']
    single = 'single model'
    rows = []

    def add(model_name, kind, path, ds, role, split, n, metrics):
        rows.append({'model': model_name, 'kind': kind, 'dataset': ds, 'role': role, 'split': split,
                     'n': n, **{k: metrics[k] for k in METRIC_KEYS}, 'checkpoint': path})

    def macro(per_dataset):
        return {k: float(np.mean([m[k] for m in per_dataset])) for k in METRIC_KEYS}

    for model_name, path in checkpoints:
        res = results[model_name]
        for ds, role, split, *_rest, loader in evals:
            add(model_name, single, str(path), ds, role, split, len(loader.dataset), res[ds])
        add(model_name, single, str(path), 'SOURCE_MACRO', 'source', 'test',
            sum(len(e[-1].dataset) for e in sources), macro([res[e[0]] for e in sources]))
        if len(externals) > 1:
            add(model_name, single, str(path), 'EXTERNAL_MACRO', 'external', 'mixed',
                sum(len(e[-1].dataset) for e in externals), macro([res[e[0]] for e in externals]))
    if 'experts' in args.models:
        kind = f'specialist reference: {len(sources)} separate models (not one deployable model)'
        paths = ';'.join(str(run_dir / 'experts' / e[0] / 'best.ckpt') for e in sources)
        for ds, role, split, *_rest, loader in sources:
            add('Specialist', kind, str(run_dir / 'experts' / ds / 'best.ckpt'), ds, role, split,
                len(loader.dataset), results[f'Expert[{ds}]'][ds])
        add('Specialist', kind, paths, 'SOURCE_MACRO', 'source', 'test',
            sum(len(e[-1].dataset) for e in sources),
            macro([results[f'Expert[{e[0]}]'][e[0]] for e in sources]))

    joint = {r['dataset']: r['BAcc'] for r in rows if r['model'] == 'Joint'}
    for row in rows:
        row['dBAcc_vs_Joint_pp'] = (100.0 * (row['BAcc'] - joint[row['dataset']])
                                    if row['dataset'] in joint else '')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()})
    metadata = {
        'checkpoints': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in checkpoints},
        'datasets': {e[0]: {'role': e[1], 'split': e[2], 'label_mapping': e[3], 'n': len(e[-1].dataset)}
                     for e in evals},
        'metric_note': 'BAcc/Kappa/Auroc/wF1/Loss from d_downstream_qformer_ce.evaluate; fp32 evaluation',
        'external_note': ('External MI datasets never touched training, selection or alpha updates. '
                          'They are absent from configs/tasks.yaml (instruction tuning); whether they '
                          'are strictly unseen also depends on the released checkpoint pretraining corpus.'),
        'seconds': time.time() - started,
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')

    columns = [e[0] for e in evals] + ['SOURCE_MACRO']
    print('\nTest BAcc (dBAcc vs Joint in pp for SOURCE_MACRO)')
    print(f'{"model":<26}' + ''.join(f'{c[:14]:>16}' for c in columns) + f'{"d_vs_Joint":>12}')
    for name in [*results, *(['Specialist'] if 'experts' in args.models else [])]:
        by_ds = {r['dataset']: r for r in rows if r['model'] == name}
        cells = ''.join(f'{by_ds[c]["BAcc"]:>16.4f}' if c in by_ds else f'{"-":>16}' for c in columns)
        delta = by_ds['SOURCE_MACRO']['dBAcc_vs_Joint_pp']
        print(f'{name:<26}{cells}{(f"{delta:+.2f}" if delta != "" else "-"):>12}')
    print(f'\nSaved {output} and {output.with_suffix(".json")}')


if __name__ == '__main__':
    main()
