"""Record real source assets, parameter groups and legacy sample identities."""
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, load_leaf, prepare_text, sha256
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from load_datasets import SUBJECT_SPLITS, PREDEFINED_SPLITS
from merge_poc import UNUSED_PREFIXES, group_for_name


def split_ids(ds):
    """Mirror the original loader, retaining subject and within-subject trial."""
    with h5py.File(DATA_FOLDER / f'{ds}.h5', 'r', locking=False) as f:
        if ds in PREDEFINED_SPLITS:
            return {split: [f'{ds}:{key}:{i}' for i in range(len(f[key]))]
                    for split, key in [('train', 'trainY'), ('val', 'validY'), ('test', 'testY')]}, False
        train_end, test_end = SUBJECT_SPLITS[ds]
        trainval, test = [], []
        for i, sub in enumerate(list(f.keys())[:test_end]):
            y = f[sub]['Y'][:]
            ids = [f'{ds}:subject={sub}:trial={j}' for j in range(len(y)) if ds != 'MI_HighGamma' or y[j] != 2]
            (trainval if i < train_end else test).extend(ids)
        return {'train': [x for i, x in enumerate(trainval) if i % 5],
                'val': [x for i, x in enumerate(trainval) if not i % 5], 'test': test}, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--outdir', type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    source = ROOT / f'runs/poc16/seed{args.seed}'
    calibration_path = source / 'merge_block/calibration_indices.npz'
    calibration = np.load(calibration_path)
    names = calibration.files
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    base = ROOT / 'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'
    leaf = load_leaf(config, base)
    state = {n: t.clone() for n, t in leaf.state_dict().items()}
    parameters = dict(leaf.named_parameters())
    groups = {}
    unused = []
    for n in parameters:
        if n.startswith(UNUSED_PREFIXES):
            unused.append(n)
        else:
            groups.setdefault(group_for_name(n), []).append(n)
    assets = {}
    paths = {'base': base, 'joint': source / 'joint/best.ckpt',
             **{ds: source / 'experts' / ds / 'best.ckpt' for ds in names}}
    for name, path in paths.items():
        if not path.is_file():
            assets[name] = {'path': str(path), 'status': 'missing'}
            continue
        candidate = torch.load(path, map_location='cpu', weights_only=True)
        leaf.load_state_dict(candidate, strict=True)
        same_buffers = all(torch.equal(candidate[n], state[n]) for n, _ in leaf.named_buffers())
        same_unused = all(torch.equal(candidate[n], state[n]) for n in unused)
        if not same_buffers or not same_unused:
            raise AssertionError(f'{name}: changed base buffers or inactive parameters')
        assets[name] = {'path': str(path), 'sha256': sha256(path), 'strict_load': True,
                        'buffers_and_inactive_equal_base': True,
                        'parameter_signature': {n: [list(t.shape), str(t.dtype)] for n, t in candidate.items()}}
        run_path = path.with_name('run.json')
        if run_path.exists():
            run = json.loads(run_path.read_text())
            assets[name]['seed'] = run['seed']
            assets[name]['initialization_sha256'] = run['source_checkpoint_sha256']
            if assets[name]['initialization_sha256'] != assets['base']['sha256']:
                raise AssertionError(f'{name} has a different initialization')
        print(f'audited {name}', flush=True)
    split_manifest, text = {}, {}
    for ds in names:
        ids, has_subjects = split_ids(ds)
        sets = {k: set(v) for k, v in ids.items()}
        assert not sets['train'] & sets['val'] and not sets['train'] & sets['test'] and not sets['val'] & sets['test']
        index = calibration[ds]
        assert len(np.unique(index)) == len(index) and index.min() >= 0 and index.max() < len(ids['train'])
        ids['calibration'] = [ids['train'][i] for i in index]
        (args.outdir / f'{ds}_sample_ids.json').write_text(json.dumps(ids) + '\n')
        path = DATA_FOLDER / f'{ds}.h5'
        split_manifest[ds] = {'file': str(path), 'bytes': path.stat().st_size,
                              'mtime_ns': path.stat().st_mtime_ns, 'counts': {k: len(v) for k, v in ids.items()},
                              'subject_ids_available': has_subjects, 'sample_sets_disjoint': True,
                              'validation_policy': 'predefined' if not has_subjects else 'every fifth source trial; subjects shared with train',
                              'ids_sha256': sha256(args.outdir / f'{ds}_sample_ids.json')}
        labels, instruction_text, instruction, prototypes = prepare_text(config, ds)
        text[ds] = {'labels_in_numeric_order': labels, 'instruction': instruction_text,
                    'instruction_hash': array_hash(instruction), 'prototypes_hash': array_hash(prototypes)}
    audit = {
        'repo_root': str(ROOT), 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty_status': subprocess.check_output(['git', 'status', '--short'], text=True),
        'assets': assets, 'datasets': split_manifest, 'text': text, 'groups': groups,
        'inactive_parameters': unused, 'buffers': [n for n, _ in leaf.named_buffers()],
        'calibration_path': str(calibration_path), 'calibration_sha256': sha256(calibration_path),
        'file_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                        [ROOT / f for f in ['LEAF.py', 'merge_poc.py', 'd_downstream_qformer_ce.py',
                         'load_datasets.py', 'configs/env.yaml', 'configs/LEAF_mpnet.yaml', 'configs/tasks.yaml',
                         'text_embeddings/mpnet-base.npy', 'text_embeddings/mpnet-base.txt']]},
        'limitations': ['Predefined HDF5 splits do not expose original subject/session IDs; split row IDs retained.',
                       'Data files identified by path/stat and sample IDs; full EEG file checksums not computed here.',
                       'Existing public initialization task exposure is not a future-task-unseen protocol.'],
    }
    (args.outdir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Saved audit and sample IDs to {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
