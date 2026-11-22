"""Evaluate the three fixed row-space experiment means on their fitting stream."""
import argparse
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


class FittingMetrics:
    def __init__(self, classes):
        self.ce = self.correct = 0.
        self.exposures = self.unique_rows = 0
        self.counts = torch.zeros(classes, dtype=torch.float64)
        self.correct_counts = torch.zeros_like(self.counts)
        self.unique_counts = torch.zeros_like(self.counts)
        self.unique_correct = torch.zeros_like(self.counts)

    def add(self, logits, truth, multiplicities, weights):
        assert torch.isfinite(logits).all()
        count = multiplicities.double()
        included = count > 0
        correct = (logits.argmax(-1) == truth).double()
        self.ce += float((F.cross_entropy(logits.double(), truth, reduction='none') * count * weights[truth]).sum())
        self.correct += float((correct * count).sum())
        self.exposures += int(count.sum())
        self.unique_rows += int(included.sum())
        self.counts += torch.bincount(truth, weights=count, minlength=len(self.counts))
        self.correct_counts += torch.bincount(truth, weights=correct*count, minlength=len(self.counts))
        self.unique_counts += torch.bincount(truth[included], minlength=len(self.counts))
        self.unique_correct += torch.bincount(truth[included], weights=correct[included], minlength=len(self.counts))

    def finish(self):
        assert self.exposures > 0 and (self.counts > 0).all() and (self.unique_counts > 0).all()
        return {'weighted_ce': self.ce/self.exposures, 'exposure_accuracy': self.correct/self.exposures,
                'exposure_bacc': float((self.correct_counts/self.counts).mean()),
                'unique_fitting_bacc': float((self.unique_correct/self.unique_counts).mean()),
                'fitting_exposures': self.exposures, 'unique_fitting_rows': self.unique_rows,
                'class_exposures': self.counts.tolist(), 'class_unique_rows': self.unique_counts.tolist()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    primary_path = Path(plan['primary_report'])
    primary = json.loads(primary_path.read_text())
    assert set(primary['average_reconstruction_checks']) == set(plan['conditions'])
    outdir = Path(plan['outdir'])
    outdir.mkdir(exist_ok=False)
    started = time.time()
    device = configure(args.gpu, plan['seed'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source = json.loads(Path(plan['source_fit']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    manifest = json.loads(Path(plan['target_cache']).read_text())
    reference = json.loads(Path(plan['initial_metrics_reference']).read_text())
    schedule = np.load(plan['schedule'])
    models, snapshots, checkpoint_sources = {}, {}, {}
    for condition, folder_name in plan['models'].items():
        folder = Path(folder_name)
        run = json.loads((folder/'run.json').read_text())
        assert run['average_steps'] == plan['average_steps']
        for path in [folder/'run.json', folder/'wrapper.json', folder/'merged.ckpt']:
            assert sha256(path) == primary['sources'][str(path)]
            checkpoint_sources[str(path)] = sha256(path)
        model = load_leaf(config, folder/'merged.ckpt').to(device).eval().requires_grad_(False)
        models[condition] = model
        snapshots[condition] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    records, forwards = {}, 0
    for ti, task in enumerate(plan['datasets']):
        task_started = time.time()
        ref = manifest['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        ids = cache['pool_indices'].numpy()
        assert np.all(ids[1:] > ids[:-1])
        stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert source['datasets'][ti] == task and array_hash(stream) == ref['stream_indices_hash']
        positions = torch.from_numpy(np.searchsorted(ids, stream))
        assert np.array_equal(ids[positions], stream)
        multiplicities = torch.bincount(positions, minlength=len(ids))
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as handle:
            rows, all_truth = row_map(handle, task, 'calibration')
            assert array_hash(np.asarray(all_truth, dtype=np.int64)) == coverage['audits'][task]['train_labels_hash']
            x = torch.from_numpy(selected_eeg(handle, [rows[int(i)] for i in ids]))
            truth = torch.from_numpy(np.asarray(all_truth[ids], dtype=np.int64))
        assert array_hash(x) == ref['selected_training_eeg_hash']
        assert torch.equal(truth, cache['truth']) and array_hash(truth) == ref['truth_hash']
        labels, text, instruction, prototypes = prepare_text(config, task)
        assert labels == source['labels'][task] and text == source['instruction'][task]
        assert array_hash(instruction) == ref['instruction_hash'] and array_hash(prototypes) == ref['prototypes_hash']
        weights = torch.tensor(ref['class_weights'], dtype=torch.float64)
        totals = FittingMetrics(len(labels))
        batch_size = plan['batch_size']
        for start in range(0, len(ids), batch_size):
            sl = slice(start, start+batch_size)
            totals.add(cache['initial_logits'][sl], truth[sl], multiplicities[sl], weights)
        results = {'initial_cached': totals.finish()}
        replay = {}
        for metric in ['weighted_ce', 'exposure_accuracy', 'exposure_bacc']:
            old = reference['records'][task]['models']['initial_cached'][metric]
            actual = results['initial_cached'][metric]
            replay[metric] = abs(actual-old)
            assert math.isclose(actual, old, rel_tol=1e-12, abs_tol=1e-12)
        for condition, model in models.items():
            wrapper = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
            assert torch.equal(wrapper.prototypes.cpu(), cache['prototypes'])
            totals = FittingMetrics(len(labels))
            for start in range(0, len(ids), batch_size):
                sl = slice(start, start+batch_size)
                logits = wrapper(x[sl].to(device)).cpu()
                assert logits.dtype == torch.float32
                totals.add(logits, truth[sl], multiplicities[sl], weights)
                forwards += len(logits)
            results[condition] = totals.finish()
        assert all(v['fitting_exposures'] == ref['fitting_exposures'] and
                   v['unique_fitting_rows'] == ref['unique_fitting_rows'] for v in results.values())
        records[task] = {'models': results, 'initial_reference_absolute_errors': replay,
                         'selected_training_eeg_hash': ref['selected_training_eeg_hash'],
                         'target_sha256': ref['target_sha256'], 'seconds': time.time()-task_started}
        (outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, {k: {'ce': v['weighted_ce'], 'bacc': v['exposure_bacc']} for k, v in results.items()}, flush=True)
        del x, cache
    for condition, model in models.items():
        assert all(torch.equal(v.detach().cpu(), snapshots[condition][k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    assert forwards == plan['budget']['native_forward_examples']
    metrics = ['weighted_ce', 'exposure_accuracy', 'exposure_bacc', 'unique_fitting_bacc']
    macro = {c: {m: float(np.mean([records[t]['models'][c][m] for t in plan['datasets']]))
                 for m in metrics} for c in ['initial_cached', *plan['conditions']]}
    contrasts = {f'{a}_vs_{b}': {m: macro[a][m]-macro[b][m] for m in metrics}
                 for a, b in [('row', 'full'), ('shuffled', 'full'), ('row', 'shuffled')]}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'primary_report_sha256': sha256(primary_path), 'checkpoint_sources': checkpoint_sources,
              'records': records, 'macro': macro, 'contrasts': contrasts,
              'native_forward_examples': forwards, 'optimizer_updates': 0,
              'all_model_states_unchanged': True, 'validation_eeg_read': False, 'test_data_read': False,
              'seconds': time.time()-started, 'scope': plan['scope']}
    with (outdir/'run.json').open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print('Completed', json.dumps({'seconds': result['seconds'], 'macro': macro}), flush=True)


if __name__ == '__main__':
    main()
