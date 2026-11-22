"""Training-stream objective audit for all four fixed prehead checkpoint means."""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from cache_preprojection_representations import selected_eeg
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, load_leaf, prepare_text, prototype_logits, sha256
from diagnose_sensor_subspaces import row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from prehead_distillation_support import CONDITIONS, target_rows

ROOT_RUN = Path('runs/paper_merge_v2')
PLAN = ROOT_RUN/'inputs/prehead_fitting_diagnostic_plan.json'


class Accumulator:
    """FP64 sums on unique inputs, with exact fitting-stream multiplicities."""
    def __init__(self, n_classes):
        self.exposures = 0
        self.ce = self.correct = self.agreement = 0.
        self.counts = torch.zeros(n_classes, dtype=torch.float64)
        self.correct_counts = torch.zeros_like(self.counts)
        self.mse = {c: 0. for c in CONDITIONS[1:]}

    def add(self, logits, h, cache, positions, multiplicities, weights):
        truth = cache['truth'][positions]
        count = multiplicities[positions].double()
        weighted = count * weights[truth]
        prediction = logits.argmax(-1)
        correct = (prediction == truth).double()
        self.exposures += int(count.sum())
        self.ce += float((F.cross_entropy(logits.double(), truth, reduction='none') * weighted).sum())
        self.correct += float((correct * count).sum())
        teacher = cache['teacher_logits'][positions].argmax(-1)
        self.agreement += float(((prediction == teacher).double() * count).sum())
        self.counts += torch.bincount(truth, weights=count, minlength=len(self.counts))
        self.correct_counts += torch.bincount(truth, weights=correct * count, minlength=len(self.counts))
        for condition in self.mse:
            target = target_rows(cache, positions, condition).double()
            self.mse[condition] += float(((h.double()-target).square().mean(-1) * weighted).sum())

    def finish(self, scale):
        assert self.exposures > 0 and (self.counts > 0).all()
        ce = self.ce / self.exposures
        mse = {k: v/self.exposures for k, v in self.mse.items()}
        result = {'fitting_exposures': self.exposures, 'weighted_ce': ce,
                  'exposure_accuracy': self.correct/self.exposures,
                  'exposure_bacc': float((self.correct_counts/self.counts).mean()),
                  'exposure_teacher_agreement': self.agreement/self.exposures,
                  'class_exposures': self.counts.tolist(), 'weighted_mse': mse,
                  'scaled_weighted_mse': {k: scale*v for k, v in mse.items()},
                  'ce_plus_scaled_mse': {k: ce+scale*v for k, v in mse.items()}}
        assert all(math.isfinite(v) for v in [ce, *mse.values()])
        return result


def prepare():
    design_path = ROOT_RUN/'inputs/prehead_distillation_design.json'
    design = json.loads(design_path.read_text())
    target_path, scale_path = Path(design['target_manifest']), Path(design['scales'])
    targets = json.loads(target_path.read_text())
    sources = {**design['source_sha256'], str(Path(__file__).relative_to(Path.cwd())): sha256(Path(__file__))}
    for name, digest in sources.items():
        assert sha256(Path(name)) == digest, name
    inputs = {**design['input_sha256'], **{str(p): sha256(p) for p in [design_path, target_path, scale_path]}}
    plan = {'fixed_unix': time.time(), 'source_sha256': sources, 'input_sha256': inputs,
            'design': str(design_path), 'datasets': design['datasets'], 'conditions': list(CONDITIONS),
            'source_fit': design['source_fit'], 'coverage_audit': design['coverage_audit'],
            'schedule': design['schedule'], 'target_manifest': str(target_path), 'scales': str(scale_path),
            'training_arms': design['training_arms'], 'average_steps': design['average_steps'],
            'primary_report': str(ROOT_RUN/'analysis/prehead_distillation_summary.json'),
            'batch_size': 128, 'seed': 0, 'initial_mse_replay_rtol': 1e-12,
            'initial_mse_replay_atol': 1e-12,
            'shards': {str(i): {'tasks': design['datasets'][i::2],
                       'outdir': str(ROOT_RUN/f'seed0/prehead_fitting_shard{i}')} for i in (0, 1)},
            'budget': {'optimizer_updates': 0, 'teacher_forward_examples': 0,
                       'native_forward_examples': 4*sum(r['forward_rows_per_model'] for r in targets['records'].values()),
                       'unique_fitting_rows_per_model': sum(r['unique_fitting_rows'] for r in targets['records'].values())},
            'timing': 'Diagnostic fixed after first-wave intermediate and fixed-mean validation scores were observed; no fitting-diagnostic outcomes observed. Scientific arms, targets, and checkpoint means were already fixed. This is an explicitly subsequent optimization diagnostic.',
            'scope': 'After all four training arms and the primary report pass, evaluate all four fixed six-state means on the existing target-cache training rows. Weight objective sums by fitting multiplicity and full-pool class weight, divide by fitting exposure count, not weight sum. Cross-evaluate every model against all three targets. Native FP32 inference; loss statistics recomputed in FP64. Initial comparator uses cached features/logits. Replay-only rows have zero objective weight but count toward forward cost. No validation/test EEG or new teacher inference; no updates or hyperparameter selection. Target fitting is an optimization diagnostic, not proof of useful semantic transfer or independent generalization.'}
    with PLAN.open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    print(json.dumps({'plan': str(PLAN), 'sha256': sha256(PLAN), 'budget': plan['budget']}, indent=2))


@torch.no_grad()
def run(plan_path, shard, gpu):
    plan = json.loads(plan_path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest, name
    report_path = Path(plan['primary_report'])
    report = json.loads(report_path.read_text())
    assert set(report['average_reconstruction_checks']) == set(CONDITIONS)
    spec = plan['shards'][str(shard)]
    outdir = Path(spec['outdir'])
    outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(gpu, plan['seed'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source = json.loads(Path(plan['source_fit']).read_text())
    coverage = json.loads(Path(plan['coverage_audit']).read_text())
    manifest = json.loads(Path(plan['target_manifest']).read_text())
    scales = json.loads(Path(plan['scales']).read_text())
    schedule = np.load(plan['schedule'])
    models, snapshots, checkpoint_sources = {}, {}, {}
    for condition in CONDITIONS:
        path = Path(plan['training_arms'][condition]['outdir'])
        receipt = json.loads((path/'run.json').read_text())
        assert [s['step'] for s in receipt['average_snapshots']] == plan['average_steps']
        for p in [path/'run.json', path/'merged.ckpt']:
            assert sha256(p) == report['sources'][str(p)]
            checkpoint_sources[str(p)] = sha256(p)
        model = load_leaf(config, path/'merged.ckpt').to(device).eval().requires_grad_(False)
        models[condition] = model
        snapshots[condition] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    records, examples = {}, 0
    for task in spec['tasks']:
        task_started = time.time()
        ref = manifest['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        ids = cache['pool_indices'].numpy()
        ti = source['datasets'].index(task)
        stream = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert array_hash(stream) == ref['stream_indices_hash']
        pos = torch.from_numpy(np.searchsorted(ids, stream))
        assert np.array_equal(ids[pos], stream)
        multiplicities = torch.bincount(pos, minlength=len(ids))
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
        scale = scales['per_task'][task]['scale']
        initial = Accumulator(len(labels))
        for start in range(0, len(ids), plan['batch_size']):
            positions = torch.arange(start, min(start+plan['batch_size'], len(ids)))
            initial.add(cache['initial_logits'][positions], cache['initial_prehead'][positions],
                        cache, positions, multiplicities, weights)
        results = {'initial_cached': initial.finish(scale)}
        replay = {}
        for c in CONDITIONS[1:]:
            old = scales['per_task'][task]['conditions'][c]['weighted_auxiliary_mean']
            actual = results['initial_cached']['weighted_mse'][c]
            replay[c] = abs(actual-old)
            assert math.isclose(actual, old, rel_tol=plan['initial_mse_replay_rtol'], abs_tol=plan['initial_mse_replay_atol'])
        for condition, model in models.items():
            wrapper = PrototypeClassifier(model, instruction, prototypes).to(device).eval()
            assert torch.equal(wrapper.prototypes.cpu(), cache['prototypes'])
            head = model.qformer.output_embed[2]
            captured = {}
            def hook(module, arguments, output):
                captured['h'], captured['raw'] = arguments[0].detach(), output.detach()
            handle = head.register_forward_hook(hook)
            totals = Accumulator(len(labels))
            try:
                for start in range(0, len(ids), plan['batch_size']):
                    positions = torch.arange(start, min(start+plan['batch_size'], len(ids)))
                    logits = wrapper(x[start:start+plan['batch_size']].to(device))
                    h, raw = captured['h'], captured['raw']
                    assert torch.equal(raw, F.linear(h, head.weight, head.bias))
                    assert torch.equal(logits, prototype_logits(F.normalize(raw, dim=-1), wrapper.prototypes))
                    assert torch.isfinite(h).all() and torch.isfinite(logits).all()
                    totals.add(logits.cpu(), h.cpu(), cache, positions, multiplicities, weights)
                    examples += len(logits)
            finally:
                handle.remove()
            results[condition] = totals.finish(scale)
            assert results[condition]['fitting_exposures'] == ref['fitting_exposures']
        records[task] = {'models': results, 'initial_mse_replay_absolute_error': replay,
                         'target_sha256': ref['target_sha256'], 'scale': scale,
                         'selected_training_eeg_hash': ref['selected_training_eeg_hash'],
                         'native_head_reconstruction_exact': True, 'seconds': time.time()-task_started}
        (outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, {k: {'paired_mse': v['weighted_mse']['ce_prehead_paired'], 'fit_bacc': v['exposure_bacc']}
                     for k, v in results.items()}, flush=True)
        del x, cache
    for condition, model in models.items():
        assert all(torch.equal(v.detach().cpu(), snapshots[condition][n]) for n, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    result = {'plan': plan, 'plan_sha256': sha256(plan_path), 'code_sha256': sha256(Path(__file__)),
              'primary_report_sha256': sha256(report_path), 'checkpoint_sources': checkpoint_sources,
              'datasets': spec['tasks'], 'records': records, 'main_forward_examples': examples,
              'all_model_states_unchanged': True, 'optimizer_updates': 0,
              'validation_eeg_read': False, 'validation_inference_performed': False, 'test_data_read': False,
              'seconds': time.time()-started, 'scope': plan['scope']}
    with (outdir/'run.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('Completed', result['seconds'], flush=True)


def report(plan_path):
    plan = json.loads(plan_path.read_text())
    records, sources, examples = {}, {}, 0
    for spec in plan['shards'].values():
        path = Path(spec['outdir'])/'run.json'
        run_data = json.loads(path.read_text())
        assert run_data['plan_sha256'] == sha256(plan_path)
        assert run_data['code_sha256'] == sha256(Path(__file__))
        assert run_data['all_model_states_unchanged'] and not run_data['test_data_read']
        records.update(run_data['records'])
        examples += run_data['main_forward_examples']; sources[str(path)] = sha256(path)
    assert set(records) == set(plan['datasets'])
    assert examples == plan['budget']['native_forward_examples']
    macro = {}
    for model in ['initial_cached', *CONDITIONS]:
        rows = [records[t]['models'][model] for t in plan['datasets']]
        macro[model] = {k: float(np.mean([r[k] for r in rows])) for k in
                        ['weighted_ce', 'exposure_bacc', 'exposure_accuracy', 'exposure_teacher_agreement']}
        for key in ['weighted_mse', 'scaled_weighted_mse', 'ce_plus_scaled_mse']:
            macro[model][key] = {c: float(np.mean([r[key][c] for r in rows])) for c in CONDITIONS[1:]}
    contrasts = {}
    for c in CONDITIONS[1:]:
        contrasts[c] = {}
        for comparator in ['initial_cached', 'balanced_ce']:
            delta = {t: records[t]['models'][c]['weighted_mse'][c]-records[t]['models'][comparator]['weighted_mse'][c]
                     for t in plan['datasets']}
            contrasts[c][comparator] = {'own_target_mse_difference': delta,
                                        'tasks_with_lower_mse': sum(v < 0 for v in delta.values()),
                                        'macro_mse_difference': float(np.mean(list(delta.values())))}
    output = ROOT_RUN/'analysis/prehead_fitting_summary.json'
    result = {'plan': str(plan_path), 'plan_sha256': sha256(plan_path), 'code_sha256': sha256(Path(__file__)),
              'sources': sources, 'datasets': plan['datasets'], 'macro': macro, 'own_target_contrasts': contrasts,
              'records': records, 'main_forward_examples': examples, 'scope': plan['scope']}
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'macro': macro}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'report'])
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--shard', type=int, choices=[0, 1])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        assert args.shard is not None and args.gpu is not None
        run(args.plan, args.shard, args.gpu)
    else:
        report(args.plan)
