"""Separate student within-class dispersion from teacher-centroid discrepancy."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import diagnose_prehead_fitting as fitting_worker
from d_downstream_qformer_ce import sha256

ROOT = Path('runs/paper_merge_v2')
PLAN = ROOT/'inputs/prehead_class_moments_plan.json'


class ClassMomentAccumulator(fitting_worker.Accumulator):
    """Extend the frozen inference loop's statistics; no change to model forwards."""
    def __init__(self, n_classes):
        super().__init__(n_classes)
        self.sum_h = self.sum_t = None
        self.sum_h2 = torch.zeros(n_classes, dtype=torch.float64)
        self.sum_t2 = torch.zeros_like(self.sum_h2)
        self.sum_ht = torch.zeros_like(self.sum_h2)

    def add(self, logits, h, cache, positions, multiplicities, weights):
        super().add(logits, h, cache, positions, multiplicities, weights)
        truth = cache['truth'][positions]
        count = multiplicities[positions].double()
        h, t = h.double(), cache['teacher_prehead'][positions].double()
        if self.sum_h is None:
            self.sum_h = torch.zeros(len(self.counts), h.shape[1], dtype=torch.float64)
            self.sum_t = torch.zeros_like(self.sum_h)
            self.stored_teacher_means = cache['teacher_class_means'].double()
            self.class_weights = weights.clone()
        assert torch.equal(self.class_weights, weights)
        self.sum_h.index_add_(0, truth, h*count[:, None])
        self.sum_t.index_add_(0, truth, t*count[:, None])
        self.sum_h2.index_add_(0, truth, h.square().mean(-1)*count)
        self.sum_t2.index_add_(0, truth, t.square().mean(-1)*count)
        self.sum_ht.index_add_(0, truth, (h*t).mean(-1)*count)

    def finish(self, scale):
        result = super().finish(scale)
        student_mean = self.sum_h/self.counts[:, None]
        teacher_mean = self.sum_t/self.counts[:, None]
        student_variance = self.sum_h2/self.counts-student_mean.square().mean(-1)
        teacher_variance = self.sum_t2/self.counts-teacher_mean.square().mean(-1)
        covariance = self.sum_ht/self.counts-(student_mean*teacher_mean).mean(-1)
        centroid_distance = (student_mean-teacher_mean).square().mean(-1)
        stored_centroid_distance = (student_mean-self.stored_teacher_means).square().mean(-1)
        mean_rounding_cross = ((student_mean-self.stored_teacher_means)*
                               (teacher_mean-self.stored_teacher_means)).mean(-1)
        assert student_variance.min() > -1e-12 and teacher_variance.min() > -1e-12
        mass = self.counts*self.class_weights/self.exposures
        components = {'student_within_class_variance': student_variance,
                      'teacher_within_class_variance': teacher_variance,
                      'within_class_covariance': covariance,
                      'centroid_mse_to_exact_teacher_mean': centroid_distance,
                      'centroid_mse_to_stored_teacher_mean': stored_centroid_distance,
                      'within_class_residual_mse': student_variance+teacher_variance-2*covariance,
                      'teacher_mean_rounding_cross_correction': mean_rounding_cross}
        totals = {k: float((v*mass).sum()) for k, v in components.items()}
        qpaired = totals['centroid_mse_to_exact_teacher_mean']+totals['within_class_residual_mse']
        qmean = totals['student_within_class_variance']+totals['centroid_mse_to_stored_teacher_mean']
        errors = {'paired_mse': abs(qpaired-result['weighted_mse']['ce_prehead_paired']),
                  'class_mean_mse': abs(qmean-result['weighted_mse']['ce_prehead_mean'])}
        assert max(errors.values()) < 1e-12, errors
        denominator = math.sqrt(totals['student_within_class_variance']*totals['teacher_within_class_variance'])
        assert denominator > 0
        cosine = totals['within_class_covariance']/denominator
        assert abs(cosine) <= 1+1e-12
        totals['true_class_centered_cosine'] = cosine
        totals['teacher_residual_projection_coefficient'] = totals['within_class_covariance']/totals['teacher_within_class_variance']
        result['class_moments'] = {'weighted_totals': totals,
              'per_class': {k: v.tolist() for k, v in components.items()},
              'class_weighted_exposure_mass': mass.tolist(),
              'teacher_mean_fp32_rounding_max_abs': float((teacher_mean-self.stored_teacher_means).abs().max()),
              'loss_identity_absolute_errors': errors}
        return result


def prepare():
    previous_path = ROOT/'inputs/prehead_fitting_diagnostic_plan.json'
    previous = json.loads(previous_path.read_text())
    assert json.loads((ROOT/'seed0/prehead_training_finish/run.json').read_text())['status'] == 'complete'
    prior_report = ROOT/'analysis/prehead_fitting_summary.json'
    plan = {**previous, 'fixed_unix': time.time(),
            'source_sha256': {**previous['source_sha256'],
                              'diagnose_prehead_class_moments.py': sha256(Path(__file__))},
            'input_sha256': {**previous['input_sha256'], str(previous_path): sha256(previous_path),
                             str(prior_report): sha256(prior_report),
                             previous['primary_report']: sha256(Path(previous['primary_report']))},
            'parent_diagnostic_plan': str(previous_path), 'prior_fitting_report': str(prior_report),
            'shards': {str(i): {'tasks': previous['datasets'][i::2],
                       'outdir': str(ROOT/f'seed0/prehead_class_moments_shard{i}')} for i in (0, 1)},
            'execution': 'The unchanged diagnose_prehead_fitting.run supplies data loading, native forward calls, and state checks. This process replaces its module-level Accumulator class with the documented subclass, extending statistics only. run.json code_sha256 identifies the inherited inference loop; wrapper.json pins the subclass and complete result hash. No source file is edited.',
            'timing': 'Subsequent diagnostic after all four training results and fitting/pairing outcomes. Fixed before new class-moment outcomes. No new training or hyperparameter selection.',
            'scope': 'All four fixed checkpoint means, exact original fitting multiset, existing teacher cache, FP64 first/second/cross moments. Decompose distance to stored teacher class mean into student within-class variance plus centroid discrepancy; decompose paired loss into true-centroid discrepancy plus within-class residual MSE. Report true class-centered cosine separately from prior teacher-mean-centered cosine. Initial features remain cached. Record repeated-inference loss differences against the prior fitting report without inventing a bitwise native replay guarantee. No validation/test EEG, no teacher inference, no updates.'}
    with PLAN.open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    print(json.dumps({'plan': str(PLAN), 'sha256': sha256(PLAN), 'budget': plan['budget']}, indent=2))


def run(plan_path, shard, gpu):
    plan = json.loads(plan_path.read_text())
    assert plan['source_sha256']['diagnose_prehead_class_moments.py'] == sha256(Path(__file__))
    fitting_worker.Accumulator = ClassMomentAccumulator
    fitting_worker.run(plan_path, shard, gpu)
    path = Path(plan['shards'][str(shard)]['outdir'])/'run.json'
    result = json.loads(path.read_text())
    prior = json.loads(Path(plan['prior_fitting_report']).read_text())
    replay = {}
    for task, record in result['records'].items():
        replay[task] = {}
        for model, metrics in record['models'].items():
            old = prior['records'][task]['models'][model]
            replay[task][model] = {'weighted_ce_difference': metrics['weighted_ce']-old['weighted_ce'],
                'exposure_bacc_difference': metrics['exposure_bacc']-old['exposure_bacc'],
                'weighted_mse_differences': {k: v-old['weighted_mse'][k] for k, v in metrics['weighted_mse'].items()}}
    wrapper = {'plan_sha256': sha256(plan_path), 'wrapper_sha256': sha256(Path(__file__)),
               'inherited_inference_loop_sha256': sha256(Path(fitting_worker.__file__)),
               'run_sha256': sha256(path), 'prior_fitting_report_sha256': sha256(Path(plan['prior_fitting_report'])),
               'repeated_native_inference_differences': replay, 'scope': plan['scope']}
    with path.with_name('wrapper.json').open('x') as f:
        json.dump(wrapper, f, indent=2); f.write('\n')
    print('Wrapper record complete', flush=True)


def report(plan_path):
    plan = json.loads(plan_path.read_text())
    records, sources, forwards, replay_error = {}, {}, 0, 0.
    for spec in plan['shards'].values():
        path = Path(spec['outdir'])/'run.json'
        wrapper_path = path.with_name('wrapper.json')
        data, wrapper = [json.loads(p.read_text()) for p in [path, wrapper_path]]
        assert data['plan_sha256'] == wrapper['plan_sha256'] == sha256(plan_path)
        assert wrapper['run_sha256'] == sha256(path)
        assert wrapper['wrapper_sha256'] == sha256(Path(__file__))
        assert data['code_sha256'] == wrapper['inherited_inference_loop_sha256'] == sha256(Path(fitting_worker.__file__))
        assert data['all_model_states_unchanged'] and not data['validation_eeg_read'] and not data['test_data_read']
        records.update(data['records']); forwards += data['main_forward_examples']
        sources.update({str(p): sha256(p) for p in [path, wrapper_path]})
        for models in wrapper['repeated_native_inference_differences'].values():
            for row in models.values():
                replay_error = max(replay_error, abs(row['weighted_ce_difference']), abs(row['exposure_bacc_difference']),
                                   *[abs(v) for v in row['weighted_mse_differences'].values()])
    assert set(records) == set(plan['datasets']) and forwards == plan['budget']['native_forward_examples']
    models = ['initial_cached', *plan['conditions']]
    keys = list(next(iter(records.values()))['models']['initial_cached']['class_moments']['weighted_totals'])
    macro = {model: {key: float(np.mean([records[t]['models'][model]['class_moments']['weighted_totals'][key]
                                       for t in plan['datasets']])) for key in keys} for model in models}
    contrasts = {}
    for model in plan['conditions'][1:]:
        contrasts[model] = {}
        for key in keys:
            differences = {t: records[t]['models'][model]['class_moments']['weighted_totals'][key]
                              -records[t]['models']['balanced_ce']['class_moments']['weighted_totals'][key]
                           for t in plan['datasets']}
            contrasts[model][key] = {'macro_difference': float(np.mean(list(differences.values()))),
                                     'positive_tasks': sum(v > 0 for v in differences.values()),
                                     'negative_tasks': sum(v < 0 for v in differences.values()),
                                     'per_task_differences': differences}
    output = ROOT/'analysis/prehead_class_moments_summary.json'
    result = {'plan': str(plan_path), 'plan_sha256': sha256(plan_path), 'code_sha256': sha256(Path(__file__)),
              'sources': sources, 'datasets': plan['datasets'], 'records': records, 'macro': macro,
              'contrasts_vs_ce': contrasts, 'new_forward_examples': forwards, 'optimizer_updates': 0,
              'max_repeated_native_metric_absolute_difference': replay_error, 'scope': plan['scope']}
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'macro': macro,
                     'max_replay_difference': replay_error}, indent=2))


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
