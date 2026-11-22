"""Exact feature-loss decomposition, using cached targets and fixed audit losses."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from aligned_support import array_hash
from d_downstream_qformer_ce import sha256

ROOT = Path('runs/paper_merge_v2')
REFERENCE = ROOT/'analysis/prehead_pairing_reference.json'
CONDITIONS = ['balanced_ce', 'ce_prehead_paired', 'ce_prehead_mean', 'ce_prehead_shuffled']


@torch.no_grad()
def prepare():
    torch.set_num_threads(4)
    started = time.time()
    plan_path = ROOT/'inputs/prehead_fitting_diagnostic_plan.json'
    plan = json.loads(plan_path.read_text())
    manifest_path = Path(plan['target_manifest'])
    scales_path = Path(plan['scales'])
    manifest, scales = [json.loads(p.read_text()) for p in [manifest_path, scales_path]]
    schedule = np.load(plan['schedule'])
    source = json.loads(Path(plan['source_fit']).read_text())
    records = {}
    for task in plan['datasets']:
        ref = manifest['records'][task]
        assert sha256(Path(ref['target_path'])) == ref['target_sha256']
        cache = torch.load(ref['target_path'], map_location='cpu', weights_only=True)
        ti = source['datasets'].index(task)
        ids = schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64)
        assert array_hash(ids) == ref['stream_indices_hash']
        pos = torch.searchsorted(cache['pool_indices'], torch.from_numpy(ids))
        assert torch.equal(cache['pool_indices'][pos], torch.from_numpy(ids))
        y = cache['truth'][pos]
        weights = torch.tensor(ref['class_weights'], dtype=torch.float64)[y]
        mean = cache['teacher_class_means'][y].double()
        teacher = cache['teacher_prehead'][pos].double()
        shuffled = cache['teacher_prehead'][cache['shuffled_pool_positions'][pos]].double()
        initial = cache['initial_prehead'][pos].double()
        residual = teacher-mean
        shuffled_residual = shuffled-mean
        variance = float((residual.square().mean(-1)*weights).mean())
        shuffled_variance = float((shuffled_residual.square().mean(-1)*weights).mean())
        assert math.isclose(variance, shuffled_variance, rel_tol=1e-12, abs_tol=1e-12)
        assert variance > 0
        centered_initial = initial-mean
        mean_loss = scales['per_task'][task]['conditions']['ce_prehead_mean']['weighted_auxiliary_mean']
        checks = {}
        for key, r in [('ce_prehead_paired', residual), ('ce_prehead_shuffled', shuffled_residual)]:
            direct = float(((centered_initial*r).mean(-1)*weights).mean())
            target_loss = scales['per_task'][task]['conditions'][key]['weighted_auxiliary_mean']
            reconstructed = (variance+mean_loss-target_loss)/2
            assert math.isclose(direct, reconstructed, rel_tol=1e-12, abs_tol=1e-12)
            checks[key] = {'direct_initial_inner_product': direct,
                           'loss_reconstructed_initial_inner_product': reconstructed,
                           'absolute_error': abs(direct-reconstructed)}
        records[task] = {'teacher_residual_energy': variance,
                         'shuffled_residual_energy': shuffled_variance,
                         'target_sha256': ref['target_sha256'], 'initial_identity_checks': checks}
    inputs = [plan_path, manifest_path, scales_path, Path(plan['schedule']), Path(plan['source_fit'])]
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'sources': {str(p): sha256(p) for p in inputs}, 'datasets': plan['datasets'],
              'records': records, 'seconds': time.time()-started, 'new_forward_examples': 0,
              'optimizer_updates': 0, 'scope': 'Subsequent algebraic analysis fixed before fitting-diagnostic outputs, after first-pair validation scores were observed. Define a_i=h_i-mu_y and r_i=t_i-mu_y using the actual FP32 cached teacher class mean. With fitting multiplicity and class weights, Q_paired=Q_mean+V_teacher-2*C_paired exactly; shuffle has the same V_teacher. No assumption that stored class means are exact in FP64 is needed. Report C and its weighted cosine C/sqrt(Q_mean*V). This centers the student at the TEACHER class mean, not its own; cosine is not class-centered CKA or semantic information. More sample pairing without accuracy benefit does not establish useful knowledge transfer.'}
    with REFERENCE.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'reference': str(REFERENCE), 'sha256': sha256(REFERENCE),
                     'seconds': result['seconds'], 'max_identity_error': max(
                         c['absolute_error'] for r in records.values() for c in r['initial_identity_checks'].values())}, indent=2))


def report():
    reference = json.loads(REFERENCE.read_text())
    assert reference['code_sha256'] == sha256(Path(__file__))
    for name, digest in reference['sources'].items():
        assert sha256(Path(name)) == digest
    fitting_path = ROOT/'analysis/prehead_fitting_summary.json'
    fitting = json.loads(fitting_path.read_text())
    assert fitting['datasets'] == reference['datasets']
    assert reference['sources'][fitting['plan']] == fitting['plan_sha256']
    records = {}
    for task in reference['datasets']:
        v = reference['records'][task]['teacher_residual_energy']
        rows = {}
        for model, metrics in fitting['records'][task]['models'].items():
            losses = metrics['weighted_mse']
            qmean = losses['ce_prehead_mean']
            assert qmean > 0
            pair, shuffled = [(v+qmean-losses[c])/2 for c in ['ce_prehead_paired', 'ce_prehead_shuffled']]
            denominator = math.sqrt(v*qmean)
            assert max(abs(pair), abs(shuffled)) <= denominator+1e-12
            rows[model] = {'teacher_residual_energy': v, 'student_error_to_teacher_class_mean': qmean,
                           'paired_residual_inner_product': pair,
                           'shuffled_residual_inner_product': shuffled,
                           'paired_residual_cosine': pair/denominator,
                           'shuffled_residual_cosine': shuffled/denominator,
                           'pairing_minus_shuffle_inner_product': pair-shuffled,
                           'pairing_minus_shuffle_cosine': (pair-shuffled)/denominator}
        records[task] = rows
    keys = list(next(iter(records.values()))['initial_cached'])
    macro = {model: {k: float(np.mean([records[t][model][k] for t in reference['datasets']]))
                     for k in keys} for model in ['initial_cached', *CONDITIONS]}
    contrasts = {}
    for comparator in ['balanced_ce', 'ce_prehead_mean', 'ce_prehead_shuffled']:
        contrasts[comparator] = {}
        for metric in ['paired_residual_cosine', 'pairing_minus_shuffle_cosine']:
            delta = {t: records[t]['ce_prehead_paired'][metric]-records[t][comparator][metric]
                     for t in reference['datasets']}
            contrasts[comparator][metric] = {'per_task_difference': delta,
                    'mean_difference': float(np.mean(list(delta.values()))),
                    'positive_tasks': sum(x > 0 for x in delta.values()),
                    'negative_tasks': sum(x < 0 for x in delta.values())}
    output = ROOT/'analysis/prehead_pairing_decomposition.json'
    result = {'code_sha256': sha256(Path(__file__)), 'sources': {str(p): sha256(p) for p in [REFERENCE, fitting_path]},
              'datasets': reference['datasets'], 'macro': macro, 'records': records,
              'paired_model_contrasts': contrasts, 'new_forward_examples': 0,
              'optimizer_updates': 0, 'scope': reference['scope']}
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha256(output), 'macro': macro}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'report'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else report()
