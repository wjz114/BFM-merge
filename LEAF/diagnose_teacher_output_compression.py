"""Descriptive class-conditioned compression of cached specialist training outputs."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256


def moments(values, group, weights):
    mean = (weights[:, None]*values).sum(0)
    codes, inverse = group.unique(sorted=True, return_inverse=True)
    mass = torch.zeros(len(codes), dtype=torch.float64).index_add_(0, inverse, weights)
    centers = torch.zeros((len(codes), values.shape[1]), dtype=torch.float64)
    centers.index_add_(0, inverse, weights[:, None]*values)
    centers /= mass[:, None]
    fitted = centers[inverse]
    total = float((weights[:, None]*(values-mean).square()).sum())
    within = float((weights[:, None]*(values-fitted).square()).sum())
    between = float((mass[:, None]*(centers-mean).square()).sum())
    assert abs(total-within-between) < 1e-12*max(1., total)
    return {'total_variance': total, 'within_group_variance': within,
            'between_group_variance': between, 'within_fraction': within/total if total else None,
            'groups': codes.tolist(), 'group_masses': mass.tolist()}, fitted


def entropy(p):
    return -(p*p.log()).sum(-1)


def kl(p, q):
    return (p*(p.log()-q.log())).sum(-1)


def target_comparison(teacher, student, replacement, truth, weights):
    gradient = student-teacher
    other = student-replacement
    error = teacher-replacement
    a = float((weights[:, None]*gradient.square()).sum())
    b = float((weights[:, None]*other.square()).sum())
    dot = float((weights[:, None]*gradient*other).sum())
    residual = float((weights[:, None]*error.square()).sum())
    tc = teacher.argmax(-1) == truth
    groups = {}
    for name, mask in [('teacher_correct', tc), ('teacher_wrong', ~tc)]:
        mass = float(weights[mask].sum())
        groups[name] = {'weighted_mass': mass,
                        'teacher_to_target_kl': float((weights[mask]*kl(teacher[mask], replacement[mask])).sum()/mass) if mass else None,
                        'probability_squared_error': float((weights[mask, None]*error[mask].square()).sum()/mass) if mass else None}
    return {'teacher_to_target_kl': float((weights*kl(teacher, replacement)).sum()),
            'probability_squared_error': residual,
            'initial_student_kl_output_gradient_squared_norm': a,
            'replacement_output_gradient_squared_norm': b,
            'relative_output_gradient_l2_change': float(np.sqrt(residual/a)) if a else None,
            'pooled_output_gradient_cosine': dot/np.sqrt(a*b) if a and b else None,
            'conditional': groups}


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    target = root/'analysis/teacher_output_compression.json'
    assert not target.exists()
    cache_path = root/'seed0/full_model_distillation_targets/run.json'
    reference_path = root/'analysis/decision_disagreements.json'
    cache, reference = [json.loads(p.read_text()) for p in [cache_path, reference_path]]
    schedule_path = Path(cache['plan']['schedule'])
    schedule = np.load(schedule_path)
    hashes = {str(p): sha256(p) for p in [cache_path, reference_path, schedule_path, Path(__file__)]}
    rows = {}
    for ti, task in enumerate(cache['datasets']):
        spec = cache['records'][task]
        path = Path(spec['target_path'])
        assert sha256(path) == spec['target_sha256']
        hashes[str(path)] = spec['target_sha256']
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        ids = torch.from_numpy(schedule['local_indices'][schedule['task'] == ti].reshape(-1).astype(np.int64))
        positions = torch.searchsorted(data['pool_indices'], ids)
        assert torch.equal(data['pool_indices'][positions], ids) and len(ids) == 12800
        truth = data['truth'][positions]
        weights = torch.tensor(spec['class_weights'], dtype=torch.float64)[truth]
        weights /= weights.sum()
        logits = data['teacher_logits'][positions].double()
        student_logits = data['initial_logits'][positions].double()
        probability, student_probability = [z.softmax(-1) for z in [logits, student_logits]]
        centered = logits-logits.mean(-1, keepdim=True)
        unit = data['teacher_unit'][positions].double()
        prototypes = data['prototypes'].double()
        prediction = logits.argmax(-1)
        k = logits.shape[1]
        tc = prediction == truth
        accuracy = float(weights[tc].sum())
        old = reference['full_stream'][task]
        assert abs(accuracy-old['teacher_weighted_accuracy']) < 1e-12
        assert abs(float((weights*kl(probability, student_probability)).sum())-old['weighted_teacher_kl']) < 1e-12
        overall_mean = (weights[:, None]*probability).sum(0)
        total_information = float((weights*kl(probability, overall_mean[None])).sum())
        assert abs(total_information-float(entropy(overall_mean)-(weights*entropy(probability)).sum())) < 1e-12
        conditioned = {}
        for name, group in [('truth', truth), ('teacher_argmax', prediction), ('truth_and_teacher_argmax', truth*k+prediction)]:
            representations = {}
            for label, values in [('centered_logits', centered), ('probabilities', probability), ('unit_features', unit)]:
                description, compressed = moments(values, group, weights)
                representations[label] = description
                if label == 'probabilities':
                    target_probability = compressed
            within = float((weights*kl(probability, target_probability)).sum())
            between = float((weights*kl(target_probability, overall_mean[None])).sum())
            assert abs(total_information-within-between) < 1e-12
            conditioned[name] = {
                'representations': representations,
                'output_probability_information_nats': {'total': total_information, 'within_group': within,
                    'between_groups': between, 'within_fraction': within/total_information if total_information else None},
                'target_comparison': target_comparison(probability, student_probability, target_probability, truth, weights)}
        row = {'classes': k, 'weighted_teacher_accuracy': accuracy, 'fitting_exposures': len(ids),
               'unique_fitting_rows': len(ids.unique()), 'teacher_normalized_entropy': float((weights*entropy(probability)).sum()/np.log(k)),
               'conditioned': conditioned}
        if k == 2:
            difference = prototypes[1]-prototypes[0]
            direction = difference/difference.norm()
            template_logits = torch.stack([-direction@prototypes.T, direction@prototypes.T])
            assert abs(float(template_logits[1, 1]-template_logits[1, 0]-difference.norm())) < 1e-12
            assert torch.equal(template_logits.argmax(-1), torch.arange(2))
            template_probability = template_logits.softmax(-1)
            row['binary_analytic_templates'] = {
                'prototype_distance': float(difference.norm()), 'logits': template_logits.tolist(),
                'probabilities': template_probability.tolist(),
                'true_label_selected_teacher_free': target_comparison(probability, student_probability, template_probability[truth], truth, weights),
                'teacher_prediction_selected': target_comparison(probability, student_probability, template_probability[prediction], truth, weights)}
        rows[task] = row
    groups = {'all16': list(rows), 'binary8': [t for t,r in rows.items() if r['classes'] == 2],
              'multiclass8': [t for t,r in rows.items() if r['classes'] > 2]}
    assert len(groups['binary8']) == len(groups['multiclass8']) == 8
    summaries = {}
    for label, tasks in groups.items():
        summaries[label] = {'tasks': tasks, 'teacher_accuracy_mean': float(np.mean([rows[t]['weighted_teacher_accuracy'] for t in tasks])),
                            'teacher_normalized_entropy_mean': float(np.mean([rows[t]['teacher_normalized_entropy'] for t in tasks])),
                            'conditioned': {}}
        for group in ['truth', 'teacher_argmax', 'truth_and_teacher_argmax']:
            entries = [rows[t]['conditioned'][group] for t in tasks]
            summary = {'variance_within_fraction': {}, 'information_within_fraction': {}, 'target_comparison_mean': {}}
            for rep in ['centered_logits', 'probabilities', 'unit_features']:
                vals = [e['representations'][rep]['within_fraction'] for e in entries]
                summary['variance_within_fraction'][rep] = {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'range': [min(vals), max(vals)]}
            vals = [e['output_probability_information_nats']['within_fraction'] for e in entries]
            summary['information_within_fraction'] = {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'range': [min(vals), max(vals)]}
            for key in ['teacher_to_target_kl', 'probability_squared_error', 'relative_output_gradient_l2_change', 'pooled_output_gradient_cosine']:
                summary['target_comparison_mean'][key] = float(np.mean([e['target_comparison'][key] for e in entries]))
            summaries[label]['conditioned'][group] = summary
    binary_summary = {name: {key: float(np.mean([rows[t]['binary_analytic_templates'][name][key] for t in groups['binary8']]))
                            for key in ['teacher_to_target_kl', 'probability_squared_error', 'relative_output_gradient_l2_change', 'pooled_output_gradient_cosine']}
                      for name in ['true_label_selected_teacher_free', 'teacher_prediction_selected']}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'source_sha256': hashes,
              'datasets': cache['datasets'], 'per_task': rows, 'summaries': summaries,
              'binary_analytic_template_summary': binary_summary, 'seconds': time.time()-started,
              'optimizer_updates': 0, 'new_model_inference': False, 'test_data_read': False,
              'identities_checked': ['Weighted total=between+within variance for every representation and grouping.',
                  'Weighted KL(q_i||q_bar)=weighted KL(q_i||q_group)+weighted KL(q_group||q_bar).',
                  'Full-stream weighted teacher accuracy and teacher-to-initial KL replay prior audit.',
                  'Binary unit-feature templates attain the exact prototype-distance logit-margin bound.'],
              'scope': 'Descriptive in-sample compression of the existing full fitting multiset; original full-pool class weights normalized per task. Empirical class means use expert outputs and are NOT teacher-free. Teacher-argmax compression still contains expert decisions. Only binary true-label-selected analytic templates use labels/prototypes alone. The probability KL decomposition concerns a class drawn from the teacher distribution on this empirical row distribution; it is not a measure of all task knowledge. Output-gradient comparisons omit parameter Jacobians and optimizer dynamics. Small compression errors do not establish equivalent training, causal explanation of KL performance, or independent generalization.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('Saved', target, sha256(target), 'seconds', result['seconds'])
    for label, summary in summaries.items():
        print(label, 'teacher accuracy', summary['teacher_accuracy_mean'], 'normalized entropy', summary['teacher_normalized_entropy_mean'])
        for group, row in summary['conditioned'].items():
            print(group, 'logit within fraction', row['variance_within_fraction']['centered_logits'],
                  'information within fraction', row['information_within_fraction'],
                  'output gradients', row['target_comparison_mean'])
    print('Binary analytic templates', json.dumps(binary_summary, indent=2))


if __name__ == '__main__':
    main()
