"""Freeze a two-shard training-only input-response diagnostic, without fitting."""
import copy
import json
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    root = Path('runs/paper_merge_v2')
    source_path = root/'seed0/original_fp32_tensor_balanced_ce/run.json'
    source = json.loads(source_path.read_text())
    assert sha256(source_path) == 'e774452c3275b9fec4e90b84910ce02ecbb7ca2587774253e37e7fb45ea765d4'
    head_path = root/'seed0/head_calibration_features/run.json'
    head = json.loads(head_path.read_text())
    fresh_data_path = root/'inputs/fresh512_data/run.json'
    fresh_data = json.loads(fresh_data_path.read_text())
    fresh_ids = root/'inputs/fresh512_data/training_indices.npz'
    assert sha256(fresh_ids) == fresh_data['selected_indices_sha256']
    old_ids = Path('runs/aligned_merge_v1/seed0/tensor_balanced_ce/calibration_indices.npz')
    assert sha256(old_ids) == fresh_data['original_calibration_sha256']
    models = {'initial': {'checkpoint': str(root/'seed0/original_fp32_tensor_balanced_avg/merged.ckpt'),
                          'sha256': 'abd0aaf09d606387f12395f6ebc7a9de54c3178699f951a80a3cf413c2c6ee01'}}
    run_paths = []
    for label, folder in [('ce', 'full_control_coverage_full_balanced_ce'), ('kl', 'full_distillation_ce_teacher_kl')]:
        path = root/f'seed0/{folder}/run.json'
        run = json.loads(path.read_text())
        run_paths.append(path)
        assert run['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['schedule_sha256'] == '223eaecd3f5573aba6330e7ca4a6e28775da07179f79342b78a6ab47182c040e'
        models[label] = {'checkpoint': str(path.with_name('merged.ckpt')), 'sha256': run['checkpoint_sha256'],
                         'source_run': str(path), 'source_run_sha256': sha256(path)}
    for spec in models.values():
        assert sha256(Path(spec['checkpoint'])) == spec['sha256']
    experts = {task: {'checkpoint': source['experts'][task], 'sha256': head['records'][task]['expert_sha256']}
               for task in source['datasets']}
    for spec in experts.values():
        assert sha256(Path(spec['checkpoint'])) == spec['sha256']
    sensor = root/'seed0/sensor_subspace_memory'
    covariances = {task: {'path': str(sensor/f'{task}.npz'), 'sha256': sha256(sensor/f'{task}.npz')}
                   for task in source['datasets']}
    inputs = [source_path, head_path, fresh_data_path, fresh_ids, old_ids, sensor/'run.json',
              root/'seed0/fresh512_control_balanced_ce/run.json', root/'inputs/full_control_coverage_data/run.json', *run_paths]
    input_hashes = {str(p): sha256(p) for p in inputs}
    input_hashes.update({row['path']: row['sha256'] for row in covariances.values()})
    source_names = ['diagnose_input_responses_v3.py', 'prepare_input_response_diagnostic_v3.py', 'adapt_experts_ri.py',
                    'aligned_support.py', 'check_merge_export_margins.py', 'd_downstream_qformer_ce.py',
                    'diagnose_sensor_subspaces.py', 'leaf_datasets/shared.py', 'load_config.py', 'load_datasets.py',
                    'LEAF.py', 'configs/LEAF_mpnet.yaml', 'configs/env.yaml']
    source_hashes = {p: sha256(Path(p)) for p in source_names}
    scope = ('Frozen-model, training-only diagnostic of exact centered-logit input Jacobians over allK-1 '
             'orthonormal class contrasts.128class-stratified rows from each existing original512/replacement512 '
             'pool per task, same local positions/labels. Replacement rows are not independent or unseen to '
             'experts/full-pool repairs; ADHD can overlap. Raw, empirical channel-support-projected and '
             'calibration-channel-covariance-weighted metrics are all retained. The covariance model is '
             'temporally independent and does not certify a physiological perturbation distribution. '
             'Existing cached validation outputs/covariances may share files but only training fields are used. '
             'No validation/test EEG reads or inference, no Joint inputs, optimizer updates or candidate export. '
             'Derivative mismatch is not causal evidence of a BAcc gap, and generic Jacobian distillation is prior art.')
    common = {'fixed_unix': time.time(), 'source_fit': str(source_path),
              'coverage_audit': str(root/'inputs/full_control_coverage_data/run.json'),
              'fresh_reference': str(root/'seed0/fresh512_control_balanced_ce/run.json'),
              'head_cache': str(head_path), 'calibration': str(old_ids), 'fresh_indices': str(fresh_ids),
              'models': models, 'experts': experts, 'sensor_covariances': covariances,
              'source_sha256': source_hashes, 'input_sha256': input_hashes,
              'samples_per_pool': 128, 'batch_size': 16, 'seed': 0, 'scope': scope}
    design = {**copy.deepcopy(common), 'all_tasks': source['datasets'],
              'hypothesis': 'Determine whether ordinary output-KL repair improves input-response fidelity beyond CE, including within actual empirical sensor support and on examples with matching teacher/student decisions. This diagnoses whether a derivative-targeted intervention has a distinct unresolved target.',
              'sampling_rule': 'Within each old512 label class, select128/K positions plus one for the first remainder classes, seed[0,task_index,20261116], without replacement. Use exactly those positions in label-matched replacement512. Smoke uses2perclass for OpenBMI and FACED only.',
              'primary': 'For each pool and all16 tasks, compare initial/CE/KL models to own experts using covariance-weighted and support-projected Jacobian relative mismatch, norm ratio and cosine; retain raw metrics, zero-order errors and agreement-conditioned statistics. No choice of best task, metric or model from results.',
              'engineering_gate': 'Before full launch, binary and nine-class smoke must pass old512 native logit replay, input-gradient/native-mode margin checks, all fixed states, Helmert basis identities, batch independence, native-to-float64 gradient relative error<5e-4 and float64 central derivatives at relative steps1e-8/1e-9 with relative error<1e-3. Record original float32 steps1e-4/5e-5 as finite-scale diagnostics without requiring local linearity there. No full Hessians or second parameter derivatives.',
              'literature': ['https://proceedings.mlr.press/v80/srinivas18a/srinivas18a.pdf',
                              'https://proceedings.mlr.press/v243/wu24a/wu24a.pdf'],
              'shards': {str(i): {'tasks': source['datasets'][i::2], 'gpu': i,
                                  'plan': str(root/f'inputs/input_response_v3_shard{i}_plan.json'),
                                  'outdir': str(root/f'seed0/input_response_v3_shard{i}'),
                                  'log': str(root/f'seed0/logs/input_response_v3_shard{i}.log')} for i in range(2)}}
    design_path = root/'inputs/input_response_v3_design.json'
    previous = root/'inputs/input_response_v2_design.json'
    failure = root/'seed0/logs/input_response_v2_smoke.log'
    design['engineering_amendment'] = {'previous_design': str(previous), 'previous_design_sha256': sha256(previous),
        'failure_log': str(failure), 'failure_log_sha256': sha256(failure),
        'numerical_probe': str(root/'analysis/input_response_numerics.json'),
        'numerical_probe_sha256': sha256(root/'analysis/input_response_numerics.json'),
        'correction': 'Version2 failed a finite-difference check at relative input step1e-4. A separate training-only numerical probe reproduced the discrepancy in float64, counted pooling switches, and found convergence to autograd at1e-8/1e-9 with no switches; native and float64 gradients agreed within2.1e-6. Use float64 only for the engineering derivative reference, retaining nativeFP32 diagnostic Jacobians and original larger-step results. Full scientific runs were not launched; model/data/derivative choices are unchanged.'}
    write_new(design_path, design)
    for key, spec in design['shards'].items():
        plan = {**copy.deepcopy(common), 'tasks': spec['tasks'], 'shard': int(key),
                'design': str(design_path), 'design_sha256': sha256(design_path)}
        write_new(Path(spec['plan']), plan)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path), 'shards': design['shards']}, indent=2))


if __name__ == '__main__':
    main()
