"""Report all fixed one-refresh branches and matched restart controls."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    design_path = ROOT/'inputs/one_refresh_intervention_design.json'
    design = json.loads(design_path.read_text())
    prefix_path = ROOT/'seed0/gradient_basis_gradient_balanced_ce/run.json'
    prefix = json.loads(prefix_path.read_text())
    history = json.loads(prefix_path.with_name('history.json').read_text())
    anchor = next(v for v in history if v['step'] == 800)
    tasks = prefix['datasets']
    sources, runs, costs = {}, {}, {}
    for branch in design['conditions']:
        path = ROOT/f'seed0/one_refresh_{branch}/run.json'
        run = json.loads(path.read_text())
        assert run['datasets'] == tasks and run['branch'] == branch
        assert run['optimizer_reset_at_anchor'] and not run['engineering_smoke']
        assert run['start_step'] == 800 and run['end_step'] == 1600 and run['updates'] == 800
        assert run['training_examples'] == 102400 and run['average_steps'] == design['average_steps']
        assert run['plan']['lr'] == design['lr'] and run['training_pool'] == 'original512'
        assert run['plan']['initial_checkpoint_sha256'] == design['anchor_sha256'] == sha(design['anchor'])
        assert run['plan']['design_sha256'] == sha(design_path)
        assert run['schedule_sha256'] == prefix['schedule_sha256']
        assert run['coefficient_shape'] == [32, 383] and run['trainable_parameters'] == 12256
        assert run['fixed_tensors_exact'] and run['initial_parameter_and_probe_replay_exact']
        assert run['mean_coefficient_vs_mean_state_max_abs'] <= 1e-5
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert all(run['initial_validation']['validation'][t]['BAcc'] == anchor['validation'][t]['BAcc'] for t in tasks)
        assert all(run['dataset_audits'][t] == prefix['dataset_audits'][t] for t in tasks)
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        assert run['checkpoint_sha256'] == sha(path.with_name('merged.ckpt'))
        runs[branch] = run
        sources[branch] = {'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': run['checkpoint_sha256']}
        costs[branch] = run['seconds']
    initial_path = ROOT/'seed0/original_fp32_tensor_balanced_avg/run.json'
    initial = json.loads(initial_path.read_text())
    old_path = ROOT/'seed0/full_control_lr1e5_balanced_ce/run.json'
    old = json.loads(old_path.read_text())
    full_path = ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'
    full = json.loads(full_path.read_text())
    joint_path = ROOT/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(joint_path.read_text())
    values = {b: np.array([v['average_validation'][t]['BAcc'] for t in tasks]) for b, v in runs.items()}
    values.update({
        'initial_merger': np.array([initial['folded_val'][t]['BAcc'] for t in tasks]),
        'anchor_step800': np.array([anchor['validation'][t]['BAcc'] for t in tasks]),
        'uninterrupted_gradient': np.array([prefix['average_validation'][t]['BAcc'] for t in tasks]),
        'old512_full_weights': np.array([old['average_validation'][t]['BAcc'] for t in tasks]),
        'full_pool_full_weights': np.array([full['average_validation'][t]['BAcc'] for t in tasks]),
        'Joint': np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])})
    for label, path in [('initial', initial_path), ('uninterrupted', prefix_path), ('old512', old_path),
                         ('full_pool', full_path), ('Joint', joint_path)]:
        sources[label] = {'path': str(path), 'sha256': sha(path)}
    contrasts = {}
    pairs = [('refreshed_gradient', c) for c in values if c not in ['refreshed_gradient', 'Joint']]
    pairs += [('fixed_basis_restart', 'uninterrupted_gradient'), ('refreshed_random', 'fixed_basis_restart')]
    for treatment, control in pairs:
        gains = 100*(values[treatment]-values[control])
        contrasts[treatment+'_vs_'+control] = {
            'macro_gain_pp': float(gains.mean()), 'per_task_gain_pp': dict(zip(tasks, gains.tolist())),
            'improved': int((gains > 0).sum()), 'declined': int((gains < 0).sum()),
            'leave_one_task_out_pp': {t: float(np.delete(gains, i).mean()) for i, t in enumerate(tasks)}}
    cache_path = ROOT/'seed0/trajectory_gradients_gradient_basis_step800/run.json'
    cache = json.loads(cache_path.read_text())
    basis_path = ROOT/'seed0/refreshed_gradient_basis_step800/run.json'
    basis = json.loads(basis_path.read_text())
    initial_cache = json.loads((ROOT/'seed0/local_gradient_cache/run.json').read_text())
    initial_basis = json.loads((ROOT/'seed0/local_gradient_basis/run.json').read_text())
    initial_macro, joint_macro = values['initial_merger'].mean(), values['Joint'].mean()
    result = {
        'unix': time.time(), 'macro_bacc': {k: float(v.mean()) for k, v in values.items()},
        'contrasts': contrasts,
        'fraction_initial_joint_gap_closed': {b: float((v.mean()-initial_macro)/(joint_macro-initial_macro))
                                             for b, v in values.items() if b in runs},
        'calibration_macro_bacc': {b: float(np.mean([v['average_calibration'][t]['BAcc'] for t in tasks])) for b, v in runs.items()},
        'branch_cost_seconds': costs,
        'shared_initial_basis_construction_seconds': initial_cache['seconds']+initial_basis['seconds'],
        'refresh_gradient_cache_seconds': cache['seconds'], 'refresh_basis_build_seconds': basis['seconds'],
        'cost_accounting': {
            'common_prefix_optimization_updates': 800, 'branch_optimization_updates': 800,
            'total_method_optimization_updates': 1600, 'total_optimization_exposures': 204800,
            'initial_basis_backward_exposures': 8192, 'refresh_backward_exposures': 8192,
            'additional_unique_rows_for_refresh': 0,
            'prefix_800_runtime_not_separately_logged': True,
            'runtime_note': 'Each branch reuses an existing step800 prefix. Report its800 updates and original construction as costs; do not claim that branch runtime is end-to-end runtime. Random control uses refreshed geometry/ranks and is charged the same construction here; this is not an intrinsic requirement of arbitrary random bases.'},
        'sources': sources, 'design_source': {'path': str(design_path), 'sha256': sha(design_path)},
        'refresh_sources': {str(p): sha(p) for p in [cache_path, basis_path]},
        'code_sha256': sha(__file__),
        'scope': 'Single bank0 bounded intervention on full16 exposed validation. All branches share the same actual step800 model, subsequent sample order, coefficient count/LR and fresh Adam state. Fixed average includes the common anchor and five later snapshots. Primary contrast tests changing the basis conditional on this restart recipe. Because diagonal Adam depends on basis orientation, this does not separately identify span refresh versus coordinate/preconditioning effects. It does not establish novelty, independent confirmation or an optimizer-independent capacity bound. No selected-checkpoint substitution.'}
    path = ROOT/'analysis/gradient_refresh_summary.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
