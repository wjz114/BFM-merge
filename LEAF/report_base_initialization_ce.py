"""Compare base and merged initializations under two fixed CE repair rates."""
import json
import time
from pathlib import Path

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    dp = ROOT/'inputs/base_initialization_ce_design_v2.json'
    design = json.loads(dp.read_text())
    native_path = ROOT/'seed0/full_distillation_ce_teacher_kl/run.json'
    native = json.loads(native_path.read_text())
    tasks = native['datasets']
    runs, sources = {}, {}
    for rate in ['lr1e5', 'lr1e4']:
        source = design['matched_existing_controls']['full_weight_'+rate]
        assert sha(source['path']) == source['sha256']
        runs['merged_'+rate] = json.loads(Path(source['path']).read_text())
        sources['merged_'+rate] = source
        path = ROOT/f'seed0/base_initialization_ce_{rate}/run.json'
        run = json.loads(path.read_text())
        reference = runs['merged_'+rate]
        assert run['condition'] == 'balanced_ce' and not run['engineering_smoke']
        assert run['datasets'] == reference['datasets'] == tasks
        assert run['initial_reference_used'] is None and run['fixed_tensors_exact']
        assert run['updates'] == reference['updates'] == 1600
        assert run['training_examples'] == 204800
        assert run['average_steps'] == reference['average_steps'] == [800, 960, 1120, 1280, 1440, 1600]
        assert run['training_pool'] == reference['training_pool'] == 'full_training'
        assert run['schedule_sha256'] == reference['schedule_sha256']
        assert run['trainable_parameters'] == reference['trainable_parameters']
        assert run['dataset_audits'] == native['dataset_audits']
        assert run['plan']['design_sha256'] == sha(dp)
        assert run['plan']['initial_checkpoint_sha256'] == design['initial_checkpoint_sha256']
        for key in ['lr', 'seed', 'batch_size', 'calibration_sha256']:
            assert run['plan'][key] == reference['plan'][key], key
        for task in tasks:
            a, b = run['dataset_audits'][task], reference['dataset_audits'][task]
            assert set(a)-set(b) == {'labels', 'instruction_hash', 'prototypes_hash'}
            assert set(b)-set(a) == {'teacher_hash', 'teacher_expert_sha256'}
            assert all(a[k] == b[k] for k in set(a)&set(b)), task
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        assert sha(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
        assert not run['test_data_read'] and not run['test_inference_performed']
        runs['base_'+rate] = run
        sources['base_'+rate] = {'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': run['checkpoint_sha256']}
    assert runs['base_lr1e5']['initial_validation'] == runs['base_lr1e4']['initial_validation']
    values = {c: np.array([run['average_validation'][t]['BAcc'] for t in tasks]) for c, run in runs.items()}
    for name in ['base', 'merged']:
        values[name+'_initial'] = np.array([runs[name+'_lr1e5']['initial_validation']['validation'][t]['BAcc'] for t in tasks])
    jp = ROOT/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(jp.read_text())
    values['Joint'] = np.array([joint['evaluations']['0'][t]['BAcc'] for t in tasks])
    fit_path = ROOT/'seed0/original_fp32_tensor_balanced_ce/run.json'
    average_path = ROOT/'seed0/original_fp32_tensor_balanced_avg/run.json'
    fit, average = [json.loads(p.read_text()) for p in [fit_path, average_path]]
    assert fit['base_checkpoint_sha256'] == design['initial_checkpoint_sha256']
    assert fit['alpha_updates_actual'] == 1600
    pairs = [(f'merged_{r}', f'base_{r}') for r in ['lr1e5', 'lr1e4']]
    pairs += [(f'{s}_{r}', f'{s}_initial') for s in ['base', 'merged'] for r in ['lr1e5', 'lr1e4']]
    pairs += [(f'{s}_lr1e4', f'{s}_lr1e5') for s in ['base', 'merged']]
    contrasts = {}
    for treatment, control in pairs:
        delta = 100*(values[treatment]-values[control])
        contrasts[treatment+'_vs_'+control] = {
            'macro_gain_pp': float(delta.mean()), 'per_task_gain_pp': dict(zip(tasks, delta.tolist())),
            'improved': int((delta > 0).sum()), 'declined': int((delta < 0).sum()),
            'leave_one_task_out_pp': {t: float(np.delete(delta, i).mean()) for i, t in enumerate(tasks)}}
    interaction = 100*((values['merged_lr1e4']-values['base_lr1e4'])-(values['merged_lr1e5']-values['base_lr1e5']))
    result = {'unix': time.time(), 'code_sha256': sha(__file__), 'datasets': tasks,
              'sources': sources, 'additional_sources': {str(p): sha(p) for p in [dp, native_path, jp, fit_path, average_path]},
              'macro_bacc': {c: float(v.mean()) for c, v in values.items()}, 'contrasts': contrasts,
              'initialization_by_rate_interaction': {'macro_pp': float(interaction.mean()), 'per_task_pp': dict(zip(tasks, interaction.tolist()))},
              'calibration_macro_bacc': {c: float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])) for c, run in runs.items()},
              'process_seconds': {c: run['seconds'] for c, run in runs.items()},
              'preexisting_merged_initialization_cost': {
                  'coefficient_updates': fit['alpha_updates_actual'],
                  'coefficient_training_exposures': fit['alpha_updates_actual']*fit['batch_size'],
                  'coefficient_unique_reference_rows': sum(fit['calibration_sizes'].values()),
                  'coefficient_fit_process_seconds': fit['seconds'],
                  'fit_and_average_process_seconds': average['seconds'],
                  'expert_training_cost_included': False,
                  'note': 'Specialist training costs are additional and not counted here. Process wall time includes loading and evaluation and is not a FLOPs match.'},
              'base_initial_all_metrics_exact_across_rates': True,
              'scope': design['budget_scope']+' Both prespecified rates and same fixed six-state mean are retained. This is a single-bank exposed-validation baseline, not an optimal Joint recipe, new method or independent confirmation.'}
    path = ROOT/'analysis/base_initialization_ce_summary.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'macro_bacc': result['macro_bacc'], 'contrasts': {k: v['macro_gain_pp'] for k, v in contrasts.items()}}, indent=2))


if __name__ == '__main__':
    main()
