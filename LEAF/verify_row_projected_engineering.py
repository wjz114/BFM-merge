"""Check deterministic full-CE replay and both projected native-model smokes."""
import json
from pathlib import Path
import time

import torch

from d_downstream_qformer_ce import sha256


def main():
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    plan_path = root/'inputs/row_projected_training_plan.json'
    plan = json.loads(plan_path.read_text())
    reference_path = root/'seed0/prehead_ce_deterministic_replay'
    reference = json.loads((reference_path/'run.json').read_text())
    conditions = ['full', 'row', 'shuffled']
    runs, wrappers, sources, checks = {}, {}, {}, {}
    for kind in conditions:
        folder = root/f'seed0/row_projected_smoke_{kind}'
        run_path, wrapper_path = folder/'run.json', folder/'wrapper.json'
        run, wrapper = json.loads(run_path.read_text()), json.loads(wrapper_path.read_text())
        assert run['updates'] == wrapper['updates'] == 32
        assert run['engineering_smoke'] and wrapper['engineering_smoke'] and wrapper['deterministic_algorithms']
        assert wrapper['condition'] == kind and wrapper['status'] == 'complete'
        assert wrapper['plan_sha256'] == run['plan_sha256'] == sha256(plan_path)
        assert wrapper['run_sha256'] == sha256(run_path)
        assert wrapper['wrapper_sha256'] == sha256(Path('train_row_projected_adam.py'))
        assert wrapper['frozen_trainer_sha256'] == sha256(Path('train_prehead_distillation.py'))
        assert run['fixed_tensors_exact'] and all(v > 0 for v in run['component_gradient_max'].values())
        assert all(v['max_abs_logit_error'] <= 1e-5 for v in run['export_checks'].values())
        assert all(all(x['margin_within_twice_error'] for x in v['changed_rows']) for v in run['export_checks'].values())
        snapshots = [torch.load(row['path'], map_location='cpu', weights_only=True) for row in run['average_snapshots']]
        averaged = torch.load(folder/'merged.ckpt', map_location='cpu', weights_only=True)
        active = json.loads(Path(plan['projection_design']).read_text())['active']
        reconstructed = {name: torch.stack([d[name].double() for d in snapshots]).mean(0).float()
                         if name in active else snapshots[0][name] for name in snapshots[0]}
        assert all(torch.equal(v, averaged[n]) for n, v in reconstructed.items())
        checks[kind] = {'all_average_tensors_reconstruct_exactly': True, 'tensor_count': len(averaged),
                        'fixed_tensors_exact': True, 'native_export_checks_pass': True}
        if kind != 'full':
            assert wrapper['coordinate_count'] == 1132560 and wrapper['zero_projection_steps'] == 0
            ratios = wrapper['step_norm_ratio_range']
            assert min(ratios) > .998 and max(ratios) < 1.002, ratios
            assert wrapper['average_coordinate_audit']['max_abs_weight_mean_vs_coordinate_mean'] < 2e-6
            checks[kind].update({k: wrapper[k] for k in ['coordinate_count', 'step_norm_ratio_range',
                                'restoration_scale_range', 'average_coordinate_audit', 'zero_projection_steps']})
        runs[kind], wrappers[kind] = run, wrapper
        sources.update({str(p): sha256(p) for p in [run_path, wrapper_path]})
    full_folder = root/'seed0/row_projected_smoke_full'
    exact = {}
    for name in ['step_16.ckpt', 'step_32.ckpt', 'selected.ckpt', 'final.ckpt', 'merged.ckpt']:
        old = torch.load(reference_path/name, map_location='cpu', weights_only=True)
        new = torch.load(full_folder/name, map_location='cpu', weights_only=True)
        assert list(old) == list(new) and all(torch.equal(v, new[n]) for n, v in old.items()), name
        exact[name] = {'all_tensors_exact': True, 'tensor_count': len(old)}
    for key in ['dataset_audits', 'component_gradient_max', 'initial_validation', 'selected_step',
                'selected_validation', 'final_validation', 'average_validation', 'average_calibration',
                'average_steps', 'schedule_sha256']:
        assert runs['full'][key] == reference[key], key
    for name in ['history.json', 'dynamics.jsonl']:
        assert (full_folder/name).read_text() == (reference_path/name).read_text(), name
    assert all(runs[k]['dataset_audits'] == runs['full']['dataset_audits'] for k in conditions)
    sources[str(reference_path/'run.json')] = sha256(reference_path/'run.json')
    sources[str(plan_path)] = sha256(plan_path)
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)), 'sources': sources,
              'checks': checks, 'full_ce_reference_state_checks': exact,
              'full_ce_reference_metrics_histories_dynamics_exact': True,
              'all_conditions_data_audits_equal': True, 'engineering_optimizer_updates': 96,
              'scope': '32-step engineering verification only. Numerical step-norm tolerance0.2% and mean-coordinate maxabs2e-6 were fixed in this verifier before observing smoke outcomes. No scientific performance threshold, stability or1600-step native bitwise guarantee.'}
    path = root/'analysis/row_projected_engineering_verification.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(path), 'sha256': sha256(path), 'checks': checks,
                      'full_ce_reference_exact': True}, indent=2))


if __name__ == '__main__':
    main()
