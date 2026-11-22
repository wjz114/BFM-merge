"""Aggregate all three fixed sampling-order replicas without selecting a seed."""
import json
from pathlib import Path

import numpy as np

from d_downstream_qformer_ce import sha256


ROOT = Path('runs/paper_merge_v2')


def main():
    paths = [ROOT/'seed0/full_control_lr1e5_balanced_ce/run.json']
    paths += [ROOT/f'seed0/full_control_lr1e5_sampler_seed{s}_balanced_ce/run.json' for s in [1, 2]]
    runs = [json.loads(p.read_text()) for p in paths]
    ref = runs[0]
    tasks = ref['datasets']
    joint = json.loads((ROOT/'analysis/full_model_controls_lr1e5_summary.json').read_text())
    joint_bacc = joint['summaries']['initial']['macro_bacc']+joint['summaries']['initial']['joint_gap_pp']/100
    initial = float(np.mean([ref['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    gain_matrix, rows = [], []
    for seed, (path, run) in enumerate(zip(paths, runs)):
        assert run['plan']['seed'] == seed
        assert run['condition'] == 'balanced_ce' and run['plan']['lr'] == 1e-5
        assert run['datasets'] == tasks and not run['engineering_smoke']
        for key in ['initial_checkpoint_sha256', 'calibration_sha256', 'source_fit', 'steps', 'batch_size', 'eval_every', 'average_steps', 'source_sha256']:
            assert run['plan'][key] == ref['plan'][key], (seed, key)
        assert run['fixed_tensors_exact'] and run['updates'] == 1600
        assert sha256(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
        assert all(run['export_checks'][t]['max_abs_logit_error'] <= 1e-5 for t in tasks)
        assert all(run['initial_validation']['validation'][t]['BAcc'] == ref['initial_validation']['validation'][t]['BAcc'] for t in tasks)
        gains = [100*(run['average_validation'][t]['BAcc']-ref['initial_validation']['validation'][t]['BAcc']) for t in tasks]
        gain_matrix.append(gains)
        macro = float(np.mean([run['average_validation'][t]['BAcc'] for t in tasks]))
        assert abs(macro-run['average_validation_macro_bacc']) < 1e-12
        rows.append({'sampling_seed': seed, 'source': str(path), 'source_sha256': sha256(path),
                     'schedule_sha256': run['schedule_sha256'], 'macro_bacc': macro,
                     'gain_pp': float(np.mean(gains)), 'joint_gap_pp': 100*(joint_bacc-macro),
                     'fraction_initial_joint_gap_closed': (macro-initial)/(joint_bacc-initial),
                     'improved_tasks': sum(v>0 for v in gains), 'declined_tasks': sum(v<0 for v in gains),
                     'leave_one_task_out_gain_pp': {t: float(np.mean([v for k,v in zip(tasks,gains) if k != t])) for t in tasks},
                     'selected_macro_bacc': float(np.mean([run['selected_validation'][t]['BAcc'] for t in tasks])),
                     'final_macro_bacc': float(np.mean([run['final_validation'][t]['BAcc'] for t in tasks])),
                     'calibration_macro_bacc': float(np.mean([run['average_calibration'][t]['BAcc'] for t in tasks])),
                     'seconds': run['seconds'], 'training_examples': run['training_examples']})
    assert len({r['schedule_sha256'] for r in rows}) == 3
    matrix = np.array(gain_matrix)
    macro_gains = matrix.mean(1)
    per_task = {task: {'gains_pp': matrix[:,i].tolist(), 'mean_gain_pp': float(matrix[:,i].mean()),
                       'sample_sd_pp': float(matrix[:,i].std(ddof=1)),
                       'positive_seeds': int((matrix[:,i]>0).sum()), 'negative_seeds': int((matrix[:,i]<0).sum())}
                for i,task in enumerate(tasks)}
    result = {'initial_macro_bacc': initial, 'joint_macro_bacc': joint_bacc, 'initial_joint_gap_pp': 100*(joint_bacc-initial),
              'replicas': rows, 'per_task': per_task,
              'mean_macro_bacc': float(np.mean([r['macro_bacc'] for r in rows])),
              'mean_gain_pp': float(macro_gains.mean()), 'sample_sd_gain_pp': float(macro_gains.std(ddof=1)),
              'range_gain_pp': [float(macro_gains.min()), float(macro_gains.max())],
              'mean_fraction_initial_joint_gap_closed': float(np.mean([r['fraction_initial_joint_gap_closed'] for r in rows])),
              'all_three_positive_tasks': [t for t in tasks if per_task[t]['positive_seeds'] == 3],
              'all_three_negative_tasks': [t for t in tasks if per_task[t]['negative_seeds'] == 3],
              'mean_leave_one_task_out_gain_pp': {t: float(np.delete(matrix,i,axis=1).mean()) for i,t in enumerate(tasks)},
              'total_replica_seconds': float(sum(r['seconds']['total'] for r in rows)),
              'scope': 'All three sampling-order replicas, fixed seed0 expert bank, initial merger and original512 calibration IDs. Only task/example order changes. Descriptive sample SD across three schedules; not full-pipeline replication, calibration-source replication, statistical significance, or independent evaluation.',
              'code_sha256': sha256(Path(__file__))}
    output = ROOT/'analysis/full_control_sampling_summary.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['mean_macro_bacc','mean_gain_pp','sample_sd_gain_pp','range_gain_pp','mean_fraction_initial_joint_gap_closed','all_three_positive_tasks','all_three_negative_tasks']}, indent=2))


if __name__ == '__main__':
    main()
