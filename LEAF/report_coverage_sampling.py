"""All three full-pool sampling replicas paired with the original512 controls."""
import json
import time

import numpy as np

from report_coverage_control import ROOT, sha


def main():
    pairs = []
    for seed in [0,1,2]:
        full_name = 'full_control_coverage_full_balanced_ce' if seed == 0 else f'full_control_coverage_sampler_seed{seed}_balanced_ce'
        old_name = 'full_control_lr1e5_balanced_ce' if seed == 0 else f'full_control_lr1e5_sampler_seed{seed}_balanced_ce'
        paths = [ROOT/'seed0'/name/'run.json' for name in [full_name,old_name]]
        runs = [json.loads(path.read_text()) for path in paths]
        pairs.append((paths,runs))
    ref = pairs[0][1][0]
    tasks = ref['datasets']
    joint_path = ROOT/'analysis/full_model_controls_lr1e5_summary.json'
    joint = json.loads(joint_path.read_text())['joint_validation_bacc']
    initial = float(np.mean([ref['initial_validation']['validation'][t]['BAcc'] for t in tasks]))
    rows, full_gains, paired_gains = [], [], []
    for seed,(paths,(full,old)) in enumerate(pairs):
        assert full['training_pool'] == 'full_training'
        assert not full['test_data_read'] and not full['test_inference_performed']
        for run in [full,old]:
            assert run['datasets'] == tasks and run['plan']['seed'] == seed
            assert run['condition'] == 'balanced_ce' and not run['engineering_smoke']
            assert run['updates'] == 1600 and run['training_examples'] == 204800
            assert run['plan']['lr'] == 1e-5 and run['fixed_tensors_exact']
            assert all(run['initial_validation']['validation'][t]['BAcc'] == ref['initial_validation']['validation'][t]['BAcc'] for t in tasks)
            assert all(run['export_checks'][t]['max_abs_logit_error'] <= 1e-5 for t in tasks)
            for key in ['initial_checkpoint_sha256','calibration_sha256','source_fit','teacher_cache_sha256',
                        'steps','batch_size','eval_every','average_steps']:
                assert run['plan'][key] == ref['plan'][key], key
        for key in ['source_sha256','coverage_audit_sha256']:
            assert full['plan'][key] == ref['plan'][key], key
        for path,run in zip(paths,[full,old]):
            assert sha(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
        for task in tasks:
            assert full['dataset_audits'][task] == ref['dataset_audits'][task]
            assert full['dataset_audits'][task]['eeg_hash'] == old['dataset_audits'][task]['eeg_hash']
        schedules = [np.load(run['plan']['schedule']) for run in [full,old]]
        assert np.array_equal(schedules[0]['task'],schedules[1]['task'])
        gains = [100*(full['average_validation'][t]['BAcc']-ref['initial_validation']['validation'][t]['BAcc']) for t in tasks]
        contrast = [100*(full['average_validation'][t]['BAcc']-old['average_validation'][t]['BAcc']) for t in tasks]
        full_gains.append(gains);paired_gains.append(contrast)
        macro = float(np.mean([full['average_validation'][t]['BAcc'] for t in tasks]))
        rows.append({'sampling_seed':seed,'full_macro_bacc':macro,'full_gain_over_initial_pp':float(np.mean(gains)),
                     'full_minus_original512_pp':float(np.mean(contrast)),
                     'joint_gap_pp':100*(joint-macro),'fraction_initial_joint_gap_closed':(macro-initial)/(joint-initial),
                     'original512_macro_bacc':old['average_validation_macro_bacc'],
                     'reference_calibration_macro_bacc':float(np.mean([full['average_calibration'][t]['BAcc'] for t in tasks])),
                     'unique_training_rows_sampled':sum(len(np.unique(schedules[0]['local_indices'][schedules[0]['task']==i])) for i in range(len(tasks))),
                     'seconds':full['seconds'],'sources':[{'path':str(path),'sha256':sha(path)} for path in paths]})
    fg,pg = np.array(full_gains),np.array(paired_gains)
    result = {'unix':time.time(),'initial_macro_bacc':initial,'joint_macro_bacc':joint,
              'joint_source':str(joint_path),'joint_source_sha256':sha(joint_path),
              'replicas':rows,'mean_macro_bacc':float(np.mean([r['full_macro_bacc'] for r in rows])),
              'mean_gain_over_initial_pp':float(fg.mean()),'sample_sd_gain_over_initial_pp':float(fg.mean(1).std(ddof=1)),
              'mean_gain_over_original512_pp':float(pg.mean()),'sample_sd_gain_over_original512_pp':float(pg.mean(1).std(ddof=1)),
              'mean_fraction_initial_joint_gap_closed':float(np.mean([r['fraction_initial_joint_gap_closed'] for r in rows])),
              'per_task':{t:{'full_gains_over_initial_pp':fg[:,i].tolist(),'full_minus_original512_pp':pg[:,i].tolist(),
                            'mean_gain_over_initial_pp':float(fg[:,i].mean()),'mean_gain_over_original512_pp':float(pg[:,i].mean())}
                          for i,t in enumerate(tasks)},
              'all_three_improve_over_initial':[t for i,t in enumerate(tasks) if (fg[:,i]>0).all()],
              'all_three_decline_from_initial':[t for i,t in enumerate(tasks) if (fg[:,i]<0).all()],
              'all_three_improve_over_original512':[t for i,t in enumerate(tasks) if (pg[:,i]>0).all()],
              'mean_leave_one_task_out_gain_over_initial_pp':{t:float(np.delete(fg,i,axis=1).mean()) for i,t in enumerate(tasks)},
              'mean_leave_one_task_out_gain_over_original512_pp':{t:float(np.delete(pg,i,axis=1).mean()) for i,t in enumerate(tasks)},
              'total_full_pool_run_seconds':sum(r['seconds']['total'] for r in rows),'code_sha256':sha(__file__),
              'scope':'Three fixed sampling-order replicas after exploratory seed0, paired to the same original512 task orders. Same seed0 expert bank and merger initialization, full original training pools, objective,204800 additional exposures and six-state mean. Increased data access is explicit. Descriptive sample SD; not full-pipeline replication, statistical significance, new-method evidence or independent validation.'}
    output = ROOT/'analysis/full_control_coverage_sampling_summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['mean_macro_bacc','mean_gain_over_initial_pp','sample_sd_gain_over_initial_pp',
          'mean_gain_over_original512_pp','sample_sd_gain_over_original512_pp','mean_fraction_initial_joint_gap_closed',
          'all_three_improve_over_initial','all_three_decline_from_initial','all_three_improve_over_original512']},indent=2))


if __name__ == '__main__':
    main()
