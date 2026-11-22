"""Report the one fixed all28-block FeatCal adaptation, including negative tasks."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path('runs/paper_merge_v2')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    path = ROOT/'seed0/featcal_fixed_full16/run.json'
    run = json.loads(path.read_text())
    assert not run['engineering_smoke'] and len(run['datasets']) == 16
    assert run['outside_blocks_exact'] and not run['test_data_read'] and not run['test_inference_performed']
    assert not run['fitting_labels_used'] and run['optimizer_updates'] == 0
    assert sha(path.with_name('merged.ckpt')) == run['checkpoint_sha256']
    tasks = run['datasets']
    assert len(run['updated_layers']) == 28
    assert all(run['actual_instruction_wrapper_checks'][t] == 0 for t in tasks)
    assert all(run['export_checks'][t]['max_abs_logit_error'] <= 1e-5 for t in tasks)
    layers = [json.loads(line) for line in path.with_name('layers.jsonl').read_text().splitlines()]
    assert [row['layer'] for row in layers] == run['updated_layers']
    assert [row['maps'] for row in layers] == [8]*24+[13]*4
    control_paths = {'original512_ce': ROOT/'seed0/full_control_lr1e5_balanced_ce/run.json',
                     'full_training_ce':ROOT/'seed0/full_control_coverage_full_balanced_ce/run.json'}
    controls = {name:json.loads(p.read_text()) for name,p in control_paths.items()}
    initial = run['initial_validation']
    for name, control in controls.items():
        assert control['datasets'] == tasks
        assert control['plan']['initial_checkpoint_sha256'] == run['plan']['initial_checkpoint_sha256']
        assert control['plan']['calibration_sha256'] == run['plan']['calibration_sha256']
        assert all(initial[t]['BAcc'] == control['initial_validation']['validation'][t]['BAcc'] for t in tasks)
        assert all(run['data_audits'][t]['calibration_eeg_sha256'] == control['dataset_audits'][t]['eeg_hash'] for t in tasks)
    metrics = {'initial': initial, 'featcal': run['final_validation']}
    metrics.update({name:control['average_validation'] for name,control in controls.items()})
    joint_path = ROOT/'analysis/full_model_controls_lr1e5_summary.json'
    joint = json.loads(joint_path.read_text())['joint_validation_bacc']
    initial_macro = float(np.mean([initial[t]['BAcc'] for t in tasks]))
    summaries = {}
    for name, values in metrics.items():
        macro = float(np.mean([values[t]['BAcc'] for t in tasks]))
        summaries[name] = {'macro_bacc':macro,'gain_over_initial_pp':100*(macro-initial_macro),
                           'joint_gap_pp':100*(joint-macro),
                           'fraction_initial_joint_gap_closed':(macro-initial_macro)/(joint-initial_macro)}
    assert abs(summaries['featcal']['macro_bacc']-run['final_validation_macro_bacc']) < 1e-12
    contrasts = {}
    for name in ['initial','original512_ce','full_training_ce']:
        gains = {t:100*(metrics['featcal'][t]['BAcc']-metrics[name][t]['BAcc']) for t in tasks}
        contrasts['featcal_vs_'+name] = {'mean_gain_pp':float(np.mean(list(gains.values()))),
                                       'per_task_gain_pp':gains,'improved_tasks':sum(v>0 for v in gains.values()),
                                       'declined_tasks':sum(v<0 for v in gains.values()),
                                       'leave_one_task_out_gain_pp':{t:float(np.mean([v for k,v in gains.items() if k!=t])) for t in tasks}}
    result = {'unix':time.time(),'summaries':summaries,'contrasts':contrasts,'joint_macro_bacc':joint,
              'per_task':{t:{name:values[t]['BAcc'] for name,values in metrics.items()} for t in tasks},
              'calibration_macro_bacc':{name:float(np.mean([values[t]['BAcc'] for t in tasks]))
                                       for name,values in [('initial',run['initial_calibration']),('featcal',run['final_calibration'])]},
              'source':str(path),'source_sha256':sha(path),'checkpoint_sha256':run['checkpoint_sha256'],
              'controls':{name:{'source':str(p),'source_sha256':sha(p)} for name,p in control_paths.items()},
              'joint_source':str(joint_path),'joint_source_sha256':sha(joint_path),
              'seconds':run['seconds'],'updated_parameter_elements':run['updated_parameter_elements'],
              'maps':sum(row['maps'] for row in layers),
              'attention_reconstruction_max_abs':max(o['attention_reconstruction_max_abs'] for row in layers for o in row['observations'].values()),
              'maximum_export_logit_error':max(run['export_checks'][t]['max_abs_logit_error'] for t in tasks),
              'outside_blocks_exact':True,'all16_initial_replay_exact':True,'code_sha256':sha(__file__),
              'scope':'Existing FeatCal baseline adapted to LEAF fused attention and uniform tokens, using pinned official solver/paper defaults and a single final28-block checkpoint. Same original512 calibration examples and supervised initial merger as original512 CE. Incremental feature calibration does not use labels; the whole pipeline is not label-free. Full-training CE accesses more examples. Fits have different computation/objectives, so this is not an equal-compute comparison. One exploratory seed on exposed validation; no novelty, statistical significance or independent-evaluation claim.'}
    output = ROOT/'analysis/featcal_summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'featcal_vs_initial':contrasts['featcal_vs_initial'],
                      'calibration_macro_bacc':result['calibration_macro_bacc'],'seconds':run['seconds']},indent=2))


if __name__ == '__main__':
    main()
