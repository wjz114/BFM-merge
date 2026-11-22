"""Verify engineering runs or report the fixed two-arm head-projection experiment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_duration_ce_controls_v2 import verify_average
from report_grouped_ce_controls import contrast

R = Path('runs/paper_merge_v2')
PLAN = R/'inputs/contrast_head_training_plan.json'


def write_new(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


def load_verified(smoke):
    plan = json.loads(PLAN.read_text())
    for p, expected in {**plan['source_sha256'], **plan['contrast_input_sha256']}.items():
        assert sha256(Path(p)) == expected, p
    runs, wrappers, checks, sources = {}, {}, {}, {str(PLAN): sha256(PLAN), str(Path(__file__)): sha256(Path(__file__))}
    for condition in plan['contrast_conditions']:
        folder = R/'seed0'/f'contrast_head_{"smoke" if smoke else "training"}_{condition}'
        rp, wp = folder/'run.json', folder/'wrapper.json'
        run, wrapper = json.loads(rp.read_text()), json.loads(wp.read_text())
        assert run['plan'] == plan
        assert run['plan_sha256'] == wrapper['plan_sha256'] == sha256(PLAN)
        assert wrapper['run_sha256'] == sha256(rp) and wrapper['status'] == 'complete'
        assert wrapper['condition'] == condition and wrapper['engineering_smoke'] == smoke
        assert wrapper['wrapper_sha256'] == sha256(Path('train_contrast_head_adam.py'))
        assert wrapper['deterministic_algorithms'] == smoke
        assert run['updates'] == wrapper['updates'] == (32 if smoke else 1600)
        assert run['training_examples'] == run['updates']*128
        assert run['schedule_sha256'] == plan['schedule_sha256']
        assert run['fixed_tensors_exact'] and run['trainable_parameters'] == 25751552
        assert run['condition'] == 'balanced_ce' and run['scales_sha256'] is None
        assert all(v['auxiliary_scale'] == 0 for v in run['distillation_audits'].values())
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert all(v == smoke for v in run['numerical_flags'].values())
        checks[condition] = verify_average({**run, 'checkpoint': str(folder/'merged.ckpt')}, [16,32] if smoke else plan['average_steps'])
        runs[condition], wrappers[condition] = run, wrapper
        sources.update({str(p): sha256(p) for p in [rp, wp, folder/'merged.ckpt']})
    assert runs['full']['dataset_audits'] == runs['contrast']['dataset_audits']
    assert runs['full']['datasets'] == runs['contrast']['datasets'] == plan['contrast_datasets']
    assert wrappers['contrast']['maximum_exact_source_contrast_relative_error'] < 1e-11
    return plan, runs, wrappers, checks, sources


def engineering():
    plan, runs, wrappers, checks, sources = load_verified(True)
    reference = R/'seed0/row_projected_smoke_full'
    reference_run = json.loads((reference/'run.json').read_text())
    current = R/'seed0/contrast_head_smoke_full'
    equality = {}
    for filename in ['step_16.ckpt', 'step_32.ckpt', 'final.ckpt', 'merged.ckpt']:
        a = torch.load(current/filename, map_location='cpu', weights_only=True)
        b = torch.load(reference/filename, map_location='cpu', weights_only=True)
        assert list(a) == list(b)
        assert all(torch.equal(a[n], b[n]) for n in a), filename
        equality[filename] = {'tensor_count': len(a), 'all_tensors_exact': True}
        sources[str(reference/filename)] = sha256(reference/filename)
    for filename in ['history.json', 'dynamics.jsonl']:
        assert (current/filename).read_bytes() == (reference/filename).read_bytes(), filename
        sources[str(reference/filename)] = sha256(reference/filename)
    for key in ['average_validation', 'average_calibration', 'initial_validation', 'dataset_audits']:
        assert runs['full'][key] == reference_run[key], key
    sources[str(reference/'run.json')] = sha256(reference/'run.json')
    result = {'unix': time.time(), 'sources': sources, 'plan_sha256': sha256(PLAN),
              'full_ce_reference_tensors_exact': equality,
              'full_ce_reference_histories_dynamics_and_metrics_exact': True,
              'average_reconstruction_checks': checks, 'contrast_runtime_checks': wrappers['contrast'],
              'engineering_optimizer_updates': 64, 'ready_for_scientific_training': True,
              'scope': 'Deterministic32-step engineering gate. No scientific validation-based parameter selection; no claim of full-run native bitwise reproducibility.'}
    path = R/'analysis/contrast_head_engineering_verification.json'
    write_new(path, result)
    print(json.dumps({'output': str(path), 'sha256': sha256(path), 'ready': True}), flush=True)


def scientific():
    plan, runs, wrappers, checks, sources = load_verified(False)
    gate_path = R/'analysis/contrast_head_engineering_verification.json'
    gate = json.loads(gate_path.read_text())
    assert gate['ready_for_scientific_training'] and gate['plan_sha256'] == sha256(PLAN)
    sources[str(gate_path)] = sha256(gate_path)
    tasks = plan['contrast_datasets']
    values = {k: np.array([r['average_validation'][t]['BAcc'] for t in tasks]) for k, r in runs.items()}
    initial_differences = []
    for task in tasks:
        a, b = [runs[k]['initial_validation']['validation'][task] for k in ['full','contrast']]
        for metric in a:
            if a[metric] != b[metric]:
                initial_differences.append({'task': task, 'metric': metric, 'full': a[metric], 'contrast': b[metric], 'difference': b[metric]-a[metric]})
        for metric in ['BAcc','wF1','Kappa']:
            assert a[metric] == b[metric], (task, metric)
    comparison = contrast(values['contrast'], values['full'], tasks)
    means = {k: float(v.mean()) for k, v in values.items()}
    jp = R/'analysis/bank_joint_fp32_validation/run.json'
    joint = json.loads(jp.read_text())
    jm = float(np.mean([joint['evaluations']['0'][t]['BAcc'] for t in tasks]))
    sources[str(jp)] = sha256(jp)
    initial = runs['full']['initial_validation']['macro_bacc']
    hp = R/'seed0/row_projected_training_full_retry2/run.json'
    historical = json.loads(hp.read_text())
    sources[str(hp)] = sha256(hp)
    old = np.array([historical['average_validation'][t]['BAcc'] for t in tasks])
    result = {'unix': time.time(), 'sources': sources, 'plan_sha256': sha256(PLAN), 'datasets': tasks,
              'macro_bacc': means, 'primary_contrast': comparison,
              'initial_macro_bacc': initial, 'joint_macro_bacc': jm,
              'remaining_joint_gap_pp': {k: 100*(jm-v) for k, v in means.items()},
              'fraction_initial_joint_gap_closed': {k: (v-initial)/(jm-initial) for k, v in means.items()},
              'fresh_full_minus_recent_full': contrast(values['full'], old, tasks),
              'initial_metric_differences': initial_differences,
              'initial_bacc_wf1_kappa_summaries_exact': True,
              'average_reconstruction_checks': checks, 'contrast_runtime_checks': wrappers['contrast'],
              'process_seconds': {k:r['seconds'] for k,r in runs.items()},
              'scientific_optimizer_updates': 3200, 'engineering_optimizer_updates': 64,
              'scope': plan['scope']+' Task-deletion ranges are sensitivity checks, not confidence intervals. Fresh-full versus recent-full comparison is descriptive native variation. Initial logits and predictions were not saved for exact replay; metric differences are explicit.'}
    path = R/'analysis/contrast_head_training_summary.json'
    write_new(path, result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax,bx) = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1,1.6]})
    ax.bar([0,1],[100*means[k] for k in ['full','contrast']],color=['#64748b','#0072B2'])
    for i,k in enumerate(['full','contrast']):
        ax.text(i,100*means[k]+.08,f'{100*means[k]:.3f}',ha='center')
    ax.axhline(100*jm,ls='--',color='#333333',label='Historical Joint')
    ax.axhline(100*initial,ls=':',color='#777777',label='Initial merger')
    ax.set_xticks([0,1],['Full CE','Head contrast'])
    ax.set_ylim(100*min(initial,*means.values())-.7,100*max(jm,*means.values())+.7)
    ax.set_ylabel('Validation macro BAcc (%)'); ax.legend(fontsize=8)
    ax.set_title('A. Fixed six-checkpoint means',loc='left')
    delta=100*(values['contrast']-values['full'])
    bx.barh(np.arange(len(tasks)),delta,color=['#0072B2' if v>=0 else '#D55E00' for v in delta])
    bx.set_yticks(np.arange(len(tasks)),tasks,fontsize=8); bx.invert_yaxis()
    bx.axvline(0,color='#777777',lw=.8)
    bx.set_xlabel('Head contrast minus full CE (pp)')
    bx.set_title(f'B. All16 tasks; macro {comparison["macro_gain_pp"]:+.3f} pp',loc='left')
    fig.text(.5,.012,'One expert bank; exposed development validation. Same CE, data and1600updates. Only applied final-head Adam steps are projected.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for extension in ['png','pdf']:
        fig.savefig(R/f'analysis/figures/contrast_head_training.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output':str(path),'sha256':sha256(path),'macro_bacc':means,'contrast_gain_pp':comparison['macro_gain_pp']},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['engineering','scientific'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    engineering() if args.mode == 'engineering' else scientific()
