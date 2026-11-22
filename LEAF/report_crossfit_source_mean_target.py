"""Audit source-mean target replays against their fixed traces and soup controls."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from report_streaming_replica_filter import audit_export


def average(states,active):
    return {n:((states[0][n].double()+states[1][n].double()+states[2][n].double())/3).float()
            if n in active else v.clone() for n,v in states[0].items()}


def main():
    root=Path('runs/crossfit_source_mean_target_v1')
    plan=json.loads((root/'plan.json').read_text())
    assert not (root/'comparison.json').exists()
    for p,h in {**plan['source_sha256'],**plan['input_sha256']}.items():
        assert sha256(Path(p))==h,p
    source=json.loads(Path(plan['source_fit']).read_text())
    base=torch.load(source['base_checkpoint'],map_location='cpu',weights_only=True)
    fixed=set(base)-set(plan['active'])
    prior_root=Path(plan['coefficient_trace_root'])
    prior_report=json.loads((prior_root/'comparison.json').read_text())
    runs,errors,results={},{},{}
    for condition in ('out_of_fold','in_fold'):
        for order in ('forward','reverse'):
            name=f'{condition}_{order}'
            run=json.loads((root/name/'run.json').read_text())
            fit=json.loads((prior_root/name/'run.json').read_text())
            assert run['status']=='completed' and len(run['stages'])==4
            assert run['plan_sha256']==sha256(root/'plan.json')
            assert run['coefficient_trace_run_sha256']==sha256(prior_root/name/'run.json')
            assert run['order']==fit['order'] and run['condition']==condition
            assert (run['torch_version'],run['cuda_version'])==(fit['torch_version'],fit['cuda_version'])
            assert run['new_optimizer_updates']==run['new_training_CE_examples']==0
            assert run['reused_coefficient_updates']==108 and run['reused_training_CE_examples']==1728
            assert not run['calibration_EEG_read'] and not run['official_test_read']
            assert run['inference_checkpoint_count']==1 and not run['inference_routing']
            assert run['strict_deterministic_algorithms']
            previous=None
            audits=[]
            for index,stage in enumerate(run['stages']):
                count=index+1
                task=run['order'][index]
                assert stage['stage']==count and stage['arriving_task']==task
                paths=[source['experts'][task],*plan['half_experts'][task]]
                assert stage['source_paths']==paths
                sources=[]
                for p in paths:
                    assert sha256(Path(p))==plan['expert_sha256'][p]
                    sources.append(torch.load(p,map_location='cpu',weights_only=True))
                incoming=average(sources,plan['active'])
                directory=root/name/f'stage_{count:02d}'
                path=directory/'merged.ckpt'
                assert path.is_symlink() and sha256(path)==stage['checkpoint_sha256']
                assert str(path.resolve())==stage['checkpoint_storage_path']
                native=torch.load(path,map_location='cpu',weights_only=True)
                assert native.keys()==base.keys()
                assert all(torch.equal(native[n],base[n]) for n in fixed)
                assert all(torch.isfinite(native[n]).all() for n in plan['active'])
                if count==1:
                    assert all(torch.equal(native[n],incoming[n]) for n in base)
                    control=torch.load(prior_root/f'source_soup_{order}'/'stage_01/merged.ckpt',
                                       map_location='cpu',weights_only=True)
                    assert all(torch.equal(native[n],control[n]) for n in base)
                else:
                    path=prior_root/name/f'stage_{count:02d}'/'coefficients.pt'
                    assert str(path)==stage['coefficient_path']
                    assert sha256(path)==stage['coefficient_payload_sha256']==fit['stages'][index]['coefficients_sha256']
                    payload=torch.load(path,map_location='cpu',weights_only=True)
                    assert payload['snapshots'].shape==(6,347666)
                    assert torch.equal(payload['mean'],payload['snapshots'].double().mean(0).float())
                    audits.append(audit_export(base,previous,incoming,native,payload['mean'],plan['active']))
                worlds=[]
                stage_fit=fit['stages'][index]
                hashes=[stage_fit['checkpoint_sha256'],*[p['sha256'] for p in stage_fit['proxies']]]
                for filename,h in zip(('merged.ckpt','proxy0.ckpt','proxy1.ckpt'),hashes):
                    path=prior_root/name/f'stage_{count:02d}'/filename
                    assert sha256(path)==h
                    worlds.append(torch.load(path,map_location='cpu',weights_only=True))
                expected=average(worlds,plan['active'])
                error=max(float((expected[n]-native[n]).abs().max()) for n in plan['active'])
                assert error<2e-6
                audits.append(error)
                if count in plan['validation_stages']:
                    assert set(stage['validation'])==set(run['order'][:count])
                    assert stage['macro_BAcc']==float(np.mean([v['BAcc'] for v in stage['validation'].values()]))
                previous=native
            assert all(run['dataset_audits'][t]['validation_size']==fit['dataset_audits'][t]['validation_size']
                       for t in run['order'])
            runs[name]=run
            errors[name]=max(audits)
            results.setdefault(condition,{})[order]=run['stages'][-1]['macro_BAcc']
    for condition in ('source_soup','single_full'):
        for order in ('forward','reverse'):
            name=f'{condition}_{order}'
            runs[name]=json.loads((prior_root/name/'run.json').read_text())
            assert runs[name]['status']=='completed'
            assert runs[name]['stages'][-1]['macro_BAcc']==prior_report['results'][condition][order]
            results.setdefault(condition,{})[order]=runs[name]['stages'][-1]['macro_BAcc']
    for condition in ('out_of_fold','in_fold'):
        for order in ('forward','reverse'):
            label=f'{condition}_full_target'
            runs[f'{label}_{order}']=json.loads((prior_root/f'{condition}_{order}'/'run.json').read_text())
            results.setdefault(label,{})[order]=runs[f'{label}_{order}']['stages'][-1]['macro_BAcc']
    for v in results.values():
        v['mean']=float(np.mean(list(v.values())))
    contrasts={}
    pairs=[('out_of_fold','in_fold'),('out_of_fold','source_soup'),('in_fold','source_soup'),
           ('out_of_fold','out_of_fold_full_target'),('in_fold','in_fold_full_target'),('out_of_fold','single_full')]
    for a,b in pairs:
        result={order:100*(results[a][order]-results[b][order]) for order in ('forward','reverse')}
        result['mean_pp']=float(np.mean(list(result.values())))
        result['task_deltas_pp']={o:{t:100*(runs[f'{a}_{o}']['stages'][-1]['validation'][t]['BAcc']-
            runs[f'{b}_{o}']['stages'][-1]['validation'][t]['BAcc']) for t in plan['pilot_tasks']}
            for o in ('forward','reverse')}
        contrasts[f'{a}_minus_{b}']=result
    output={'plan_sha256':sha256(root/'plan.json'),'results':results,'contrasts':contrasts,
        'checks':{'four_complete_replays':True,'original_coefficient_traces_unchanged':True,
            'all_native_states_independently_reconstructed':True,'source_mean_commutation_checked_every_stage':True,
            'first_target_equals_source_soup_control':True,'same_sources_and_validation':True},
        'max_reconstruction_errors':errors,'new_scientific_optimizer_updates':0,'new_training_CE_examples':0,
        'source_budget':plan['source_budget'],'overall_goal_achieved':False,'official_test_read':False,
        'scope':'Fixed target change after the direct-target pilot. Four development tasks and two correlated orders; same source and coefficient-training budget. Each replay reuses a previously learned108-update coefficient trace. Affine collapse is an algebraic identity, not a novelty claim. Source fold exclusion is expert fine-tuning only.'}
    with (root/'comparison.json').open('x') as out:
        json.dump(output,out,indent=2)
        out.write('\n')
    print(json.dumps({'results':results,'contrasts':contrasts},indent=2),flush=True)


if __name__=='__main__':
    main()
