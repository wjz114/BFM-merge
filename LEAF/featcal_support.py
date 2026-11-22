"""LEAF observation adapter for the pinned, MIT-licensed official FeatCal solver.

Uses uniform token weights because LEAF has no CLS token. Functional fused
attention projections are observed explicitly; ordinary model modules remain
unchanged. Layer statistics are collected before any within-layer updates.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from regmean_support import attention_context, tensor_tree, weight_part


SOLVER_PATH = Path(__file__).parent/'runs/paper_merge_v2/inputs/external/featcal_56f7fbb9f693/src/featcal/solver.py'
spec = importlib.util.spec_from_file_location('pinned_featcal_solver', SOLVER_PATH)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


@contextmanager
def observe(layer):
    cache, handles, errors = {}, [], []
    attention_names = [n for n,m in layer.named_modules() if isinstance(m,torch.nn.MultiheadAttention)]
    try:
        for name, module in layer.named_modules():
            if isinstance(module,torch.nn.MultiheadAttention):
                def hook(module,args,kwargs,output,name=name):
                    for part,x in zip(['q','k','v'],args[:3]):
                        cache[name+'.in_proj_weight:'+part] = ('linear',x.detach())
                    context = attention_context(module,args,kwargs)
                    cache[name+'.out_proj.weight'] = ('linear',context.detach())
                    reproduced = F.linear(context,module.out_proj.weight,module.out_proj.bias)
                    errors.append(float((reproduced-output[0]).abs().max()))
                handles.append(module.register_forward_hook(hook,with_kwargs=True))
            elif isinstance(module,torch.nn.Linear) and not any(name.startswith(a+'.') for a in attention_names):
                def hook(module,args,name=name):
                    cache[name+'.weight'] = ('linear',args[0].detach())
                handles.append(module.register_forward_pre_hook(hook))
            elif isinstance(module,torch.nn.LayerNorm):
                def hook(module,args,name=name):
                    z = F.layer_norm(args[0],module.normalized_shape,None,None,module.eps)
                    cache[name] = ('layernorm',z.detach())
                handles.append(module.register_forward_pre_hook(hook))
        yield cache, errors
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def paired_statistics(student,expert,student_batches,expert_batches,alpha,device):
    assert len(student_batches) == len(expert_batches)
    accum, kinds, counts, next_expert = {}, {}, {}, []
    with observe(student) as (sc,se), observe(expert) as (ec,ee):
        for (sargs,skw),(eargs,ekw) in zip(student_batches,expert_batches):
            sc.clear(); ec.clear()
            student(*tensor_tree(sargs,device),**tensor_tree(skw,device))
            output = expert(*tensor_tree(eargs,device),**tensor_tree(ekw,device))
            next_expert.append(((output.detach().cpu(),*eargs[1:]),ekw))
            assert sc.keys() == ec.keys()
            for name,(kind,xs) in sc.items():
                ekind,xe = ec[name]
                assert kind == ekind and xs.shape == xe.shape
                xs = xs.reshape(-1,xs.shape[-1]).float()
                xe = xe.reshape(-1,xe.shape[-1]).float()
                if kind == 'linear':
                    xt = alpha*xe+(1-alpha)*xs
                    values = {'sum_xs':xs.sum(0),'sum_xt':xt.sum(0),'sum_ss':xs.T@xs,'sum_st':xs.T@xt}
                else:
                    values = {'sum_z':xs.sum(0),'sum_z2':xs.square().sum(0)}
                kinds[name] = kind
                counts[name] = counts.get(name,0)+len(xs)
                if name not in accum:
                    accum[name] = values
                else:
                    for key,value in values.items():
                        accum[name][key] += value
        max_error = max([0.,*se,*ee])
        assert max_error < 1e-4, max_error
    stats = {}
    for name,values in accum.items():
        means = {k:(v/counts[name]).cpu() for k,v in values.items()}
        if kinds[name] == 'linear':
            stats[name] = {'kind':'linear','mean_student':means['sum_xs'],'mean_teacher':means['sum_xt'],
                           'cov_ss':means['sum_ss'],'cross_st':means['sum_st']}
        else:
            stats[name] = {'kind':'layernorm','mean_student':means['sum_z'],'energy_student':means['sum_z2']}
    return stats,next_expert,{'attention_reconstruction_max_abs':max_error,'tokens_per_map':counts}


def calibrate_state(merged,base,experts,statistics,ridge,rho,eps=1e-8):
    result = {k:v.detach().cpu().clone() for k,v in merged.items()}
    assert all(s.keys() == statistics[0].keys() for s in statistics)
    for name, first in statistics[0].items():
        stats = [s[name] for s in statistics]
        if first['kind'] == 'linear':
            weights = [weight_part(s,name).cpu() for s in experts]
            new_weight = solver.featcal_merge_linear_weight(weights,stats,ridge_lambda=ridge,
                merged_param=weight_part(merged,name).cpu(),base_param=weight_part(base,name).cpu(),
                anchor_blend_rho=rho,covariance_eps=eps)
            weight_part(result,name).copy_(new_weight)
            bias_name = name.replace('weight','bias')
            new_bias = solver.featcal_merge_linear_bias([weight_part(s,bias_name).cpu() for s in experts],weights,stats,new_weight,
                ridge_lambda=ridge,merged_param=weight_part(merged,bias_name).cpu(),base_param=weight_part(base,bias_name).cpu(),
                anchor_blend_rho=rho,covariance_eps=eps)
            weight_part(result,bias_name).copy_(new_bias)
        else:
            wn,bn = name+'.weight',name+'.bias'
            w,b = solver.featcal_merge_layernorm_affine([s[wn].cpu() for s in experts],[s[bn].cpu() for s in experts],stats,
                ridge_lambda=ridge,merged_weight=merged[wn].cpu(),base_weight=base[wn].cpu(),
                merged_bias=merged[bn].cpu(),base_bias=base[bn].cpu(),anchor_blend_rho=rho,covariance_eps=eps)
            result[wn],result[bn] = w,b
    assert all(torch.isfinite(v).all() for v in result.values())
    return result
