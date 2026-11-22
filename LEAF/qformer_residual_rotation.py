"""Fixed-width query-stream orthogonal symmetry preserving the LayerNorm mean axis."""
import copy

import torch


def mean_zero_basis(dimension):
    a=torch.eye(dimension,dtype=torch.float64)[:,:-1]
    a[-1,:]=-1
    return torch.linalg.qr(a,mode='reduced').Q


def constrained_procrustes(cross):
    dimension=len(cross)
    b=mean_zero_basis(dimension).to(cross)
    u,s,vh=torch.linalg.svd(b.T@cross@b,full_matrices=False)
    r=torch.ones_like(cross)/dimension+b@(u@vh)@b.T
    return r,s


def rotate_qformer_state(state,rotation,prefix='qformer.'):
    """Return ordinary parameters for q_new=q_old R; no model architecture changes.

    Absorb each query LayerNorm affine map into its input projections, rotate
    residual outputs, and fold the final affine norm into the existing head.
    LEAF uses equal query/key/value embedding widths and packed projections.
    """
    result={k:v.clone() for k,v in state.items()}
    d=rotation.shape[0];r=rotation.double()
    eye=torch.eye(d,dtype=r.dtype,device=r.device)
    assert float((r.T@r-eye).abs().max())<1e-11
    assert float((r@torch.ones(d,dtype=r.dtype,device=r.device)-1).abs().max())<1e-11

    def value(name):return state[prefix+name].double()
    def put(name,x):result[prefix+name]=x.to(state[prefix+name])

    put('base_queries',value('base_queries')@r)
    layer=0
    while prefix+f'layers.{layer}.norm1.weight' in state:
        stem=f'layers.{layer}.'
        for norm,attention,only_query in [('norm1','self_attn',False),('norm2','cross_attn',True)]:
            gamma,beta=value(stem+norm+'.weight'),value(stem+norm+'.bias')
            w,bias=value(stem+attention+'.in_proj_weight'),value(stem+attention+'.in_proj_bias')
            assert w.shape==(3*d,d)
            neww,newb=w.clone(),bias.clone();sl=slice(0,d) if only_query else slice(None)
            neww[sl]=(w[sl]*gamma)@r
            newb[sl]=bias[sl]+w[sl]@beta
            put(stem+attention+'.in_proj_weight',neww);put(stem+attention+'.in_proj_bias',newb)
            put(stem+attention+'.out_proj.weight',r.T@value(stem+attention+'.out_proj.weight'))
            put(stem+attention+'.out_proj.bias',r.T@value(stem+attention+'.out_proj.bias'))
            put(stem+norm+'.weight',torch.ones_like(gamma));put(stem+norm+'.bias',torch.zeros_like(beta))
        gamma,beta=value(stem+'norm3.weight'),value(stem+'norm3.bias')
        w=value(stem+'ff.0.weight')
        put(stem+'ff.0.weight',(w*gamma)@r)
        put(stem+'ff.0.bias',value(stem+'ff.0.bias')+w@beta)
        put(stem+'ff.2.weight',r.T@value(stem+'ff.2.weight'))
        put(stem+'ff.2.bias',r.T@value(stem+'ff.2.bias'))
        put(stem+'norm3.weight',torch.ones_like(gamma));put(stem+'norm3.bias',torch.zeros_like(beta))
        layer+=1
    assert layer>0
    gamma,beta=value('final_norm.weight'),value('final_norm.bias')
    w=value('output_embed.2.weight');blocks=w.reshape(w.shape[0],-1,d)
    put('output_embed.2.weight',((blocks*gamma)@r).reshape_as(w))
    put('output_embed.2.bias',value('output_embed.2.bias')+torch.einsum('oqd,d->o',blocks,beta))
    put('final_norm.weight',torch.ones_like(gamma));put('final_norm.bias',torch.zeros_like(beta))
    return result


def toy_checks():
    from LEAF import TextConditionedQFormer
    torch.set_num_threads(4);torch.manual_seed(20261116)
    model=TextConditionedQFormer(num_q=3,dim_q=8,dim_kv=8,n_heads=2,n_layers=2,text_dim=6,max_tokens=6).double().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,torch.nn.LayerNorm):
                module.weight.copy_(torch.randn_like(module.weight)*.3+1)
                module.bias.copy_(torch.randn_like(module.bias)*.2)
    tokens=torch.randn(4,9,8,dtype=torch.float64);instruction=torch.randn(4,6,dtype=torch.float64)
    b=mean_zero_basis(8);q=torch.linalg.qr(torch.randn(7,7,dtype=torch.float64)).Q
    rotation=torch.ones(8,8,dtype=torch.float64)/8+b@q@b.T
    errors={}
    with torch.no_grad():
        original,hidden=model(tokens,instruction,return_all_q=True)
        for name,r in [('affine_fold_identity',torch.eye(8,dtype=torch.float64)),('mean_preserving_rotation',rotation)]:
            transformed=copy.deepcopy(model)
            transformed.load_state_dict(rotate_qformer_state(model.state_dict(),r,prefix=''),strict=True)
            actual,actual_hidden=transformed(tokens,instruction,return_all_q=True)
            output_error=float((actual-original).abs().max())
            hidden_error=max(float((a-h@r).abs().max()) for a,h in zip(actual_hidden,hidden))
            assert output_error<1e-11 and hidden_error<1e-11
            errors[name]={'raw_output_max_error':output_error,'residual_equivariance_max_error':hidden_error}
    x=torch.randn(100,8,dtype=torch.float64)
    recovered,_=constrained_procrustes(x.T@(x@rotation))
    error=float((recovered-rotation).abs().max());assert error<1e-11
    return {'actual_qformer_checks':errors,'known_rotation_recovery_max_error':error,
            'toy_forward_examples':12,'new_eeg_forward_examples':0}
