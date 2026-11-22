"""LEAF layer statistics and RegMean solves, including fused attention weights.

Algorithm reference: nthehai01/RegMean-plusplus, commit
a9fdd97629707062a5872653dd9b34f7e25c7243 (MIT; archived under inputs/external).
The implementation below observes ordinary LEAF layers without changing them.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def tensor_tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(tensor_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    return value


@contextmanager
def capture_layer_inputs(model, names):
    captured, handles = {}, []
    modules = dict(model.named_modules())
    try:
        for name in names:
            def hook(module, args, kwargs, name=name):
                captured[name] = tensor_tree((args, kwargs), 'cpu')
            handles.append(modules[name].register_forward_pre_hook(hook, with_kwargs=True))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def attention_context(module, args, kwargs):
    """Run PyTorch's own attention with an identity output projection.

    nn.MultiheadAttention calls its output projection functionally, so a hook
    on its out_proj submodule cannot observe this input. No model is modified.
    """
    if not module.batch_first or not module._qkv_same_embed_dim or module.training:
        raise ValueError('Expected LEAF eval-mode batch-first equal-dimension attention')
    query, key, value = args[:3]
    if query is key is value:
        query = key = value = query.transpose(0, 1)
    else:
        query, key, value = [x.transpose(0, 1) for x in (query, key, value)]
    output, _ = F.multi_head_attention_forward(
        query, key, value, module.embed_dim, module.num_heads,
        module.in_proj_weight, module.in_proj_bias, module.bias_k, module.bias_v,
        module.add_zero_attn, module.dropout,
        torch.eye(module.embed_dim, device=query.device, dtype=query.dtype), None,
        training=False, key_padding_mask=kwargs.get('key_padding_mask'),
        need_weights=kwargs.get('need_weights', True), attn_mask=kwargs.get('attn_mask'),
        average_attn_weights=kwargs.get('average_attn_weights', True),
        is_causal=kwargs.get('is_causal', False))
    return output.transpose(0, 1)


@torch.no_grad()
def layer_statistics(layer, batches, device):
    """Token-averaged input Grams for every linear map in one LEAF layer."""
    sums, counts, handles = {}, {}, []
    attention_error = 0.0

    def add(name, x):
        x = x.detach().reshape(-1, x.shape[-1]).float()
        gram = (x.T @ x).cpu().double()
        sums[name] = sums.get(name, 0) + gram
        counts[name] = counts.get(name, 0) + len(x)

    attention_names = [name for name, module in layer.named_modules()
                       if isinstance(module, torch.nn.MultiheadAttention)]
    try:
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                def attention_hook(module, args, kwargs, output, name=name):
                    nonlocal attention_error
                    for suffix, x in zip(('q', 'k', 'v'), args[:3]):
                        add(name + '.in_proj_weight:' + suffix, x)
                    context = attention_context(module, args, kwargs)
                    add(name + '.out_proj.weight', context)
                    reconstructed = F.linear(context, module.out_proj.weight, module.out_proj.bias)
                    attention_error = max(attention_error, (reconstructed - output[0]).abs().max().item())
                handles.append(module.register_forward_hook(attention_hook, with_kwargs=True))
            elif isinstance(module, torch.nn.Linear) and not any(
                    name.startswith(prefix + '.') for prefix in attention_names):
                def linear_hook(module, args, output, name=name):
                    add(name + '.weight', args[0])
                handles.append(module.register_forward_hook(linear_hook))
        for args, kwargs in batches:
            layer(*tensor_tree(args, device), **tensor_tree(kwargs, device))
    finally:
        for handle in handles:
            handle.remove()
    return {name: gram / counts[name] for name, gram in sums.items()}, {
        'attention_reconstruction_max_abs': attention_error, 'tokens_per_map': counts}


def weight_part(state, name):
    if ':' not in name:
        return state[name]
    parameter, part = name.split(':')
    return state[parameter].chunk(3, dim=0)[{'q': 0, 'k': 1, 'v': 2}[part]]


def solve_regmean(weights, grams, alpha):
    """Solve Eq. 2 with mean-per-task Grams and off-diagonal shrinkage.

    FP64 solve replaces explicit inversion. Exactly unobserved input columns
    retain their arithmetic mean; the objective cannot identify them.
    """
    if not 0 <= alpha < 1:
        raise ValueError('Use 0 <= alpha < 1 for the prepared regularized solve')
    weights = [weight.detach().cpu().double() for weight in weights]
    grams = [gram.detach().cpu().double() for gram in grams]
    shrunk = [alpha * gram + (1 - alpha) * torch.diag(gram.diag()) for gram in grams]
    lhs = sum(shrunk)
    rhs = sum(gram @ weight.T for gram, weight in zip(shrunk, weights))
    average = torch.stack(weights).mean(0)
    merged = average.T.clone()
    active = lhs.diag() > 0
    if active.any():
        merged[active] = torch.linalg.solve(lhs[active][:, active], rhs[active])
    residual = (lhs @ merged - rhs).norm() / rhs.norm().clamp_min(1e-30)
    if not torch.isfinite(merged).all() or residual > 1e-8:
        raise RuntimeError(f'RegMean solve failed: relative residual={residual.item():.3g}')
    merged = merged.T

    def cost(value):
        return sum(((value - weight) @ gram * (value - weight)).sum().item()
                   for weight, gram in zip(weights, shrunk))

    return merged.float(), {'relative_solve_residual': residual.item(),
        'inactive_columns': int((~active).sum()), 'objective_average': cost(average),
        'objective_merged': cost(merged), 'weight_delta_from_average_l2': (merged - average).norm().item()}


def merge_layer(states, statistics, alpha):
    names = set(statistics[0])
    if not names or any(set(stats) != names for stats in statistics):
        raise ValueError('All experts must provide the same nonempty linear-map set')
    merged = {name: torch.stack([state[name].cpu() for state in states]).mean(0)
              for name in states[0]}
    diagnostics = {}
    for name in sorted(names):
        value, diagnostics[name] = solve_regmean(
            [weight_part(state, name) for state in states], [stats[name] for stats in statistics], alpha)
        weight_part(merged, name).copy_(value)
    return merged, diagnostics


@torch.no_grad()
def advance_layer(layer, batches, device):
    outputs = []
    for args, kwargs in batches:
        value = layer(*tensor_tree(args, device), **tensor_tree(kwargs, device)).cpu()
        outputs.append(((value, *args[1:]), kwargs))
    return outputs
