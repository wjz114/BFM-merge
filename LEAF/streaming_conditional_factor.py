"""Joint FiLM-scale / cross-attention projection factors, folded into native weights."""
import torch

from streaming_bilateral_merge import StreamingMerger as BilateralMerger

FILM_W = 'qformer.film.weight'
FILM_B = 'qformer.film.bias'
CROSS = tuple(f'qformer.layers.{i}.cross_attn.in_proj_weight' for i in range(4))
WIDTH = 256


def projection_columns(state):
    """One row per modulated input channel; stack all four layers' K/V columns."""
    assert all(state[n].shape == (3 * WIDTH, WIDTH) for n in CROSS)
    return torch.cat([state[n][WIDTH:] for n in CROSS], dim=0).T.contiguous()


@torch.no_grad()
def append_expert(history, state, normalized_instruction, task):
    instruction = normalized_instruction.detach().cpu().double()
    assert instruction.shape == (768,) and abs(float(instruction.norm()) - 1.) < 1e-6
    x = torch.cat([instruction, torch.ones(1, dtype=torch.float64)])
    gamma = state[FILM_W][:WIDTH].double() @ instruction + state[FILM_B][:WIDTH].double()
    history.append({'task': task, 'columns': projection_columns(state).cpu().clone(),
                    'scale': 1 + gamma, 'instruction': x})


def history_bytes(history):
    return sum(v.numel() * v.element_size() for row in history
               for v in row.values() if isinstance(v, torch.Tensor))


@torch.no_grad()
def fit_factors(target, instructions, reference_columns, reference_affine):
    """Best rank-one fit with its task factor in the affine instruction span.

    target: channels x stacked-output-dim x tasks, all FP64 on CPU.
    The gauge matches the reference column norm and orientation. The affine
    instruction weights change only in the observed instruction row space.
    """
    x = instructions.double()
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    keep = singular > 1e-10 * singular.max()
    ux, sx, vx = u[:, keep], singular[keep], vh[keep]
    projector = ux @ ux.T
    pinverse = (vx.T / sx) @ ux.T
    projected = target.double() @ projector
    gram = projected.transpose(-1, -2) @ projected
    _, vectors = torch.linalg.eigh((gram + gram.transpose(-1, -2)) / 2)
    v = vectors[..., -1]
    product = (projected @ v[..., None]).squeeze(-1)
    sigma = product.norm(dim=1)
    unit = product / sigma.clamp_min(1e-300)[:, None]
    orientation = (unit * reference_columns).sum(1)
    fallback = unit.gather(1, unit.abs().argmax(1, keepdim=True)).squeeze(1)
    sign = torch.where(torch.where(orientation != 0, orientation, fallback) < 0, -1., 1.)
    unit, v = unit * sign[:, None], v * sign[:, None]
    norm = reference_columns.norm(dim=1)
    gauge = torch.where(norm > 0, norm, sigma.sqrt())
    columns = unit * gauge[:, None]
    scales = v * (sigma / gauge.clamp_min(1e-300))[:, None]
    zero = sigma == 0
    columns[zero] = reference_columns[zero]
    scales[zero] = 0.
    both_zero = zero & (norm == 0)
    scales[both_zero] = (reference_affine @ x.T)[both_zero]
    affine = reference_affine + (scales - reference_affine @ x.T) @ pinverse.T
    reconstructed_scales = affine @ x.T
    fit = columns[..., None] * reconstructed_scales[:, None, :]
    squared_error = (target - fit).square().sum()
    optimum = (target - projected).square().sum() + projected.square().sum() - sigma.square().sum()
    audit = {'instruction_rank': int(keep.sum()), 'seen_experts': len(x),
             'target_squared_norm': float(target.square().sum()),
             'fit_squared_error': float(squared_error),
             'rank_one_optimum_squared_error': float(optimum.clamp_min(0)),
             'relative_optimality_gap': float((squared_error - optimum).abs() / target.square().sum().clamp_min(1e-300)),
             'max_affine_scale_reconstruction_error': float((scales - reconstructed_scales).abs().max()),
             'max_relative_column_norm_error': float(torch.where(norm > 0,
                 (columns.norm(dim=1) - norm).abs() / norm.clamp_min(1e-300), 0.).max())}
    assert torch.isfinite(columns).all() and torch.isfinite(affine).all()
    return columns, affine, audit


@torch.no_grad()
def factor_correction(reference, history, condition):
    assert condition in ('coupled_factors', 'factorized_control')
    columns = torch.stack([r['columns'].double() for r in history], dim=-1)
    scales = torch.stack([r['scale'] for r in history], dim=-1)
    x = torch.stack([r['instruction'] for r in history])
    if condition == 'factorized_control':
        columns = columns.mean(-1, keepdim=True).expand_as(columns)
    target = columns * scales[:, None, :]
    a0 = projection_columns(reference).cpu().double()
    f0 = torch.cat([reference[FILM_W][:WIDTH].cpu().double(),
                    reference[FILM_B][:WIDTH, None].cpu().double() + 1.], dim=1)
    fitted, affine, audit = fit_factors(target, x, a0, f0)
    result = {n: torch.zeros_like(reference[n], device='cpu') for n in (FILM_W, FILM_B, *CROSS)}
    result[FILM_W][:WIDTH] = affine[:, :-1].float() - reference[FILM_W][:WIDTH].cpu()
    result[FILM_B][:WIDTH] = (affine[:, -1] - 1.).float() - reference[FILM_B][:WIDTH].cpu()
    for i, name in enumerate(CROSS):
        new = fitted[:, i * 2 * WIDTH:(i + 1) * 2 * WIDTH].T.float()
        result[name][WIDTH:] = new - reference[name][WIDTH:].cpu()
    audit['state_bytes'] = history_bytes(history)
    audit['correction_squared_norm'] = sum(float(v.double().square().sum()) for v in result.values())
    return result, audit


class StreamingMerger(BilateralMerger):
    def __init__(self, leaf, base, previous, incoming, history, count, active, condition, projections):
        super().__init__(leaf, base, previous, incoming, {}, count, active, 'old_rows', projections)
        self.condition = condition
        self.bilateral_count = self.coefficients.numel()
        assert self.bilateral_count == 347666 and len(history) == count + 1
        with torch.no_grad():
            initial = BilateralMerger.compose(self)
            reference = {n: initial[n].detach().cpu() for n in (FILM_W, FILM_B, *CROSS)}
            del initial
            correction, self.factor_audit = factor_correction(reference, history, condition)
            self.factor_directions = {n: v.to(self.coefficients.device) for n, v in correction.items()}
        self.coefficients = torch.nn.Parameter(torch.cat([self.coefficients.detach(),
            torch.zeros(WIDTH, device=self.coefficients.device)]))

    def compose(self, coefficients=None):
        c = self.coefficients if coefficients is None else coefficients
        result = super().compose(c[:self.bilateral_count])
        gates = c[self.bilateral_count:]
        row_gates = torch.cat([gates, torch.zeros_like(gates)])
        result[FILM_W] = result[FILM_W] + row_gates[:, None] * self.factor_directions[FILM_W]
        result[FILM_B] = result[FILM_B] + row_gates * self.factor_directions[FILM_B]
        for name in CROSS:
            result[name] = result[name] + gates[None, :] * self.factor_directions[name]
        return result
