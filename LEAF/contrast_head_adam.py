"""Project only the applied final-head Adam step into current task contrasts."""
import json
from pathlib import Path

import torch


class ContrastHeadAdam:
    def __init__(self, named_parameters, bases, task_order, outdir, **adam_args):
        named = dict(named_parameters)
        self.weight = named['qformer.output_embed.2.weight']
        self.bias = named['qformer.output_embed.2.bias']
        self.optimizer = torch.optim.Adam(list(named.values()), **adam_args)
        self.bases = {task: q.to(device=self.weight.device, dtype=torch.float64) for task, q in bases.items()}
        self.task_order = task_order
        self.outdir = Path(outdir)
        self.steps, self.statistics = 0, []

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        task = self.task_order[self.steps]
        basis = self.bases[task]
        old = torch.cat([self.weight.detach(), self.bias.detach()[:, None]], dim=1).double()
        self.optimizer.step()
        native = torch.cat([self.weight.detach(), self.bias.detach()[:, None]], dim=1).double()
        proposal = native-old
        coordinates = basis@proposal
        projected = basis.T@coordinates
        replacement = (old+projected).to(self.weight.dtype)
        self.weight.copy_(replacement[:, :-1])
        self.bias.copy_(replacement[:, -1])
        actual = replacement.double()-old
        denominator = coordinates.norm().clamp_min(1e-300)
        exact_error = float((basis@projected-coordinates).norm()/denominator)
        assert exact_error < 1e-11
        assert torch.isfinite(actual).all()
        self.steps += 1
        row = {'step': self.steps, 'task': task, 'contrast_rank': len(basis),
               'head_adam_proposal_norm': float(proposal.norm()),
               'head_projected_norm': float(projected.norm()),
               'head_applied_norm': float(actual.norm()),
               'head_decision_null_update_energy_fraction': float((proposal-projected).square().sum()/proposal.square().sum().clamp_min(1e-300)),
               'exact_source_contrast_relative_error': exact_error,
               'materialized_source_contrast_relative_error': float((basis@actual-coordinates).norm()/denominator),
               'materialization_max_parameter_error': float((actual-projected).abs().max())}
        self.statistics.append(row)
        with (self.outdir/'contrast_dynamics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')


def toy_check(directory):
    """Momentum-bearing Adam, untouched body, preserved source-score contrasts."""
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261116)
    p = torch.randn(3, 9, generator=generator, dtype=torch.float64)
    centered = p-p.mean(0)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[s > s[0]*1e-8]
    initial = {'body': torch.randn(5, 7, generator=generator),
               'qformer.output_embed.2.weight': torch.randn(9, 11, generator=generator),
               'qformer.output_embed.2.bias': torch.randn(9, generator=generator)}
    parameters = {n: torch.nn.Parameter(v.clone()) for n, v in initial.items()}
    body_reference = torch.nn.Parameter(initial['body'].clone())
    reference_optimizer = torch.optim.Adam([body_reference], lr=1e-5)
    directory = Path(directory)
    directory.mkdir(exist_ok=False, parents=True)
    optimizer = ContrastHeadAdam(parameters.items(), {'toy': basis}, ['toy']*4, directory, lr=1e-5)
    for _ in range(4):
        for n, value in parameters.items():
            value.grad = torch.randn(value.shape, generator=generator)
        body_reference.grad = parameters['body'].grad.clone()
        optimizer.step()
        reference_optimizer.step()
        assert torch.equal(parameters['body'], body_reference)
        for key, value in reference_optimizer.state[body_reference].items():
            assert torch.equal(value, optimizer.optimizer.state[parameters['body']][key])
    return {'four_step_body_parameters_and_adam_states_exact': True,
            'four_step_momentum_used': True,
            'maximum_exact_source_contrast_relative_error': max(r['exact_source_contrast_relative_error'] for r in optimizer.statistics),
            'maximum_fp32_materialized_source_contrast_relative_error': max(r['materialized_source_contrast_relative_error'] for r in optimizer.statistics)}
