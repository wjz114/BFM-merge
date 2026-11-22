"""Residual tensorwise expert-span updates around an exact frozen LEAF state."""
import json
from pathlib import Path

import torch
from torch.func import functional_call

from d_downstream_qformer_ce import sha256
from merge_poc import UNUSED_PREFIXES


class ResidualTensorMerger:
    def __init__(self, leaf, source):
        self.leaf = leaf.eval().requires_grad_(False)
        self.tasks = source['datasets']
        self.initial = {name: value.detach().clone() for name, value in leaf.state_dict().items()}
        self.parameter_names = [name for name, _ in leaf.named_parameters()]
        self.active = [name for name in self.parameter_names if not name.startswith(UNUSED_PREFIXES)]
        assert self.active == source['groups']
        assert len(self.active) == source['merged_tensors']
        self.fixed = set(self.initial)-set(self.active)
        device = next(leaf.parameters()).device
        base_path = Path(source['base_checkpoint'])
        assert sha256(base_path) == source['base_checkpoint_sha256']
        base = torch.load(base_path, map_location='cpu', weights_only=True)
        assert set(base) == set(self.initial)
        assert all(torch.equal(self.initial[name].cpu(), base[name]) for name in self.fixed)
        pieces = {name: [] for name in self.active}
        self.sources = {}
        for task in self.tasks:
            path = Path(source['experts'][task])
            meta = json.loads(path.with_name('run.json').read_text())
            assert meta['datasets'] == [task]
            assert meta['source_checkpoint_sha256'] == source['base_checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert set(state) == set(base)
            assert all(state[name].shape == base[name].shape for name in state)
            assert all(torch.equal(state[name], base[name]) for name in self.fixed)
            for name in self.active:
                pieces[name].append(state[name].float()-base[name].float())
            self.sources[task] = {'path':str(path),'sha256':sha256(path)}
        self.deltas = {name:torch.stack(values).to(device) for name,values in pieces.items()}
        assert all(torch.isfinite(value).all() for value in self.deltas.values())
        self.beta = torch.nn.Parameter(torch.zeros(len(self.tasks),len(self.active),device=device))
        assert self.beta.numel() == source['num_coefficients']
        self.component_columns = {prefix:[i for i,name in enumerate(self.active) if name.startswith(prefix+'.')]
                                  for prefix in ['tower','qformer']}

    def compose(self, beta=None):
        beta = self.beta if beta is None else beta
        params = {name:self.initial[name] for name in self.parameter_names}
        for column,name in enumerate(self.active):
            params[name] = self.initial[name]+torch.tensordot(beta[:,column],self.deltas[name],dims=1)
        return params

    def logits(self, wrapper, x, params=None):
        params = self.compose() if params is None else params
        return functional_call(wrapper, {'leaf.'+name:value for name,value in params.items()}, (x,))

    @torch.no_grad()
    def materialize(self, beta=None):
        params = self.compose(beta)
        state = {name:params.get(name,value).detach().cpu().clone() for name,value in self.initial.items()}
        assert all(torch.equal(state[name],self.initial[name].cpu()) for name in self.fixed)
        assert all(torch.isfinite(state[name]).all() for name in self.active)
        return state
