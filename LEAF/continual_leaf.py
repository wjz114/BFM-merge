"""LEAF encoder for the new student. All learned readout parameters live in W."""
import torch
from torch import nn

from LEAF import Tower
from continual_readout import pool_branches, scores
from load_config import build_model_config


class LeafFeatures(Tower):
    def __init__(self, config, bins=1):
        super().__init__(config)
        self.bins = bins
        # Only retain parameters used by Tower.forward().
        del self.mask_token, self.mask_outProj, self.next_outProj

    def forward(self, x):
        return pool_branches(super().forward(x), self.bins)


def expert_features(config, checkpoint, bins=1):
    encoder = LeafFeatures(config, bins)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    encoder.load_state_dict({k: state['tower.' + k] for k in encoder.state_dict()})
    return encoder.requires_grad_(False).eval()


class LeafStudent(nn.Module):
    def __init__(self, config, encoder, W, prototypes):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.register_buffer('W', W)
        self.prototypes = nn.ParameterDict({task: nn.Parameter(P, requires_grad=False)
                                            for task, P in prototypes.items()})

    def forward(self, x, task):
        return scores(self.encoder(x), self.W, self.prototypes[task])


def save_student(path, encoder, W, vocab, model_config):
    torch.save({'encoder': encoder.state_dict(), 'W': W.detach(), 'vocab': vocab,
                'model_config': model_config, 'bins': encoder.bins}, path)


def load_student(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    cfg = build_model_config({'model': data['model_config']})
    encoder = LeafFeatures(cfg, data.get('bins', 1))
    encoder.load_state_dict(data['encoder'])
    return LeafStudent(cfg, encoder, data['W'],
                       {t: v['prototypes'] for t, v in data['vocab'].items()}).eval()


class RecursiveMerge(nn.Module):
    """Two signed scalars per active named tensor; buffers retain their base values."""
    def __init__(self, base, old, expert, stage):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        old_weights = dict(old.named_parameters())
        expert_weights = dict(expert.named_parameters())
        self.old_delta = {k: old_weights[k].detach() - v for k,v in base.named_parameters()}
        self.new_delta = {k: expert_weights[k].detach() - v for k,v in base.named_parameters()}
        first = next(base.parameters()).new_tensor([(stage - 1)/stage, 1/stage])
        self.theta = nn.Parameter(first.repeat(len(self.old_delta), 1))
        for k, b in base.named_buffers():
            assert torch.equal(b, old.get_buffer(k)) and torch.equal(b, expert.get_buffer(k)), k

    def weights(self, theta):
        from continual_readout import merge_tensors
        return merge_tensors(dict(self.base.named_parameters()), self.old_delta, self.new_delta, theta)

    def forward(self, x):
        return torch.func.functional_call(self.base, self.weights(self.theta), (x,))

    @torch.no_grad()
    def fold(self, mean_theta):
        from copy import deepcopy
        folded = deepcopy(self.base)
        folded.load_state_dict({**dict(self.base.named_buffers()), **self.weights(mean_theta)})
        return folded
