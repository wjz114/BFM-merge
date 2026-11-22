"""The historical tokenizer/rest parameter partition, applied to fixed-rate Adam."""
import torch


def build_optimizer(leaf, lr, tokenizer_multiplier):
    active = [(name, value) for name, value in leaf.named_parameters() if value.requires_grad]
    fast = [(name, value) for name, value in active if name.startswith('tower.tokenizer.')]
    slow = [(name, value) for name, value in active if not name.startswith('tower.tokenizer.')]
    assert len(fast) == 6 and len(slow) == 377
    assert len({id(value) for _, value in fast+slow}) == len(active) == 383
    optimizer = torch.optim.Adam([
        {'params': [v for _, v in fast], 'lr': lr*tokenizer_multiplier},
        {'params': [v for _, v in slow], 'lr': lr},
    ], weight_decay=0)
    audit = {'optimizer': 'Adam', 'weight_decay': 0,
             'tokenizer_multiplier': tokenizer_multiplier,
             'groups': {name: {'names': [n for n, _ in values],
                               'parameters': sum(v.numel() for _, v in values), 'lr': rate}
                        for name, values, rate in [('tokenizer', fast, lr*tokenizer_multiplier),
                                                   ('rest', slow, lr)]}}
    return optimizer, audit
