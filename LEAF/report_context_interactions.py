"""Separate classification-boundary effects from nonadditive logit changes."""
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import classification_metrics, sha256


def main():
    root = Path('runs/paper_merge_v2')
    folder = root/'seed0/transfer_context_diagnostic'
    source = json.loads((folder/'run.json').read_text())
    rows = []
    for pair in source['plan']['pairs']:
        expert, target = pair['expert'], pair['target']
        data = torch.load(folder/f'{target}_logits.pt', map_location='cpu', weights_only=True)
        n = data['n_calibration']
        for bank in ['original', 'ri']:
            for scale in source['plan']['scales']:
                base = data['context_logits']['base'].double()
                anchor = data['context_logits']['target_expert'].double()
                base_plus = data['compositions'][f'{expert}__{bank}__base__{scale}'].double()
                target_plus = data['compositions'][f'{expert}__{bank}__target_expert__{scale}'].double()
                delta_base, delta_target = base_plus-base, target_plus-anchor
                additive = anchor+delta_base
                interaction = target_plus-additive
                assert torch.allclose(interaction, delta_target-delta_base, atol=1e-12, rtol=0)
                record = {'expert': expert, 'target': target, 'bank': bank, 'scale': scale, 'splits': {}}
                for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
                    truth = data['truth'][ids]
                    arrays = {'base': base[ids], 'base_plus': base_plus[ids], 'target': anchor[ids],
                              'target_plus': target_plus[ids], 'additive_logit_proxy': additive[ids]}
                    metrics = {key: classification_metrics(value, truth) for key, value in arrays.items()}
                    centered = [value[ids]-value[ids].mean(-1, keepdim=True)
                                for value in [delta_base, delta_target, interaction]]
                    u, v, residual = [value.reshape(-1) for value in centered]
                    rms = lambda value: float(value.square().mean().sqrt())
                    denom = float(u.norm()*v.norm())
                    row = {'metrics': metrics,
                           'base_delta_centered_rms': rms(u), 'target_delta_centered_rms': rms(v),
                           'interaction_centered_rms': rms(residual),
                           'interaction_relative_to_sum_rms': rms(residual)/(rms(u)+rms(v)) if rms(u)+rms(v)>0 else None,
                           'delta_cosine': float(u.dot(v))/denom if denom>0 else None,
                           'base_transfer_pp': 100*(metrics['base_plus']['BAcc']-metrics['base']['BAcc']),
                           'actual_composition_change_pp': 100*(metrics['target_plus']['BAcc']-metrics['target']['BAcc']),
                           'additive_proxy_change_pp': 100*(metrics['additive_logit_proxy']['BAcc']-metrics['target']['BAcc']),
                           'actual_minus_additive_proxy_pp': 100*(metrics['target_plus']['BAcc']-metrics['additive_logit_proxy']['BAcc'])}
                    if arrays['base'].shape[1] == 2:
                        mb = delta_base[ids, 1]-delta_base[ids, 0]
                        mt = delta_target[ids, 1]-delta_target[ids, 0]
                        row['opposite_margin_change_fraction'] = float((mb*mt<0).float().mean())
                    record['splits'][split] = row
                rows.append(record)
    result = {'records': rows, 'source_run_sha256': sha256(folder/'run.json'),
              'code_sha256': sha256(Path(__file__)),
              'identity': 'f(target+s*delta_i) = f(target) + [f(base+s*delta_i)-f(base)] + interaction',
              'interpretation': 'BAcc sign reversal alone does not prove nonlinear functional interference: a common logit displacement can affect different starting margins differently. The additive-logit proxy separates that effect from the exact finite-difference interaction, descriptively. Neither is a causal proof of full learned-merger performance.',
              'scope': 'All previously fixed four directed pairs, both banks and both scales, calibration and full exposed validation. No new inference or optimization.'}
    (root/'analysis/context_logit_interactions.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in rows:
        s = row['splits']['validation']
        print(row['expert'], row['target'], row['bank'], row['scale'],
              {k: round(s[k], 4) for k in ['base_transfer_pp', 'actual_composition_change_pp', 'additive_proxy_change_pp', 'delta_cosine', 'interaction_relative_to_sum_rms']})


if __name__ == '__main__':
    main()
