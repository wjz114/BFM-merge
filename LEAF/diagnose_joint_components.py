"""Oracle component swaps between the current merger and Joint; no fitting."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_sensor_projection import CHANGED, calibration_and_validation
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    for prefix in ['merged', 'joint']:
        assert sha256(Path(plan[prefix+'_checkpoint'])) == plan[prefix+'_sha256']
    args.outdir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    device = configure(args.gpu, 0)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    source_dir = Path(plan['source_fit'])
    source = json.loads((source_dir/'run.json').read_text())
    indices = np.load(source_dir/'calibration_indices.npz')
    merged = torch.load(plan['merged_checkpoint'], map_location='cpu', weights_only=True)
    joint = torch.load(plan['joint_checkpoint'], map_location='cpu', weights_only=True)
    active = set(source['groups'])
    assert all(torch.equal(merged[k], joint[k]) for k in merged if k not in active)
    tower = {k for k in active if k.startswith('tower.')}
    qformer = {k for k in active if k.startswith('qformer.')}
    tokenizer = {k for k in active if k.startswith('tower.tokenizer.')}
    spatial = set(CHANGED)
    assert tower | qformer == active and not tower & qformer
    selections = {'merged': set(), 'joint': active, 'joint_tower': tower,
                  'joint_qformer': qformer, 'joint_tokenizer': tokenizer,
                  'joint_except_tokenizer': active-tokenizer, 'joint_spatial': spatial,
                  'joint_except_spatial': active-spatial}
    assert list(selections) == plan['conditions']
    states = {name: {k: joint[k] if k in keys else v for k,v in merged.items()} for name,keys in selections.items()}
    leaf = load_leaf(config, Path(plan['merged_checkpoint'])).to(device).eval().requires_grad_(False)
    average = json.loads(Path(plan['merged_checkpoint']).with_name('run.json').read_text())
    joint_reference = json.loads(Path(plan['joint_reference']).read_text())['evaluations']['joint']['per_task']
    data_reference = json.loads(Path(plan['data_reference']).read_text())['dataset_audits']
    evaluations = {name: {} for name in states}
    for task in source['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        assert array_hash(cx) == data_reference[task]['eeg_hash']
        assert len(vy) == data_reference[task]['validation_size']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy
        loader = make_loader(x, y, config, len(labels), 128, False)
        truth = torch.from_numpy(y)
        logits = {}
        for name, state in states.items():
            leaf.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx,_ in loader])
            assert torch.isfinite(values).all()
            logits[name] = values
            evaluations[name][task] = {split: classification_metrics(values[ids], truth[ids])
                                       for split,ids in [('calibration', slice(0,n)), ('validation', slice(n,None))]}
        assert abs(evaluations['merged'][task]['validation']['BAcc']-average['folded_val'][task]['BAcc']) < 1e-12
        assert abs(evaluations['joint'][task]['validation']['BAcc']-joint_reference[task]['BAcc']) < 1e-12
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {c: round(100*(evaluations[c][task]['validation']['BAcc']-evaluations['merged'][task]['validation']['BAcc']),4)
                     for c in states if c!='merged'}, flush=True)
    summaries, factorials = {}, {}
    for split in ['calibration', 'validation']:
        summaries[split] = {c: float(np.mean([v[split]['BAcc'] for v in rows.values()])) for c,rows in evaluations.items()}
        factorials[split] = {}
        for partition, a, b in [('tower_qformer','joint_tower','joint_qformer'),
                                ('tokenizer_rest','joint_tokenizer','joint_except_tokenizer'),
                                ('spatial_rest','joint_spatial','joint_except_spatial')]:
            per_task = {t: 100*(evaluations['joint'][t][split]['BAcc']+evaluations['merged'][t][split]['BAcc']
                               -evaluations[a][t][split]['BAcc']-evaluations[b][t][split]['BAcc']) for t in source['datasets']}
            factorials[split][partition] = {'interaction_pp': float(np.mean(list(per_task.values()))),
                                            'per_task_interaction_pp': per_task,
                                            'a_vs_merged_pp': 100*(summaries[split][a]-summaries[split]['merged']),
                                            'b_vs_merged_pp': 100*(summaries[split][b]-summaries[split]['merged'])}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'evaluations': evaluations, 'summaries': summaries, 'factorials': factorials,
              'joint_tensor_selections': {c: sorted(keys) for c,keys in selections.items()},
              'joint_parameter_counts': {c: sum(joint[k].numel() for k in keys) for c,keys in selections.items()},
              'seconds': time.time()-start, 'oracle_only': True, 'test_data_read': False,
              'test_inference_performed': False, 'interpretation': plan['interpretation']}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', summaries['validation'], flush=True)


if __name__ == '__main__':
    main()
