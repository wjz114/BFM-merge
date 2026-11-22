"""Check whether calibration-positive base transfer survives expert composition."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha256(Path(__file__)) == plan['code_sha256']
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    source = json.loads((Path(plan['source'])/'run.json').read_text())
    indices = np.load(Path(plan['source'])/'calibration_indices.npz')
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    active = {name for name,_ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    base = {name:value.detach().clone() for name,value in leaf.state_dict().items()}
    needed = list(dict.fromkeys(t for pair in plan['pairs'] for t in [pair['expert'],pair['target']]))
    states, hashes = {'original': {}, 'ri': {}}, {}
    for task in needed:
        for bank,path in [('original',Path(source['experts'][task])),('ri',Path(plan['ri_bank'])/task/'best.ckpt')]:
            state = torch.load(path,map_location=device,weights_only=True)
            assert all(torch.equal(state[name],base[name]) for name in base if name not in active)
            states[bank][task] = state
            hashes[f'{bank}/{task}'] = sha256(path)
    results = []
    for target in dict.fromkeys(pair['target'] for pair in plan['pairs']):
        labels, _, instruction, prototypes = prepare_text(config,target)
        wrapper = PrototypeClassifier(leaf,instruction,prototypes).to(device).eval()
        _, tx,ty,vx,vy,sx,sy = load_dataset(target)
        x = np.concatenate([tx[indices[target]],vx]);y = np.concatenate([ty[indices[target]],vy])
        n = len(indices[target]);del tx,ty,vx,vy,sx,sy
        loader = make_loader(x,y,config,len(labels),128,False)
        truth = torch.from_numpy(y.astype('int64'))
        contexts = {'base':base,'target_expert':states['original'][target]}
        context_logits, predictions = {}, {}
        for name,state in contexts.items():
            leaf.load_state_dict(state,strict=True)
            context_logits[name] = torch.cat([wrapper(xx.to(device)).cpu() for xx,_ in loader])
        old = np.load(Path(plan['original_transfer'])/f'{target}_logits.npz')
        assert np.array_equal(old['truth'],y) and int(old['n_calibration'])==n
        for context, original_key in [('base','base'),('target_expert',target)]:
            assert float((context_logits[context]-torch.from_numpy(old[original_key])).abs().max()) <= 1e-5
        for pair in [p for p in plan['pairs'] if p['target']==target]:
            expert = pair['expert']
            for bank in ['original','ri']:
                for context, anchor in contexts.items():
                    for scale in plan['scales']:
                        composed = {name:(anchor[name]+scale*(states[bank][expert][name]-base[name])
                                          if name in active else base[name]) for name in base}
                        leaf.load_state_dict(composed,strict=True)
                        logits = torch.cat([wrapper(xx.to(device)).cpu() for xx,_ in loader])
                        key=f'{expert}__{bank}__{context}__{scale}'
                        predictions[key] = logits
                        row={'expert':expert,'target':target,'pair_selection':pair['selection'],
                             'bank':bank,'context':context,'scale':scale,'splits':{}}
                        for split,ids in [('calibration',slice(0,n)),('validation',slice(n,None))]:
                            metrics=classification_metrics(logits[ids],truth[ids])
                            baseline=classification_metrics(context_logits[context][ids],truth[ids])
                            row['splits'][split]={'composed':metrics,'context_baseline':baseline,
                                                  'bacc_change_pp':100*(metrics['BAcc']-baseline['BAcc'])}
                        results.append(row)
                        print(expert,target,bank,context,scale,'val change pp',row['splits']['validation']['bacc_change_pp'],flush=True)
        torch.save({'truth':truth,'n_calibration':n,'context_logits':context_logits,'compositions':predictions},args.outdir/f'{target}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    result={'records':results,'checkpoint_hashes':hashes,'seconds':time.time()-started,
            'plan':plan,'plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
            'scope':'Fixed calibration plus complete original validation. Source updates are added to either base or frozen original target expert, with base buffers. No optimization, selection or official test.',
            'interpretation':'This checks context dependence in two fixed composition settings. Neither equals the learned tensor merger, so improvement or reversal is not a final merge result.'}
    (args.outdir/'run.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
