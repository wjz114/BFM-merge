"""Full source-validation retention of each original and RI-adapted expert."""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import (
    ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader,
    prepare_text, sha256,
)
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--experts-dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    source = json.loads((args.source/'run.json').read_text())
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    records = {}
    for task in source['datasets']:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        _, tx, ty, vx, vy, sx, sy = load_dataset(task)
        del tx, ty, sx, sy
        loader = make_loader(vx, vy, config, len(labels), 128, False)
        truth = torch.from_numpy(vy.astype('int64'))
        logits, metrics, hashes = {}, {}, {}
        for name, checkpoint in [('original', Path(source['experts'][task])),
                                 ('adapted', args.experts_dir/task/'best.ckpt')]:
            leaf.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            logits[name] = torch.cat([wrapper(x.to(device)).cpu() for x, _ in loader])
            metrics[name] = classification_metrics(logits[name], truth)
            hashes[name] = sha256(checkpoint)
        original_logp, adapted_logp = logits['original'].log_softmax(-1), logits['adapted'].log_softmax(-1)
        row = {'metrics': metrics, 'checkpoint_hashes': hashes, 'validation_examples': len(truth),
               'gain_pp': 100*(metrics['adapted']['BAcc']-metrics['original']['BAcc']),
               'kl_from_original': float((original_logp.exp()*(original_logp-adapted_logp)).sum(-1).mean()),
               'prediction_disagreement': float((logits['original'].argmax(-1) != logits['adapted'].argmax(-1)).float().mean())}
        torch.save({'truth': truth, **logits}, args.outdir/f'{task}_logits.pt')
        records[task] = row
        (args.outdir/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(task, 'full validation gain pp', row['gain_pp'], flush=True)
    result = {'records': records, 'macro_gain_pp': sum(v['gain_pp'] for v in records.values())/len(records),
              'seconds': time.time()-started, 'test_inference_performed': False,
              'scope': 'Full development-exposed source validation, native training instructions, FP32, TF32 off',
              'source': str(args.source), 'adapted_bank': str(args.experts_dir),
              'code_sha256': sha256(Path(__file__))}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
