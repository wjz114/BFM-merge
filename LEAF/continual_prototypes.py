"""Prepare Stage-A prototypes with a local all-mpnet-base-v2 or the short-label control."""
import argparse
from pathlib import Path

import torch
from torch.nn.functional import normalize

from d_downstream_qformer_ce import prepare_text
from load_config import build_model_config, load_yaml

DESCRIPTIONS = {
    'MI_BCIC_IV2a': {'Left': 'imagined movement of the left hand',
                     'Right': 'imagined movement of the right hand',
                     'Foot': 'imagined movement of both feet',
                     'Tongue': 'imagined movement of the tongue'},
    'MI_OpenBMI': {'Right': 'imagined movement of the right hand', 'Left': 'imagined movement of the left hand'},
    'EMO_SEED_3_seg4': {'Negative': 'the participant is experiencing a negative emotional state',
                        'Neutral': 'the participant is experiencing a neutral emotional state',
                        'Positive': 'the participant is experiencing a positive emotional state'},
    'EMO_FACED': {label: 'the participant is experiencing ' + emotion
                  for label, emotion in zip(
                      ['Anger','Fear','Disgust','Sad','Amusement','Inspiration','Joy','Tenderness','Neutral'],
                      ['anger','fear','disgust','sadness','amusement','inspiration','joy','tenderness','a neutral emotional state'])}
}


@torch.inference_mode()
def encode(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=384, return_tensors='pt')
    hidden = model(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1)
    return normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mpnet', type=Path)
    mode.add_argument('--short-label-control', action='store_true')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=DESCRIPTIONS, default=['MI_BCIC_IV2a','EMO_FACED'])
    args = p.parse_args()
    if args.mpnet:
        from transformers import AutoTokenizer, MPNetModel
        model = MPNetModel.from_pretrained(args.mpnet, local_files_only=True, add_pooling_layer=False).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.mpnet, local_files_only=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = build_model_config(load_yaml())
    for task in args.tasks:
        descriptions = DESCRIPTIONS[task]
        labels, _, _, short = prepare_text(cfg, task)
        texts = [descriptions[k] for k in labels] if args.mpnet else labels
        P = encode(model, tokenizer, texts) if args.mpnet else short
        assert P.shape == short.shape and torch.isfinite(P).all()
        torch.testing.assert_close(P.norm(dim=-1), torch.ones(len(labels)), atol=1e-5, rtol=0)
        torch.save({'labels': labels, 'texts': texts, 'prototypes': P.cpu(),
                    'source': str(args.mpnet) if args.mpnet else 'mpnet-base short-label control'},
                   args.outdir / f'{task}.pt')


if __name__ == '__main__':
    main()
