"""Adapt the published TSV-M baseline to the static LEAF parameter dictionary.

Algorithm: Gargiulo et al., Task Singular Vectors, CVPR 2025.
Reference implementation: AntoAndGar/task_singular_vectors, commit
a6c97188f7aa0bb20753de4f2939be89dd64f1c3, src/utils/TSVM_utils.py.
This implements the paper's equations; it is not a proposed new method.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from d_downstream_qformer_ce import ROOT, PrototypeClassifier, evaluate, load_leaf, make_loader, prepare_text, selection_key, sha256
from load_config import build_model_config, load_yaml
from load_datasets import load_dataset
from merge_poc import UNUSED_PREFIXES


@torch.no_grad()
def tsv_matrix(vectors):
    """Polar factors with the official implementation's zero-padded remainder."""
    k, m, n = vectors.shape
    retained = min(m, n) // k
    if retained == 0:
        return torch.zeros_like(vectors[0]), {'rank_per_expert':0,'retained_energy_fraction':0.}
    left, singular, right, full_energy, kept_energy = [], [], [], 0., 0.
    for matrix in vectors:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        left.append(u[:, :retained])
        singular.append(s[:retained])
        right.append(vh[:retained])
        full_energy += s.double().square().sum().item()
        kept_energy += s[:retained].double().square().sum().item()
    left, singular, right = torch.cat(left, 1), torch.cat(singular), torch.cat(right, 0)
    remainder = min(m,n) - len(singular)
    if remainder:
        left = torch.cat([left,left.new_zeros((m,remainder))],1)
        singular = torch.cat([singular,singular.new_zeros(remainder)])
        right = torch.cat([right,right.new_zeros((remainder,n))],0)
    u, _, vh = torch.linalg.svd(left, full_matrices=False)
    left = u @ vh
    u, _, vh = torch.linalg.svd(right, full_matrices=False)
    right = u @ vh
    merged = (left * singular[None]) @ right
    return merged, {'rank_per_expert':retained,'retained_energy_fraction':kept_energy/max(full_energy,1e-30)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--affine-only', action='store_true',
                        help='Apply spectral merging only to affine weight matrices; average learned queries/embeddings')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = json.loads((args.source / 'run.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
    model = load_leaf(config, Path(source['base_checkpoint'])).to(device).eval()
    base = model.state_dict()
    active = [name for name, _ in model.named_parameters() if not name.startswith(UNUSED_PREFIXES)]
    affine = set()
    for prefix,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            affine.add(prefix+'.weight')
        elif isinstance(module,torch.nn.MultiheadAttention):
            affine.update(prefix+'.'+name for name in ['in_proj_weight','q_proj_weight','k_proj_weight','v_proj_weight']
                          if getattr(module,name,None) is not None)
    excluded_matrices = [n for n in active if base[n].ndim==2 and n not in affine]
    assert excluded_matrices==['qformer.base_queries'],excluded_matrices
    fixed = [name for name in base if name not in active]
    bank = {name:[] for name in active}
    hashes = {'base':sha256(Path(source['base_checkpoint']))}
    for task, path in source['experts'].items():
        state = torch.load(path, map_location=device, weights_only=True)
        assert set(state)==set(base) and all(torch.equal(state[n],base[n]) for n in fixed)
        for name in active:
            bank[name].append(state[name].float()-base[name].float())
        hashes[task] = sha256(Path(path))
    del state
    state, decompositions = {n:v.cpu().clone() for n,v in base.items()}, {}
    for name in active:
        vectors = torch.stack(bank.pop(name))
        if base[name].ndim == 2 and (not args.affine_only or name in affine):
            delta, information = tsv_matrix(vectors)
        else:
            delta = vectors.mean(0)
            information = {'operation':'arithmetic mean for non-spectral parameter'}
        state[name] = (base[name]+delta).cpu()
        decompositions[name] = information
        assert torch.isfinite(state[name]).all(), name
    assert all(torch.equal(state[n],base[n].cpu()) for n in fixed)
    checkpoint = args.outdir / 'merged.ckpt'
    torch.save(state,checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    assert all(torch.equal(value.cpu(),state[name]) for name,value in model.state_dict().items())
    setup_seconds = time.time()-started
    metrics = {}
    for task in source['datasets']:
        labels, text, instruction, prototypes = prepare_text(config,task)
        assert labels==source['labels'][task] and text==source['instruction'][task]
        _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(task)
        del train_x,train_y,test_x,test_y
        loader = make_loader(val_x,val_y,config,len(labels),128,False)
        metrics[task] = evaluate(PrototypeClassifier(model,instruction,prototypes).to(device),loader,device)
        (args.outdir/'validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(task,metrics[task]['BAcc'],flush=True)
    bacc,ce,_ = selection_key(metrics)
    result = {'method':'TSV-M published baseline adaptation','seed':source['seed'],'datasets':source['datasets'],
        'base_checkpoint':source['base_checkpoint'],'experts':source['experts'],'source_hashes':hashes,
        'labels':source['labels'],'instruction':source['instruction'],'best_val':metrics,
        'best_val_macro_BAcc':bacc,'best_val_macro_Loss':ce,'best_step':None,
        'alpha_updates_actual':0,'steps_requested':0,'scaling':1.,
        'selection':'Published fixed alpha=1, floor(min(m,n)/16) components/expert; no hyperparameter selection',
        'matrix_policy':('Affine weight matrices only; learned queries use arithmetic mean' if args.affine_only
                         else 'All active 2D parameters including learned queries'),
        'affine_only':args.affine_only,'non_affine_matrices':excluded_matrices,
        'other_parameter_policy':'Other active tensors use arithmetic mean; buffers/inactive parameters stay at base. No CLIP text_projection exists here.',
        'decompositions':decompositions,'fixed_base_tensors_exact':len(fixed),'reload_state_exact':True,
        'test_inference_performed':False,'checkpoint_sha256':sha256(checkpoint),
        'code_sha256':sha256(Path(__file__)),'source_run_sha256':sha256(args.source/'run.json'),
        'reference':'https://arxiv.org/html/2412.00081v3','reference_code_commit':'a6c97188f7aa0bb20753de4f2939be89dd64f1c3',
        'seconds':{'setup':setup_seconds,'total':time.time()-started}}
    (args.outdir/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TSV-M validation macro BAcc',bacc,flush=True)


if __name__ == '__main__':
    main()
