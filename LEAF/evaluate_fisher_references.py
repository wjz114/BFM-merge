"""Evaluate uncalibrated uniform and predictive-Fisher merged references."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import load_leaf, sha256
from fisher_moment_merge import add_moments, moment_centers
from run_streaming_subspace_merge import TaskData, evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    reference_plan = json.loads(args.plan.read_text())
    for path, digest in reference_plan['sha256'].items():
        assert sha256(Path(path)) == digest, path
    plan = json.loads(Path(reference_plan['arrival_plan']).read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    cache = json.loads(Path(reference_plan['cache']).read_text())
    assert cache['status'] == 'completed'
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    history = {n: torch.zeros((3, *base[n].shape), dtype=torch.float64) for n in plan['active']}
    for index, task in enumerate(plan['tasks']):
        expert = Path(source['experts'][task])
        record = cache['experts'][task]
        assert sha256(expert) == record['expert_sha256'] == plan['expert_sha256'][task]
        assert sha256(Path(record['path'])) == record['sha256']
        incoming = torch.load(expert, map_location='cpu', weights_only=True)
        fisher = torch.load(record['path'], map_location='cpu', weights_only=True)
        add_moments(history, fisher, incoming, base, plan['active'], index, plan['seed'])
        del incoming, fisher
        data.arrive(task)
    uniform, weighted = moment_centers(history, len(plan['tasks']))
    del history
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    results = {}
    for label, direction in [('uniform', uniform), ('fisher', weighted)]:
        state = dict(base)
        state.update({n: base[n] + direction[n] for n in plan['active']})
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        leaf.load_state_dict(state, strict=True)
        checkpoint = args.outdir / f'{label}.ckpt'
        torch.save(state, checkpoint)
        metrics = evaluate(leaf, data, plan['tasks'], plan['validation_batch_size'], device)
        results[label] = {'validation': metrics, 'macro_BAcc': float(np.mean([v['BAcc'] for v in metrics.values()])),
                          'checkpoint_sha256': sha256(checkpoint)}
        print(label, results[label]['macro_BAcc'], time.monotonic() - started, flush=True)
    write_json(args.outdir / 'run.json', {'status': 'completed', 'plan': str(args.plan),
               'plan_sha256': sha256(args.plan), 'results': results, 'dataset_audits': data.records,
               'elapsed_seconds': time.monotonic() - started, 'coefficient_updates': 0,
               'fisher_preparation_forward_examples': cache['forward_examples'],
               'fisher_preparation_backward_probes': cache['backward_probes'],
               'fisher_preparation_seconds': cache['elapsed_seconds'], 'official_test_read': False,
               'scope': 'Existing Fisher-merging baseline variant: global-L2-normalized model-predictive diagonal Fisher,128training-calibration rows per expert with random-VJP estimates. No coefficient or validation tuning. Pure uniform reference has no statistical estimation requirement.'})


if __name__ == '__main__':
    main()
