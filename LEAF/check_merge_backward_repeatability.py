"""Repeat one unchanged calibration graph with default or strict determinism."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from run_streaming_history_sketch_merge import TaskData, write_json
from streaming_history_sketch_merge import RANK
from streaming_predictive_sketch import StreamingMerger, observe_predictive, update_sketch


def difference(a, b):
    d = a.double() - b.double()
    return {'bitwise_equal': torch.equal(a, b), 'max_abs': float(d.abs().max()),
            'relative_L2': float(d.norm() / b.double().norm().clamp_min(1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime', choices=['default', 'strict'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=False)
    started = time.monotonic()
    plan_path = Path('runs/streaming_predictive_sketch_v1/plan.json')
    plan = json.loads(plan_path.read_text())
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(name)) == digest
    if args.regime == 'strict':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    task, incoming_task = plan['tasks'][:2]
    data.arrive(task)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    previous = torch.load(source['experts'][task], map_location='cpu', weights_only=True)
    incoming = torch.load(source['experts'][incoming_task], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    history = {name: torch.zeros((RANK, *base[name].shape)) for name in plan['active']}
    update_sketch(history, previous, base, plan['active'], 'energy_history')
    merger = StreamingMerger(leaf, base, previous, incoming, history, 1,
                             plan['active'], 'energy_history', {})
    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
    x, y = data.train[task]
    x, y = x[:16].to(device), y[:16].to(device)
    initial = merger.coefficients.detach().cpu().clone()
    snapshots = []
    record = {'regime': args.regime, 'status': 'running', 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'plan_sha256': sha256(plan_path),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'batch_eeg_hash': array_hash(x.cpu().numpy()),
              'task': task, 'batch_size': 16, 'repetitions': 4,
              'optimizer_updates': 0, 'completed_CE_backwards': 0, 'completed_score_VJPs': 0,
              'development_validation_read': False, 'official_test_read': False,
              'scope': 'Same initial model and same 16 training rows reused four times in one process. Tests fixed-graph repeatability, not full-stream or cross-device reproducibility.'}
    write_json(args.outdir / 'run.json', record)
    try:
        for repetition in range(4):
            merger.coefficients.grad = None
            parameters = merger.compose()
            logits = merger.logits(wrapper, x, parameters)
            generator = torch.Generator(device=device).manual_seed(60333)
            signs = torch.randint(0, 2, logits.shape, generator=generator,
                                  device=device, dtype=torch.int8) * 2 - 1
            observe_predictive(merger, parameters, logits, signs, data.weights[task][y])
            record['completed_score_VJPs'] += 1
            response = torch.cat([merger.responses.last_response[name].flatten()
                                  for name in plan['active']]).cpu()
            loss = (F.cross_entropy(logits, y, reduction='none') * data.weights[task][y]).mean()
            loss.backward()
            record['completed_CE_backwards'] += 1
            snapshots.append({'logits': logits.detach().cpu(),
                              'CE_coefficient_gradient': merger.coefficients.grad.detach().cpu(),
                              'predictive_response': response})
            assert torch.equal(initial, merger.coefficients.detach().cpu())
            write_json(args.outdir / 'run.json', record)
            del parameters, logits, signs, loss
        torch.save(snapshots, args.outdir / 'snapshots.pt')
        record.update(status='completed', comparisons=[
            {key: difference(current[key], snapshots[0][key]) for key in snapshots[0]}
            for current in snapshots[1:]],
            CE_example_passes=record['completed_CE_backwards'] * 16,
            score_VJP_example_passes=record['completed_score_VJPs'] * 16,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            elapsed_seconds=time.monotonic() - started)
        assert not data.valid
        write_json(args.outdir / 'run.json', record)
        print(json.dumps(record, indent=2), flush=True)
    except Exception as error:
        record.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        write_json(args.outdir / 'run.json', record)
        raise


if __name__ == '__main__':
    main()
