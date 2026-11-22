"""Evaluate pinned, optimization-free GS3R on arriving original LEAF experts."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import ROOT, load_leaf, prepare_text, sha256
from gs3r_reference import GS3RReference
from load_config import build_model_config, load_yaml
from run_streaming_subspace_merge import TaskData, evaluate, write_json


class ValidationOnlyData(TaskData):
    """Reuse validation parsing without opening training or calibration arrays."""

    def __init__(self, plan, source, device):
        self.source, self.device = source, device
        self.audit = json.loads(Path(plan['coverage_audit']).read_text())['audits']
        self.config = build_model_config(load_yaml(ROOT / 'configs/LEAF_mpnet.yaml'))
        self.valid, self.readouts, self.records = {}, {}, {}

    def arrive(self, task):
        labels, text, instruction, prototypes = prepare_text(self.config, task)
        assert labels == self.source['labels'][task]
        assert text == self.source['instruction'][task]
        self.readouts[task] = instruction.to(self.device), prototypes.to(self.device)
        self.records[task] = {'training_arrays_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--order', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    reference_plan = json.loads(args.plan.read_text())
    for path, digest in reference_plan['sha256'].items():
        assert sha256(Path(path)) == digest, path
    plan = json.loads(Path(reference_plan['arrival_plan']).read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    source = json.loads(Path(plan['source_fit']).read_text())
    tasks = plan['tasks'] if args.order == 'forward' else list(reversed(plan['tasks']))
    data = ValidationOnlyData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    merger = GS3RReference(base, plan['active'], device)
    records = []
    run = {'status': 'running', 'plan': str(args.plan), 'plan_sha256': sha256(args.plan),
           'method': 'official_GS3R_full_rank_no_auxiliary_optimization', 'order': tasks,
           'stages': records, 'gpu': args.gpu, 'updates': 0, 'training_examples': 0,
           'training_arrays_read': False, 'official_test_read': False}
    for index, task in enumerate(tasks):
        stage = index + 1
        data.arrive(task)
        expert_path = Path(source['experts'][task])
        assert sha256(expert_path) == plan['expert_sha256'][task]
        incoming = torch.load(expert_path, map_location='cpu', weights_only=True)
        merge_started = time.monotonic()
        record = merger.arrive(incoming)
        record['merging_seconds'] = time.monotonic() - merge_started
        # As in the learned-merger protocol, evaluate the first original expert
        # directly; preserve the official task-vector representation for merging.
        state = incoming if stage == 1 else merger.export()
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        assert all(torch.equal(state[n], base[n]) for n in base if n not in plan['active'])
        leaf.load_state_dict(state, strict=True)
        stage_dir = args.outdir / f'stage_{stage:02d}'
        stage_dir.mkdir()
        checkpoint = stage_dir / 'merged.ckpt'
        torch.save(state, checkpoint)
        record.update(stage=stage, arriving_task=task, checkpoint_sha256=sha256(checkpoint))
        del incoming, state
        gc.collect()
        torch.cuda.empty_cache()
        if stage in plan['validation_stages']:
            record['validation'] = evaluate(leaf, data, tasks[:stage], plan['validation_batch_size'], device)
            record['macro_BAcc'] = float(np.mean([v['BAcc'] for v in record['validation'].values()]))
        records.append(record)
        run.update(dataset_audits=data.records, elapsed_seconds=time.monotonic() - started,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.outdir / 'run.json', run)
        print(json.dumps({'stage': stage, 'macro_BAcc': record.get('macro_BAcc'),
                          'seconds': run['elapsed_seconds']}), flush=True)
    torch.save({'vector': merger.vector, 'restoration': merger.restoration, 'count': merger.count},
               args.outdir / 'resume_state.pt')
    run.update(status='completed',
               resume_vector_bytes=sum(x.numel() * x.element_size() for x in merger.vector.values()),
               restoration_bytes=records[-1]['restoration_bytes'],
               scope='Existing data-free GS3R reference, adapted only for device and LEAF I/O. '
                     'Full-rank matrix merging; official recursive half-averaging for nonmatrices. '
                     'No coefficient fitting, validation selection, or future expert access. '
                     'Different training-data and compute budget from supervised replay candidates. '
                     'Exact resume vector storage is reported separately from restoration metadata.')
    write_json(args.outdir / 'run.json', run)


if __name__ == '__main__':
    main()
