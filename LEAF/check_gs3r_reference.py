"""Check the GS3R LEAF adapter on three training-only expert arrivals."""
import json
import time
from pathlib import Path

import torch

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, load_leaf, sha256
from gs3r_reference import GS3RReference
from run_streaming_subspace_merge import TaskData, write_json


def main():
    root = Path('runs/gs3r_reference_v1')
    assert not (root / 'engineering.json').exists()
    plan = json.loads(Path('runs/streaming_history_sketch_v1/plan.json').read_text())
    source = json.loads(Path(plan['source_fit']).read_text())
    device = configure(0, 0)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    data = TaskData(plan, source, device)
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    merger = GS3RReference(base, plan['active'], device)
    tasks = plan['tasks'][:3]
    records, forwards = [], 0
    started = time.monotonic()
    for stage, task in enumerate(tasks, 1):
        data.arrive(task)
        path = Path(source['experts'][task])
        assert sha256(path) == plan['expert_sha256'][task]
        incoming = torch.load(path, map_location='cpu', weights_only=True)
        record = merger.arrive(incoming)
        state = incoming if stage == 1 else merger.export()
        assert all(torch.isfinite(state[n]).all() for n in plan['active'])
        assert all(torch.equal(state[n], base[n]) for n in base if n not in plan['active'])
        leaf.load_state_dict(state, strict=True)
        outputs = {}
        with torch.no_grad():
            for seen in tasks[:stage]:
                wrapper = PrototypeClassifier(leaf, *data.readouts[seen]).eval()
                logits = wrapper(data.train[seen][0][:2].to(device)).cpu()
                assert torch.isfinite(logits).all()
                outputs[seen] = logits
                forwards += 2
        record.update(stage=stage, tensor_count=len(state), finite_native_logits=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    checkpoint = root / 'engineering_stage03.ckpt'
    torch.save(state, checkpoint)
    reloaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert all(torch.equal(v, reloaded[n]) for n, v in state.items())
    leaf.load_state_dict(reloaded, strict=True)
    errors = {}
    with torch.no_grad():
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            logits = wrapper(data.train[task][0][:2].to(device)).cpu()
            errors[task] = float((logits - outputs[task]).abs().max())
            assert errors[task] <= 1e-5
            assert torch.equal(logits.argmax(1), outputs[task].argmax(1))
            forwards += 2
    write_json(root / 'engineering.json', {
        'passed': True, 'records': records, 'native_updates': 0,
        'real_expert_arrivals': 3, 'real_matrix_merge_calls': 250,
        'training_only_forward_examples': forwards, 'reload_max_abs_errors': errors,
        'serialized_state_exact': True, 'checkpoint_sha256': sha256(checkpoint),
        'elapsed_seconds': time.monotonic() - started,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'source_sha256': {p: sha256(Path(p)) for p in
                          ['gs3r_reference.py', 'run_gs3r_reference.py', 'check_gs3r_reference.py']},
        'scope': 'Native engineering only; no validation or official-test arrays read, '
                 'no optimizer updates and no performance selection.'})


if __name__ == '__main__':
    main()
