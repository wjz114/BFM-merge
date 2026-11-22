"""Same-input predictions from every arrived expert, with fixed target readouts.

The four-expert bank is a diagnostic teacher, not an inference proposal.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import PrototypeClassifier, classification_metrics, load_leaf, sha256
from run_streaming_strict_history import TaskData, strict_settings, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    root = args.plan.parent
    assert not (root / 'run.json').exists()
    strict_settings()
    device = configure(args.gpu, plan['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'], device)
    torch.set_grad_enabled(False)
    source = json.loads(Path(plan['source_fit']).read_text())
    data = TaskData(plan, source, device)
    tasks = plan['pilot_tasks']
    # This probe concerns the final four-task arrival, not an earlier prefix.
    for task in tasks:
        data.arrive(task)
        data.validation(task)
    leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
    outputs = {task: {split: [] for split in ('calibration', 'development')} for task in tasks}
    start, examples = time.monotonic(), 0
    for expert in tasks:
        path = Path(source['experts'][expert])
        assert sha256(path) == plan['expert_sha256'][str(path)]
        leaf.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        for task in tasks:
            wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
            for split, (x, y) in [('calibration', data.train[task]), ('development', data.valid[task])]:
                bs = plan['validation_batch_size']
                logits = torch.cat([wrapper(x[k:k + bs].to(device)).cpu() for k in range(0, len(y), bs)])
                assert logits.shape == (len(y), len(source['labels'][task])) and torch.isfinite(logits).all()
                outputs[task][split].append(logits)
                examples += len(y)
        print(json.dumps(dict(expert=expert, forward_examples=examples,
                             elapsed_seconds=time.monotonic() - start)), flush=True)
    results, payload = {}, {}
    for task in tasks:
        results[task], payload[task] = {}, {}
        own = tasks.index(task)
        for split, (_, y) in [('calibration', data.train[task]), ('development', data.valid[task])]:
            logits = torch.stack(outputs[task][split])
            probability_mean = logits.double().softmax(-1).mean(0).log().float()
            logit_mean = logits.double().mean(0).float()
            correct = logits.argmax(-1).eq(y[None])
            any_correct = correct.any(0)
            # Label-dependent coverage is explicitly an oracle diagnostic.
            oracle_bacc = float(torch.stack([any_correct[y == c].float().mean()
                for c in y.unique()]).mean())
            results[task][split] = dict(
                individual={expert: classification_metrics(logits[j], y) for j, expert in enumerate(tasks)},
                own_expert=classification_metrics(logits[own], y),
                uniform_probability=classification_metrics(probability_mean, y),
                uniform_logits=classification_metrics(logit_mean, y),
                oracle_any_correct_bacc=oracle_bacc,
                own_wrong_any_other_correct=int((~correct[own] & any_correct).sum()),
                examples=len(y), labels_hash=array_hash(y.numpy()), logits_hash=array_hash(logits.numpy()))
            payload[task][split] = dict(logits=logits, labels=y)
    torch.save(payload, root / 'predictions.pt')
    macro = {split: {name: float(np.mean([results[t][split][name]['BAcc'] for t in tasks]))
        for name in ['own_expert', 'uniform_probability', 'uniform_logits']}
        for split in ['calibration', 'development']}
    run = dict(status='completed', plan_sha256=sha256(args.plan), results=results,
        macro_bacc=macro, dataset_audits=data.records, forward_examples=examples,
        optimizer_updates=0, source_training_steps=0, official_test_read=False,
        predictions_sha256=sha256(root / 'predictions.pt'),
        elapsed_seconds=time.monotonic() - start, scope=plan['scope'])
    assert examples == 4 * (4 * 512 + 5534)
    write_json(root / 'run.json', run)
    print(json.dumps(dict(status='completed', macro_bacc=macro, forward_examples=examples)), flush=True)


if __name__ == '__main__':
    main()
