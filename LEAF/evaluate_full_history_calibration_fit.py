"""Compare final native checkpoints on the complete existing calibration sets."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_experts_ri import configure
from d_downstream_qformer_ce import PrototypeClassifier, classification_metrics, load_leaf, sha256
from run_streaming_strict_history import TaskData, strict_settings, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    spec = json.loads(args.plan.read_text())
    output = args.plan.parent / 'comparison.json'
    assert not output.exists()
    for path, digest in spec['sha256'].items():
        assert sha256(Path(path)) == digest, path
    main_report = json.loads(Path(spec['scientific_report']).read_text())
    assert all(main_report['checks'].values())
    strict_settings()
    device = configure(args.gpu, 0)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    started = time.monotonic()
    results, sources, examples = {}, {}, 0
    for seed in spec['seeds']:
        plan_path = Path(spec['scientific_root']) / f'seed{seed}' / 'plan.json'
        plan = json.loads(plan_path.read_text())
        for path, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
            assert sha256(Path(path)) == digest, path
        source = json.loads(Path(plan['source_fit']).read_text())
        data = TaskData(plan, source, device)
        for task in plan['tasks']:
            data.arrive(task)
        leaf = load_leaf(data.config, Path(source['base_checkpoint'])).to(device).eval().requires_grad_(False)
        results[str(seed)] = {}
        for order in spec['orders']:
            arms = {}
            folders = {'full_history': str(plan_path.parent / f'full_history_{order}'),
                       'old_rows': plan['reference_runs'][order]['old_rows']}
            for condition, folder in folders.items():
                folder = Path(folder)
                run_path = folder / 'run.json'
                assert sha256(run_path) == main_report['sources'][str(run_path)]
                run = json.loads(run_path.read_text())
                assert run['status'] == 'completed' and len(run['stages']) == 16
                checkpoint = folder / 'stage_16' / 'merged.ckpt'
                assert sha256(checkpoint) == run['stages'][-1]['checkpoint_sha256']
                sources[str(checkpoint)] = sha256(checkpoint)
                leaf.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
                metrics = {}
                for task in plan['tasks']:
                    x, y = data.train[task]
                    assert len(y) == 512
                    for key, value in data.records[task].items():
                        assert run['dataset_audits'][task][key] == value, (seed, order, condition, task, key)
                    wrapper = PrototypeClassifier(leaf, *data.readouts[task]).eval()
                    logits = torch.cat([wrapper(x[i:i + spec['batch_size']].to(device)).cpu()
                                        for i in range(0, len(y), spec['batch_size'])])
                    weighted = F.cross_entropy(logits, y, reduction='none') * data.weights[task].cpu()[y]
                    metrics[task] = classification_metrics(logits, y)
                    metrics[task]['balanced_CE'] = float(weighted.double().mean())
                    examples += len(y)
                    del wrapper, logits, weighted
                arms[condition] = {'tasks': metrics,
                    'macro_balanced_CE': float(np.mean([v['balanced_CE'] for v in metrics.values()])),
                    'macro_BAcc': float(np.mean([v['BAcc'] for v in metrics.values()]))}
                print(json.dumps({'seed': seed, 'order': order, 'condition': condition,
                                  'macro_balanced_CE': arms[condition]['macro_balanced_CE'],
                                  'macro_BAcc': arms[condition]['macro_BAcc']}), flush=True)
            results[str(seed)][order] = {'arms': arms,
                'full_history_minus_bilateral_balanced_CE': arms['full_history']['macro_balanced_CE'] - arms['old_rows']['macro_balanced_CE'],
                'full_history_minus_bilateral_BAcc_pp': 100 * (arms['full_history']['macro_BAcc'] - arms['old_rows']['macro_BAcc']),
                'development_validation_gain_pp': main_report['banks'][str(seed)]['orders'][order]['contrasts']['full_history_minus_old_rows']['gain_pp']}
        assert not data.valid
        del leaf, data
        torch.cuda.empty_cache()
    assert examples == spec['forward_example_passes'] == 65536
    write_json(output, {'plan_sha256': sha256(args.plan), 'sources': sources, 'banks': results,
        'scientific_report_sha256': sha256(Path(spec['scientific_report'])),
        'optimizer_updates': 0, 'backward_passes': 0, 'forward_example_passes': examples,
        'new_development_validation_read': False, 'official_test_read': False,
        'strict_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'elapsed_seconds': time.monotonic() - started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'scope': 'Post-hoc fit diagnostic on the original512 calibration rows per task, at the final fixed-average checkpoint. No model changes. Lower calibration loss with poorer development accuracy is suggestive of a fit/generalization mismatch, not causal proof of overfitting. Higher calibration loss does not prove the optimizer cannot reach a better solution.'})


if __name__ == '__main__':
    main()
