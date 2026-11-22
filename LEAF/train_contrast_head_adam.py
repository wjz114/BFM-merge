"""Use the frozen CE trainer with current-task contrast-projected head updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if '--deterministic' in sys.argv:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import train_prehead_distillation as trainer
from contrast_head_adam import ContrastHeadAdam
from d_downstream_qformer_ce import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=['full', 'contrast'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    assert args.condition in plan['contrast_conditions']
    assert sha256(Path(__file__)) == plan['source_sha256'][Path(__file__).name]
    for p, expected in plan['contrast_input_sha256'].items():
        assert sha256(Path(p)) == expected, p
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    captured, instances = [], []
    if args.condition == 'contrast':
        bases = torch.load(plan['contrast_bases'], map_location='cpu', weights_only=True)
        schedule = np.load(plan['schedule'])
        task_order = [plan['contrast_datasets'][int(t)] for t in schedule['task']]
        original_loader = trainer.load_leaf

        def capture_loader(*a, **kw):
            leaf = original_loader(*a, **kw)
            if not captured:
                captured.append(leaf)
            return leaf

        def optimizer_factory(parameters, **kw):
            named = [(n, p) for n, p in captured[0].named_parameters() if p.requires_grad]
            parameters = list(parameters)
            assert len(named) == len(parameters) and all(p is q for (_, p), q in zip(named, parameters))
            optimizer = ContrastHeadAdam(named, bases, task_order, args.outdir, **kw)
            instances.append(optimizer)
            return optimizer

        class TorchProxy:
            optim = SimpleNamespace(Adam=optimizer_factory)

            def __getattr__(self, name):
                return getattr(torch, name)

        trainer.load_leaf = capture_loader
        trainer.torch = TorchProxy()
    sys.argv = [trainer.__file__, '--plan', str(args.plan), '--condition', 'balanced_ce',
                '--outdir', str(args.outdir), '--gpu', str(args.gpu)]
    if args.smoke:
        sys.argv.append('--smoke')
    trainer.main()
    run_path = args.outdir/'run.json'
    run = json.loads(run_path.read_text())
    result = {'status': 'complete', 'condition': args.condition, 'plan_sha256': sha256(args.plan),
              'wrapper_sha256': sha256(Path(__file__)), 'frozen_trainer_sha256': sha256(Path(trainer.__file__)),
              'run_sha256': sha256(run_path), 'engineering_smoke': args.smoke,
              'deterministic_algorithms': args.deterministic, 'updates': run['updates'],
              'seconds_including_wrapper': time.time()-started,
              'execution': 'Unchanged balanced-CE trainer. Full arm uses native Adam directly. Contrast arm retains native gradients and Adam moments for every parameter, but replaces only the proposed final-linear-head weight/bias step with its FP64 projection into the current schedule task prototype-contrast span, materialized back to FP32. No norm restoration, cumulative-coordinate parameterization, hooks, new losses or runtime components.'}
    if args.condition == 'contrast':
        assert len(instances) == 1 and instances[0].steps == run['updates']
        stats = instances[0].statistics
        result['maximum_exact_source_contrast_relative_error'] = max(r['exact_source_contrast_relative_error'] for r in stats)
        result['maximum_materialized_source_contrast_relative_error'] = max(r['materialized_source_contrast_relative_error'] for r in stats)
        result['mean_head_decision_null_update_energy_fraction'] = float(np.mean([r['head_decision_null_update_energy_fraction'] for r in stats]))
        result['contrast_dynamics_sha256'] = sha256(args.outdir/'contrast_dynamics.jsonl')
    with (args.outdir/'wrapper.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('Wrapper completed', args.condition, flush=True)


if __name__ == '__main__':
    main()
