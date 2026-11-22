"""Run the frozen balanced-CE trainer with an explicitly scoped optimizer wrapper."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if '--deterministic' in sys.argv:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import train_prehead_distillation as trainer
from d_downstream_qformer_ce import sha256
from row_projected_adam import ProjectedAdam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--condition', choices=['full', 'row', 'shuffled'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    assert args.condition in plan['projection_conditions']
    assert sha256(Path(__file__)) == plan['source_sha256'][Path(__file__).name]
    assert sha256(Path(plan['projection_design'])) == plan['projection_design_sha256']
    assert sha256(Path(plan['basis_manifest'])) == plan['basis_manifest_sha256']
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    instances, captured = [], []
    if args.condition != 'full':
        manifest = json.loads(Path(plan['basis_manifest']).read_text())
        spec = manifest['outputs'][args.condition]
        assert sha256(Path(spec['path'])) == spec['sha256']
        records = torch.load(spec['path'], map_location='cpu', weights_only=True)
        assert list(records) == manifest['active']
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
            snapshots = [16, 32] if args.smoke else plan['average_steps']
            optimizer = ProjectedAdam(named, records, args.outdir, snapshots, **kw)
            instances.append(optimizer)
            return optimizer

        class TorchProxy:
            # Only this trainer module sees the substituted optimizer constructor.
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
    result = {'status': 'complete', 'condition': args.condition,
              'plan_sha256': sha256(args.plan), 'wrapper_sha256': sha256(Path(__file__)),
              'frozen_trainer_sha256': sha256(Path(trainer.__file__)), 'run_sha256': sha256(run_path),
              'engineering_smoke': args.smoke, 'deterministic_algorithms': args.deterministic,
              'updates': run['updates'], 'seconds_including_wrapper': time.time()-started,
              'execution': 'Unchanged train_prehead_distillation.main runs balanced_ce with auxiliary scale0. Row/shuffled processes replace only that module torch lookup with a proxy whose optim.Adam constructor returns ProjectedAdam; the global torch module and all source files remain unchanged. Full processes use the original torch.optim.Adam without a proxy. Existing cached teacher features are loaded by the frozen worker but do not contribute to the CE objective; no teacher inference.'}
    if args.condition != 'full':
        assert len(instances) == 1 and instances[0].steps == run['updates']
        o = instances[0]
        assert all(len(v) == len(records) for v in [o.params, o.initial, o.coordinates])
        result['coordinate_count'] = sum(b.numel() for b in o.coordinates)
        result['average_coordinate_audit'] = o.audit_average(run['average_steps'])
        result['step_norm_ratio_range'] = [min(x['actual_to_proposal_norm_ratio'] for x in o.statistics if x['actual_to_proposal_norm_ratio'] is not None),
                                           max(x['actual_to_proposal_norm_ratio'] for x in o.statistics if x['actual_to_proposal_norm_ratio'] is not None)]
        result['restoration_scale_range'] = [min(x['restoration_scale'] for x in o.statistics), max(x['restoration_scale'] for x in o.statistics)]
        result['zero_projection_steps'] = sum(x['zero_projection'] for x in o.statistics)
        result['basis_sha256'] = spec['sha256']
        result['projection_dynamics_sha256'] = sha256(args.outdir/'projection_dynamics.jsonl')
        result['coordinate_snapshot_sha256'] = {str(p): sha256(p) for p in args.outdir.glob('projection_coordinates_*.pt')}
    with (args.outdir/'wrapper.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('Wrapper completed', args.condition, flush=True)


if __name__ == '__main__':
    main()
