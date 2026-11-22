"""Execute the two frozen direct-base CE stages using the existing worker."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--rate', choices=['lr1e5', 'lr1e4'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    design_hash = sha256(args.design)
    for name, expected in design['source_sha256'].items():
        assert sha256(Path(name)) == expected, name
    arm = design['arms'][args.rate]
    result_dir = Path(arm['controller_outdir'])
    result_dir.mkdir(exist_ok=False)
    events_path = result_dir/'events.jsonl'
    started = time.time()

    def event(**value):
        with events_path.open('a') as f:
            f.write(json.dumps({'unix': time.time(), **value})+'\n')

    def execute(stage, plan_path, outdir):
        stage_plan = json.loads(plan_path.read_text())
        assert stage_plan['design_sha256'] == design_hash
        assert stage_plan['lr'] == arm['lr']
        command = [sys.executable, design['worker'], '--plan', str(plan_path),
                   '--condition', 'balanced_ce', '--outdir', str(outdir), '--gpu', str(args.gpu)]
        log_path = Path(arm[f'stage{stage}_log'])
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            event(stage=stage, state='running', pid=process.pid, command=command,
                  plan_sha256=sha256(plan_path), log=str(log_path))
            print('started', stage, 'pid', process.pid, flush=True)
            code = process.wait()
        event(stage=stage, state='terminal', exit_code=code)
        if code:
            raise RuntimeError(f'Stage {stage} exited with code {code}; retain partial output and log.')
        run_path = outdir/'run.json'
        run = json.loads(run_path.read_text())
        assert run['plan_sha256'] == sha256(plan_path)
        assert run['updates'] == 1600 and run['training_examples'] == 204800
        assert run['average_steps'] == design['average_steps']
        assert run['fixed_tensors_exact'] and not run['engineering_smoke']
        assert run['trainable_parameters'] == 25751552
        assert run['training_pool'] == ('original512' if stage == 1 else 'full_training')
        assert not run['test_data_read'] and not run['test_inference_performed']
        assert sha256(outdir/'merged.ckpt') == run['checkpoint_sha256']
        assert max(v['max_abs_logit_error'] for v in run['export_checks'].values()) <= 1e-5
        return run, run_path

    stage1, stage1_path = execute(1, Path(arm['stage1_plan']), Path(arm['stage1_outdir']))
    # The transition is fixed in the design: always the six-state mean and fresh Adam.
    reference_path = Path(arm['stage2_reference'])
    reference = {'folded_val': stage1['average_validation'],
                 'checkpoint': str(stage1_path.with_name('merged.ckpt')),
                 'checkpoint_sha256': stage1['checkpoint_sha256'],
                 'source_run': str(stage1_path), 'source_run_sha256': sha256(stage1_path),
                 'scope': 'Schema adapter for exact stage-transition metric replay; no selection.'}
    write_new(reference_path, reference)
    stage2_plan = copy.deepcopy(arm['stage2_plan_template'])
    stage2_plan.update(fixed_unix=time.time(), design=str(args.design), design_sha256=design_hash,
                       initial_checkpoint=reference['checkpoint'],
                       initial_checkpoint_sha256=reference['checkpoint_sha256'],
                       initial_reference=str(reference_path),
                       previous_stage_run=str(stage1_path), previous_stage_run_sha256=sha256(stage1_path))
    stage2_plan['source_sha256'].update({str(reference_path): sha256(reference_path),
                                       str(stage1_path): sha256(stage1_path)})
    stage2_path = Path(arm['stage2_plan'])
    write_new(stage2_path, stage2_plan)
    event(stage=2, state='plan_materialized', plan=str(stage2_path), plan_sha256=sha256(stage2_path),
          initializer='fixed_stage1_six_checkpoint_mean', optimizer='fresh_Adam')
    stage2, stage2_run_path = execute(2, stage2_path, Path(arm['stage2_outdir']))
    assert all(stage2['initial_validation']['validation'][t]['BAcc'] == stage1['average_validation'][t]['BAcc'] for t in stage1['datasets'])
    result = {'design': str(args.design), 'design_sha256': design_hash, 'rate': args.rate, 'lr': arm['lr'],
              'code_sha256': sha256(Path(__file__)),
              'stage1': {'path': str(stage1_path), 'sha256': sha256(stage1_path)},
              'stage2': {'path': str(stage2_run_path), 'sha256': sha256(stage2_run_path)},
              'stage_transition_bacc_exact': True, 'updates': 3200, 'training_exposures': 409600,
              'primary_checkpoint': str(stage2_run_path.with_name('merged.ckpt')),
              'primary_checkpoint_sha256': stage2['checkpoint_sha256'],
              'seconds': time.time()-started,
              'scope': design['budget_scope']}
    write_new(result_dir/'run.json', result)
    event(state='completed', run_sha256=sha256(result_dir/'run.json'))
    print('completed', args.rate, 'fixed mean', stage2['average_validation_macro_bacc'], flush=True)


if __name__ == '__main__':
    main()
