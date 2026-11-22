"""Freeze one native-dropout contrast at each of two existing initializations."""
import copy
import json
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    root = Path('runs/paper_merge_v2')
    old_path = root/'inputs/base_initialization_ce_lr1e5_plan_v2.json'
    old = json.loads(old_path.read_text())
    for name, digest in old['source_sha256'].items():
        assert sha256(Path(name)) == digest, name
    controls = {'merged': root/'seed0/full_control_coverage_full_balanced_ce/run.json',
                'base': root/'seed0/base_initialization_ce_lr1e5/run.json'}
    control_runs = {key: json.loads(path.read_text()) for key, path in controls.items()}
    assert all(run['schedule_sha256'] == old['schedule_sha256'] for run in control_runs.values())
    worker = Path('train_dropout_ce_control.py')
    source_hashes = copy.deepcopy(old['source_sha256'])
    source_hashes.update({str(p): sha256(p) for p in [worker, Path(__file__), old_path]})
    arms = {key: {'plan': str(root/f'inputs/dropout_ce_{key}_plan.json'),
                  'outdir': str(root/f'seed0/dropout_ce_{key}'),
                  'log': str(root/f'seed0/logs/dropout_ce_{key}.log'),
                  'gpu': i, 'control': str(path), 'control_sha256': sha256(path)}
            for i, (key, path) in enumerate(controls.items())}
    design_path = root/'inputs/dropout_ce_design.json'
    design = {
        'fixed_unix': time.time(), 'worker': str(worker), 'source_sha256': source_hashes,
        'hypothesis': 'Test whether the existing full-weight CE results depend on disabling the architecture-native training dropout that is enabled in the historical Joint recipe.',
        'intervention': 'Enable train_mode_fixed_bn before every optimizer update; eval mode for every metric and export. Keep native dropout probabilities and all BN statistics fixed. No other optimizer or data changes.',
        'arms': arms, 'datasets': control_runs['base']['datasets'],
        'lr': 1e-5, 'optimizer': 'Adam, weight_decay0, FP32, same383 active tensors',
        'steps': 1600, 'batch_size': 128, 'schedule': old['schedule'],
        'schedule_sha256': old['schedule_sha256'], 'average_steps': old['average_steps'],
        'primary': 'Both within-initialization dropout-on minus existing dropout-off full16 validation BAcc effects, their difference-in-differences, all16 task effects and leave-one-task-out sensitivity. The prescribed six-state mean is primary.',
        'engineering': 'Before full runs, require32-step smoke completion, finite active tower/QFormer gradients, unchanged frozen tensors, stochastic training predictions, exact deterministic eval replay, RNG-restored forward probe and ordinary export agreement. Full runs must replay initial all16 metrics.',
        'scope': 'One existing learning rate and one exposed validation bank/order. A conventional training-mode control, not a new method or an optimized short-Joint baseline. Same additional1600 updates and204800label exposures within each initialization; the initial merger retains its expert/coefficient costs. No dropout probability, rate or checkpoint sweep.',
        'literature': 'https://jmlr.org/papers/v15/srivastava14a.html',
        'non_dropout_joint_differences_retained': ['Adam versus AdamW', 'flat versus grouped learning rates and warmup/cosine', 'FP32 versus BF16', '1600 versus48000updates'],
    }
    write_new(design_path, design)
    for key, arm in arms.items():
        reference = root/f'inputs/dropout_ce_{key}_initial_reference.json'
        previous = control_runs[key]
        checkpoint = old['initial_checkpoint'] if key == 'base' else str(root/'seed0/original_fp32_tensor_balanced_avg/merged.ckpt')
        write_new(reference, {'folded_val': previous['initial_validation']['validation'],
                              'source': arm['control'], 'source_sha256': arm['control_sha256'],
                              'scope': 'Existing initial metrics; no selection.'})
        plan = {k: copy.deepcopy(old[k]) for k in ['source_fit', 'calibration', 'calibration_sha256',
                'schedule', 'schedule_sha256', 'training_pool', 'coverage_audit', 'coverage_audit_sha256',
                'seed', 'steps', 'batch_size', 'eval_every', 'average_steps', 'lr', 'conditions']}
        plan.update(fixed_unix=time.time(), design=str(design_path), design_sha256=sha256(design_path),
                    initial_checkpoint=checkpoint, initial_checkpoint_sha256=sha256(Path(checkpoint)),
                    initial_reference=str(reference), source_sha256=copy.deepcopy(source_hashes),
                    training_mode='native_dropout_fixed_bn', initialization=key, scope=design['scope'])
        plan['source_sha256'].update({str(reference): sha256(reference), arm['control']: arm['control_sha256']})
        write_new(Path(arm['plan']), plan)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path), 'arms': arms}, indent=2))


if __name__ == '__main__':
    main()
