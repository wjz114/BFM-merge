"""Freeze one historical tokenizer/rest LR-ratio control from each starting point."""
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
    gate_path = root/'analysis/grouped_ce_optimizer_gate.json'
    gate = json.loads(gate_path.read_text())
    for path, digest in {**old['source_sha256'], **gate['source_sha256']}.items():
        assert sha256(Path(path)) == digest, path
    joint_path = Path('runs/poc16/seed0/joint/run.json')
    joint = json.loads(joint_path.read_text())
    ratio = joint['lr_tokenizer']/joint['lr_rest']
    assert abs(ratio-10) < 1e-12
    controls = {'merged': root/'seed0/full_control_coverage_full_balanced_ce/run.json',
                'base': root/'seed0/base_initialization_ce_lr1e5/run.json'}
    control_runs = {key: json.loads(path.read_text()) for key, path in controls.items()}
    assert all(run['schedule_sha256'] == old['schedule_sha256'] for run in control_runs.values())
    worker = Path('train_grouped_ce_control.py')
    source_hashes = copy.deepcopy(old['source_sha256'])
    source_hashes.update({str(p): sha256(p) for p in [worker, Path('grouped_ce_optimizer.py'),
                         Path(__file__), old_path, gate_path, joint_path]})
    arms = {key: {'plan': str(root/f'inputs/grouped_ce_{key}_plan.json'),
                  'outdir': str(root/f'seed0/grouped_ce_{key}'),
                  'log': str(root/f'seed0/logs/grouped_ce_{key}.log'),
                  'gpu': i, 'control': str(path), 'control_sha256': sha256(path)}
            for i, (key, path) in enumerate(controls.items())}
    design_path = root/'inputs/grouped_ce_design.json'
    design = {
        'fixed_unix': time.time(), 'worker': str(worker), 'source_sha256': source_hashes,
        'hypothesis': 'Test whether flat-rate adaptation understates the value of tokenizer adaptation relative to the historical tokenizer/rest split, and whether any effect depends on merger initialization.',
        'intervention': 'Only replace flatAdam1e-5 by tokenizerAdam1e-4/restAdam1e-5. Six tokenizer affine tensors follow the exact historical partition. Native dropout remains off; BN fixed. No Joint checkpoint values enter construction.',
        'ratio_rule': 'Historical Joint configuration ratio10; not fitted to checkpoint geometry, losses or validation. Absolute rest rate is the already-existing1e-5 repair rate.',
        'arms': arms, 'datasets': control_runs['base']['datasets'],
        'lr': 1e-5, 'tokenizer_multiplier': 10, 'optimizer': 'Adam, weight_decay0, FP32, eval mode,383active tensors',
        'steps': 1600, 'batch_size': 128, 'schedule': old['schedule'],
        'schedule_sha256': old['schedule_sha256'], 'average_steps': old['average_steps'],
        'engineering_gate': str(gate_path), 'engineering_gate_sha256': sha256(gate_path),
        'primary': 'Both within-initialization grouped-minus-flat fixed-mean full16 validation BAcc effects, their difference-in-differences, all16task effects and task-deletion sensitivity. Report both starts, without best-rate/checkpoint substitution.',
        'engineering': 'Optimizer-only gate: identical ratio1 legacy steps, exact historical parameter partition, unchanged slow parameters/moments and10xfast displacement within FP32 rounding under identical cached gradients. Reuse verified data/update/averaging/export loop. Full runs must reproduce all16 initial metrics, data hashes and fixed states.',
        'motivation_diagnostic': {'path': str(root/'analysis/optimizer_group_geometry.json'),
                                  'sha256': sha256(root/'analysis/optimizer_group_geometry.json'),
                                  'role': 'Uses Joint weights only for descriptive coordinate geometry; not causal evidence or candidate construction.'},
        'scope': 'Conventional optimizer control, not a new method, full Joint recipe or independent confirmation. Same additional1600updates and204800exposures; expert/coefficient costs of the merger remain separate. One fixed historical ratio and exposed bank/order; no ratio/rate sweep.',
        'remaining_joint_differences': ['Adam versus AdamW', 'flat versus warmup/cosine schedule',
             'absolute rates', 'dropout off versus on', 'FP32 versus BF16', '1600 versus48000updates',
             'class-balanced versus ordinary CE', 'sampling stream'],
    }
    write_new(design_path, design)
    for key, arm in arms.items():
        reference = root/f'inputs/grouped_ce_{key}_initial_reference.json'
        previous = control_runs[key]
        checkpoint = old['initial_checkpoint'] if key == 'base' else str(root/'seed0/original_fp32_tensor_balanced_avg/merged.ckpt')
        write_new(reference, {'folded_val': previous['initial_validation']['validation'],
                              'source': arm['control'], 'source_sha256': arm['control_sha256'],
                              'scope': 'Existing initial metrics, no selection.'})
        plan = {k: copy.deepcopy(old[k]) for k in ['source_fit', 'calibration', 'calibration_sha256',
                'schedule', 'schedule_sha256', 'training_pool', 'coverage_audit', 'coverage_audit_sha256',
                'seed', 'steps', 'batch_size', 'eval_every', 'average_steps', 'lr', 'conditions']}
        plan.update(fixed_unix=time.time(), design=str(design_path), design_sha256=sha256(design_path),
                    initial_checkpoint=checkpoint, initial_checkpoint_sha256=sha256(Path(checkpoint)),
                    initial_reference=str(reference), source_sha256=copy.deepcopy(source_hashes),
                    tokenizer_multiplier=10, initialization=key, scope=design['scope'])
        plan['source_sha256'].update({str(reference): sha256(reference), arm['control']: arm['control_sha256']})
        write_new(Path(arm['plan']), plan)
    print(json.dumps({'design': str(design_path), 'sha256': sha256(design_path), 'arms': arms}, indent=2))


if __name__ == '__main__':
    main()
