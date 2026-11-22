"""Freeze the three-arm experiment and build data-free grouped expert bases."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from d_downstream_qformer_ce import sha256
from row_projected_adam import toy_checks

R = Path('runs/paper_merge_v2')
DESIGN = R/'inputs/row_projected_training_design.json'
BASIS = R/'seed0/row_projected_training_bases'
PLAN = R/'inputs/row_projected_training_plan.json'


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def prepare():
    row_plan_path = R/'inputs/row_expert_span_plan.json'
    row_plan = json.loads(row_plan_path.read_text())
    reference_path = R/'inputs/prehead_distillation_training_plan.json'
    reference = json.loads(reference_path.read_text())
    bank_path = Path(reference['source_fit'])/'run.json'; bank = json.loads(bank_path.read_text())
    paths = [row_plan_path, reference_path, bank_path, Path(bank['base_checkpoint']),
             *map(Path, bank['experts'].values()),
             R/'analysis/row_expert_span_summary.json',
             R/'analysis/prehead_engineering_verification.json',
             R/'seed0/prehead_ce_deterministic_replay/run.json']
    sources = {**reference['source_sha256'], **{p: sha256(Path(p)) for p in
               ['row_projected_adam.py', 'train_row_projected_adam.py', 'prepare_row_projected_training.py']}}
    check_directory = R/'seed0/row_projected_toy_checks'
    checks = toy_checks(check_directory)
    design = {'fixed_unix': time.time(), 'reference_training_plan': str(reference_path),
              'row_partition_plan': str(row_plan_path), 'source_fit': str(bank_path),
              'source_sha256': sources, 'input_sha256': {str(p): sha256(p) for p in paths},
              'conditions': ['full', 'row', 'shuffled'], 'active': row_plan['active'],
              'eligible': row_plan['eligible'], 'toy_checks': checks,
              'basis_rule': 'Same operator rows or fixed random within-tensor coordinate groups as row_expert_span_plan; other tensors keep their whole-tensor expert span. Expert-minus-base subtractionFP32, thenFP64 Gram rank check atrelative1e-10 and thin QR to construct an orthonormal basis. Every group must retain rank16, as in the completed diagnostic. Persist FP32 bases and permutations. Construction uses only base/expert weights and architecture, not the full-CE displacement, labels, teacher activations or Joint.',
              'optimizer_rule': 'All arms compute ordinary balanced CE and full native-coordinate Adam moments, lr1e-5, defaultbetas/eps, no decay. Full applies Adam directly. Constrained arms compute the actual FP32 Adam proposal u, encode/decode its orthogonal projection Pu, multiply by global ||u||/||Pu|| (zero projection gives zero move), accumulate projected coordinates, and rematerialize theta_initial+Qbeta. The norm is restored to that arms own current unprojected proposal, not matched across different trajectories. This controls norm loss due to projection without rotating Adam moments or imposing coordinatewise latent Adam. All actual step norms and scale factors are logged. No clipping or scale search.',
              'fixed_protocol': {'steps': 1600, 'lr': 1e-5, 'batch_size': 128, 'seed': 0,
                                 'average_steps': [800,960,1120,1280,1440,1600],
                                 'schedule': reference['schedule'], 'schedule_sha256': reference['schedule_sha256'],
                                 'objective': 'Existing full-pool class-balanced CE; dropoutoff/BNfixed; nativeFP32.'},
              'primary': 'All three six-state means, all16tasks, row versus shuffled and both versus fresh full CE; report effects, task-deletion sensitivity and Joint-gap comparison. Selected/final states secondary. No change of objective, learning rate, data sequence, partition or averaging window from validation results.',
              'engineering': 'Before scientific launch:32 deterministic updates for all3 arms. Full must reproduce the previous same-worker full-pool deterministic CE reference tensors/history exactly. Both projected smokes must be finite, active, ordinary-export-compatible, and show near-equal proposed/actual step norms at FP32 precision. Independently reconstruct stored means. This is engineering verification, not a 1600-step native bitwise guarantee.',
              'budget': {'scientific_optimizer_updates': 4800, 'engineering_optimizer_updates': 96,
                         'teacher_forward_examples': 0, 'scientific_training_exposures': 614400},
              'scope': 'Subsequent exploratory single-bank experiment motivated by completed row/shuffled fixed-update controls. It tests constrained optimization, not an existing established paper method. Generic fine-grained/chunkwise merging and projected optimization are prior art. Full Adam states and expert bases are retained during training; no memory/compute saving claim. All exports are ordinary LEAF checkpoints. Existing development validation is exposed; independent confirmation unresolved.'}
    write_new(DESIGN, design)
    print(json.dumps({'design': str(DESIGN), 'sha256': sha256(DESIGN), 'toy_checks': checks,
                      'budget': design['budget']}, indent=2))


@torch.no_grad()
def build():
    started = time.time(); torch.set_num_threads(4)
    design = json.loads(DESIGN.read_text())
    for p,h in {**design['source_sha256'], **design['input_sha256']}.items(): assert sha256(Path(p)) == h, p
    BASIS.mkdir(exist_ok=False)
    bank = json.loads(Path(design['source_fit']).read_text())
    base = torch.load(bank['base_checkpoint'], map_location='cpu', weights_only=True)
    experts = [torch.load(bank['experts'][t], map_location='cpu', weights_only=True) for t in bank['datasets']]
    outputs, geometry = {}, {}
    for kind in ['row', 'shuffled']:
        records, measurements = {}, {}
        for index, name in enumerate(design['active']):
            original_shape = list(base[name].shape)
            matrix = torch.stack([(e[name]-base[name]).flatten() for e in experts]).double()
            if name in design['eligible']:
                spec = design['eligible'][name]; groups, width = spec['rows'], spec['coordinates_per_row']
                if kind == 'shuffled':
                    seed = (int(hashlib.sha256(name.encode()).hexdigest()[:16],16)+20261119) % (2**63-1)
                    perm = torch.randperm(matrix.shape[-1], generator=torch.Generator().manual_seed(seed))
                    matrix = matrix[:,perm]
                else: perm = None
            else: groups, width, perm = 1, matrix.shape[-1], None
            matrix = matrix.reshape(16, groups, width).permute(1,0,2)
            pieces, ranks = [], []
            qr_error = fp32_error = source_error = 0.
            for start in range(0, groups, 128):
                a = matrix[start:start+128]
                values = torch.linalg.eigvalsh(a @ a.transpose(-1,-2))
                rank = (values > values[:,-1:].clamp_min(0)*1e-10).sum(-1)
                assert (rank == 16).all(), (kind, name, rank.min())
                q = torch.linalg.qr(a.transpose(-1,-2), mode='reduced').Q.transpose(-1,-2).contiguous()
                eye = torch.eye(16, dtype=torch.float64)
                qr_error = max(qr_error, float((q @ q.transpose(-1,-2)-eye).abs().max()))
                q32 = q.float(); qback = q32.double()
                fp32_error = max(fp32_error, float((qback @ qback.transpose(-1,-2)-eye).abs().max()))
                reconstruction = (a @ q.transpose(-1,-2)) @ q
                error = (reconstruction-a).flatten(1).norm(dim=-1)/a.flatten(1).norm(dim=-1)
                source_error = max(source_error, float(error.max()))
                pieces.append(q32); ranks.append(rank)
            assert qr_error < 1e-12 and source_error < 1e-12 and fp32_error < 1e-6
            records[name] = {'q': torch.cat(pieces), 'shape': original_shape, 'permutation': perm}
            measurements[name] = {'groups': groups, 'width': width, 'rank_sum': int(torch.cat(ranks).sum()),
                                  'fp64_orthogonality_max_abs': qr_error,
                                  'fp32_basis_gram_max_abs_error': fp32_error,
                                  'source_span_relative_reconstruction_error': source_error}
        path = BASIS/f'{kind}.pt'; torch.save(records, path)
        outputs[kind] = {'path': str(path), 'sha256': sha256(path),
                         'coordinate_count': sum(x['rank_sum'] for x in measurements.values())}
        geometry[kind] = measurements
        del records
        print('Basis saved', kind, outputs[kind], flush=True)
    assert outputs['row']['coordinate_count'] == outputs['shuffled']['coordinate_count'] == 1132560
    result = {'status': 'complete', 'design_sha256': sha256(DESIGN), 'code_sha256': sha256(Path(__file__)),
              'active': design['active'], 'outputs': outputs, 'geometry': geometry, 'seconds': time.time()-started,
              'forward_examples': 0, 'optimizer_updates': 0,
              'basis_rule': design['basis_rule']}
    write_new(BASIS/'run.json', result)
    print('Completed', result['seconds'], flush=True)


def materialize():
    design = json.loads(DESIGN.read_text()); basis = json.loads((BASIS/'run.json').read_text())
    assert basis['status'] == 'complete' and basis['design_sha256'] == sha256(DESIGN)
    reference = json.loads(Path(design['reference_training_plan']).read_text())
    plan = {**reference, 'fixed_unix': time.time(), 'source_sha256': design['source_sha256'],
            'conditions': ['balanced_ce'], 'projection_conditions': design['conditions'],
            'projection_design': str(DESIGN), 'projection_design_sha256': sha256(DESIGN),
            'basis_manifest': str(BASIS/'run.json'), 'basis_manifest_sha256': sha256(BASIS/'run.json'),
            'scope': design['scope'], 'projection_execution': 'Use train_row_projected_adam.py; inherited trainer condition is balanced_ce for all arms. Scientific condition is explicitly recorded in wrapper.json.'}
    for key,value in design['fixed_protocol'].items():
        if key in plan: assert plan[key] == value, key
    write_new(PLAN, plan)
    print(json.dumps({'plan': str(PLAN), 'sha256': sha256(PLAN)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'build', 'materialize'])
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'build': build()
    else: materialize()
