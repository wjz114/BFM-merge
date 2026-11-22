"""Fit the same fixed ridge heads with a Kronecker eigen preconditioner."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from d_downstream_qformer_ce import sha256


WEIGHT = 'qformer.output_embed.2.weight'
BIAS = 'qformer.output_embed.2.bias'


def conjugate_gradient(operator, rhs, precondition, tolerance=1e-8, maximum=1000):
    value = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = float(rhs.norm())
    assert rhs_norm > 0
    z = precondition(residual)
    direction = z.clone()
    rz = (residual*z).sum()
    history = []
    for iteration in range(1, maximum+1):
        product = operator(direction)
        denominator = (direction*product).sum()
        assert denominator > 0 and torch.isfinite(denominator)
        alpha = rz/denominator
        value += alpha*direction
        residual -= alpha*product
        relative = float(residual.norm())/rhs_norm
        if iteration == 1 or iteration % 50 == 0 or relative <= tolerance:
            history.append({'iteration': iteration, 'recursive_relative_residual': relative})
            print('CG', iteration, relative, flush=True)
        if relative <= tolerance:
            break
        z = precondition(residual)
        next_rz = (residual*z).sum()
        direction = z+(next_rz/rz)*direction
        rz = next_rz
    actual = float((operator(value)-rhs).norm())/rhs_norm
    assert actual <= tolerance*1.01, ('normal-equation residual', actual)
    return value, {'iterations': iteration, 'actual_relative_residual': actual, 'history': history}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(path)) == expected, path
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    cache = json.loads(Path(plan['feature_cache']).read_text())
    design = json.loads(Path(plan['design']).read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    assert sha256(Path(geometry['basis_path'])) == geometry['basis_sha256']
    bases = torch.load(geometry['basis_path'], map_location='cpu', weights_only=True)
    union, own = bases['union_basis'].double(), bases['own_bases']
    assert union.shape == (22, 768)
    assert float((union@union.T-torch.eye(22)).abs().max()) < 1e-10
    initial = torch.load(plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    tasks = cache['datasets']
    stats, per_task = {}, {}
    for task in tasks:
        row = cache['records'][task]
        assert sha256(Path(row['path'])) == row['sha256']
        data = torch.load(row['path'], map_location='cpu', weights_only=True)
        n = data['n_calibration']
        assert n == 512
        x = data['features'][:n].double()
        x = torch.cat([x, torch.ones((n, 1), dtype=torch.float64)], dim=1)
        raw = data['raw_output'][:n].double()
        teacher = data['teacher_raw_calibration'].double()
        target = raw.norm(dim=1, keepdim=True)*F.normalize(teacher, dim=-1)-raw
        weights = data['class_weights'][data['truth'][:n]].double()
        assert abs(float(weights.mean())-1) < 1e-6
        covariance = x.T@(weights[:, None]*x)/n
        cross = target.T@(weights[:, None]*x)/n
        target_covariance = target.T@(weights[:, None]*target)/n
        stats[task] = {'S': covariance, 'F': cross, 'R': target_covariance}
        per_task[task] = {'fit_examples': n, 'input_dimension': x.shape[1],
                          'own_contrast_rank': own[task].shape[0],
                          'target_squared_norm_mean': float(target_covariance.trace())}
        del data, x, raw, teacher, target
    dimension = initial[WEIGHT].shape[1]+1
    average_s = sum(v['S'] for v in stats.values())/len(tasks)
    mu = float(average_s.trace()/dimension)
    ridge = design['lambda']*mu
    input_eigenvalues, input_eigenvectors = torch.linalg.eigh(average_s)
    assert float(input_eigenvalues.min()) >= -1e-8*float(input_eigenvalues.max())
    input_eigenvalues = input_eigenvalues.clamp_min(0)
    generator = torch.Generator().manual_seed(design['random_seed'])
    random_frame, _ = torch.linalg.qr(torch.randn((768, 22), generator=generator, dtype=torch.float64), mode='reduced')
    frames = {'prototype_contrast': union, 'rotated_contrast_control': random_frame.T}
    outputs = {}
    for condition in ['prototype_contrast', 'rotated_contrast_control', 'full_feature_ridge']:
        fit_started = time.time()
        if condition == 'full_feature_ridge':
            system = average_s+ridge*torch.eye(dimension, dtype=torch.float64)
            rhs = sum(v['F'] for v in stats.values())/len(tasks)
            factor = torch.linalg.cholesky(system)
            delta = torch.cholesky_solve(rhs.T, factor).T
            actual = float((delta@system-rhs).norm()/rhs.norm())
            assert actual < 1e-8
            solver = {'method': 'Cholesky', 'actual_relative_residual': actual}
            frame = torch.eye(768, dtype=torch.float64)
            b = delta
        else:
            frame = frames[condition]
            rhs = torch.zeros((22, dimension), dtype=torch.float64)
            q_matrices = {}
            for task, st in stats.items():
                d = own[task].double()@union.T
                a = d@frame
                rank = d.shape[0]
                q = d.T@d/rank
                q_matrices[task] = q
                rhs += d.T@a@st['F']/(rank*len(tasks))

            def operator(value):
                result = ridge/22*value
                for task, st in stats.items():
                    result = result+q_matrices[task]@value@st['S']/len(tasks)
                return result

            q_mean = sum(q_matrices.values())/len(tasks)
            q_values, q_vectors = torch.linalg.eigh(q_mean)
            assert float(q_values.min()) >= -1e-10
            denominator = q_values.clamp_min(0)[:, None]*input_eigenvalues[None, :]+ridge/22

            def precondition(residual):
                rotated = q_vectors.T@residual@input_eigenvectors
                return q_vectors@(rotated/denominator)@input_eigenvectors.T

            b, solver = conjugate_gradient(operator, rhs, precondition)
            solver['method'] = 'Kronecker-eigen-preconditioned CG'
            delta = frame.T@b
        state = dict(initial)
        state[WEIGHT] = (initial[WEIGHT].double()+delta[:, :-1]).float()
        state[BIAS] = (initial[BIAS].double()+delta[:, -1]).float()
        assert all(torch.isfinite(state[k]).all() for k in [WEIGHT, BIAS])
        assert all(torch.equal(state[k], initial[k]) for k in initial if k not in [WEIGHT, BIAS])
        # Evaluate only the fitting objective from calibration sufficient statistics.
        losses, initial_losses = {}, {}
        for task, st in stats.items():
            if condition == 'full_feature_ridge':
                metric = torch.eye(768, dtype=torch.float64)/768
            else:
                a = (own[task].double()@union.T)@frame
                metric = a.T@a/a.shape[0]
            initial_losses[task] = float((metric*st['R'].T).sum())
            residual_covariance = delta@st['S']@delta.T-delta@st['F'].T-st['F']@delta.T+st['R']
            losses[task] = float((metric*residual_covariance.T).sum())
        regularizer = ridge/frame.shape[0]*float(b.square().sum())
        before = sum(initial_losses.values())/len(tasks)
        after = sum(losses.values())/len(tasks)+regularizer
        assert after <= before+1e-9
        path = args.outdir/f'{condition}.ckpt'
        torch.save(state, path)
        outputs[condition] = {'path': str(path), 'sha256': sha256(path), 'solver': solver,
                              'delta_norm': float(delta.norm()), 'changed_parameters': [WEIGHT, BIAS],
                              'fit_objective_before': before, 'fit_objective_after': after,
                              'regularization_term': regularizer, 'fit_loss_per_task': losses,
                              'seconds': time.time()-fit_started}
        (args.outdir/'progress.json').write_text(json.dumps(outputs, indent=2)+'\n')
        print(condition, 'objective', before, after, 'residual', solver['actual_relative_residual'], flush=True)
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'datasets': tasks, 'per_task': per_task, 'mu': mu, 'lambda': design['lambda'],
              'outputs': outputs, 'seconds': time.time()-started,
              'trunk_tensors_unchanged': True, 'model_inference_performed': False,
              'validation_used_in_fit': False, 'test_data_read': False,
              'scope': 'Fixed class-balanced ridge fits using only512-row calibration prefixes. Cached validation tensors are loaded with each artifact but never used in statistics, objective or fitting. This produces three ordinary checkpoints; native full-validation/export checks and efficacy are still pending. No novelty claim.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
