"""Analytic boundary of update-only projection; no EEG data or model evaluation."""
import hashlib
import json
from pathlib import Path
import time

import torch

from row_projected_adam import ProjectedAdam


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    root = Path('runs/paper_merge_v2')
    out = root/'analysis/projected_optimizer_geometry'
    out.mkdir(exist_ok=False)
    dtype = torch.float64

    # P is an exact orthogonal projector onto {(x,x)}. For this positive
    # diagonal preconditioner, Euclidean projection can reverse descent.
    p = torch.full((2, 2), .5, dtype=dtype)
    g = torch.tensor([2., -1.], dtype=dtype)
    a = torch.diag(torch.tensor([.1, 1.], dtype=dtype))
    direction = -p @ a @ g
    tangent_direction = -p @ a @ p @ g
    assert torch.allclose(p @ p, p)
    assert float(g @ direction) > 0
    assert float(g @ tangent_direction) < 0
    eta = .1
    loss = lambda x: float(.5 * x.square().sum() + g @ x)
    analytic = {
        'projector': p.tolist(), 'gradient': g.tolist(),
        'positive_diagonal_preconditioner': a.diag().tolist(),
        'update_only_direction': direction.tolist(),
        'gradient_dot_update_only_direction': float(g @ direction),
        'project_gradient_and_update_direction': tangent_direction.tolist(),
        'gradient_dot_project_gradient_and_update_direction': float(g @ tangent_direction),
        'objective': '0.5*||theta||^2 + [2,-1]^T theta; theta=(x,x)',
        'step_size': eta, 'initial_loss': loss(torch.zeros(2, dtype=dtype)),
        'update_only_loss': loss(eta * direction),
        'project_gradient_and_update_loss': loss(eta * tangent_direction),
        'scope': 'A fixed positive diagonal preconditioner counterexample, not an Adam convergence theorem or a LEAF trajectory measurement. Positive global norm restoration cannot reverse the directional-derivative sign.'}

    # Two ambient objectives agree everywhere on {(x,x,z)}. Their normal
    # gradients differ. Exercise the unchanged, norm-restored Adam wrapper.
    q = torch.tensor([[[2**-.5, 2**-.5, 0.], [0., 0., 1.]]], dtype=dtype)
    ambient_gradients = [torch.tensor(v, dtype=dtype) for v in [[1., 1., 1.], [4., -2., 1.]]]
    projector = q[0].T @ q[0]
    assert torch.allclose(projector @ ambient_gradients[0], projector @ ambient_gradients[1], atol=1e-14, rtol=0)
    records = {}
    for i, linear in enumerate(ambient_gradients):
        folder = out/f'ambient_objective_{i}'
        folder.mkdir()
        theta = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        optimizer = ProjectedAdam([('w', theta)],
            {'w': {'q': q, 'shape': [3], 'permutation': None}},
            folder, [], lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0)
        objective = .5*theta.square().sum() + linear @ theta
        objective.backward()
        optimizer.step()
        delta = theta.detach()
        assert torch.allclose(projector @ delta, delta, atol=1e-14, rtol=0)
        records[str(i)] = {
            'ambient_linear_term': linear.tolist(),
            'tangent_gradient': (projector @ linear).tolist(),
            'first_update': delta.tolist(),
            'first_step_statistics': optimizer.statistics[0]}
    x, y = [torch.tensor(records[str(i)]['first_update'], dtype=dtype) for i in range(2)]
    cosine = float(x @ y / (x.norm()*y.norm()))
    assert cosine < .6
    result = {
        'unix': time.time(),
        'sources': {str(p): digest(p) for p in [Path(__file__), Path('row_projected_adam.py')]},
        'analytic_counterexample': analytic,
        'same_restricted_objective_counterexample': {
            'allowed_space': 'theta=(x,x,z)',
            'common_restricted_objective': 'x^2 + z^2/2 + 2*x + z',
            'difference_of_ambient_objectives': '3*(theta[0]-theta[1]), identically zero on the allowed space',
            'optimizer': 'unchanged ProjectedAdam, CPU FP64, one step, native Adam moments then update projection and global norm restoration',
            'runs': records, 'update_cosine': cosine,
            'max_abs_update_difference': float((x-y).abs().max()),
            'interpretation': 'Identical restricted objectives and tangent gradients need not produce identical feasible updates under this rule. Full-gradient Adam statistics retain information from the ambient extension.'},
        'implications': [
            'The running experiment evaluates a fixed feasible-update recipe, not optimization independent of the ambient objective extension.',
            'An adverse result cannot by itself establish that the row expert space lacks a useful solution.',
            'A favorable row-versus-shuffled result identifies the joint partition/optimizer intervention; it does not isolate representational capacity from optimizer interaction.',
            'This toy does not establish that the observed LEAF trajectories encounter ascent, stall, or would improve under another optimizer.',
            'No live worker, frozen design, learning rate, projection rule or scientific condition was changed.'],
        'new_leaf_forwards': 0, 'new_leaf_updates': 0, 'toy_optimizer_updates': 2}
    path = root/'analysis/projected_optimizer_geometry.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(path), 'sha256': digest(path),
        'ascent_directional_derivative': analytic['gradient_dot_update_only_direction'],
        'same_objective_update_cosine': cosine}, indent=2))


if __name__ == '__main__':
    main()
