"""Independent active-set and native-coordinate tests for the proposed projector."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from d_downstream_qformer_ce import sha256
from streaming_step_projection import RowGeometry, project_proposal


def exhaustive_primal(proposal, gradients, metric):
    """Enumerate active constraints and solve the primal KKT linear system."""
    p, g, m = proposal.numpy(), gradients.numpy(), metric.numpy()
    candidates = []
    for count in range(len(g) + 1):
        for selected in itertools.combinations(range(len(g)), count):
            a = g[list(selected)]
            kkt = np.block([[m, -a.T], [a, np.zeros((count, count))]])
            rhs = np.concatenate([m @ p, np.zeros(count)])
            solution = np.linalg.lstsq(kkt, rhs, rcond=1e-13)[0]
            q, dual = solution[:len(p)], solution[len(p):]
            if (np.max(np.abs(kkt @ solution-rhs)) < 1e-8
                    and (not count or dual.min() >= -1e-8) and (g @ q).min() >= -1e-8):
                candidates.append((float((q-p) @ m @ (q-p)), q))
    assert candidates
    return min(candidates, key=lambda x:x[0])[1]


def main():
    torch.set_num_threads(4)
    root = Path('runs/streaming_step_projection_v1')
    assert not (root / 'algebra.json').exists()
    generator = torch.Generator().manual_seed(217)
    errors, violations = [], []
    for case in range(24):
        rows, tasks = 3, 4
        a = torch.randn(rows, 2, 2, generator=generator, dtype=torch.float64)
        metric = a @ a.transpose(1, 2) + .1 * torch.eye(2, dtype=torch.float64)
        proposal = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
        gradients = torch.randn(tasks, rows, 2, generator=generator, dtype=torch.float64)
        if case == 0:
            gradients.zero_()
        if case == 1:
            proposal.zero_()
        if case == 2:
            gradients[1] = -gradients[0]
        if case == 3:
            gradients[1] = gradients[0]
        projected, info = project_proposal(proposal, gradients, metric, torch.linalg.inv(metric))
        dense_metric = torch.block_diag(*metric)
        expected = exhaustive_primal(proposal.flatten(), gradients.flatten(1), dense_metric)
        errors.append(float(np.max(np.abs(projected.flatten().numpy() - expected))))
        violations.append(min(info['after_margins']))
    assert max(errors) < 2e-5, max(errors)
    assert min(violations) > -2e-5, min(violations)
    # Toy tensors include a matrix, scalar, rank-one row plane, and zero delta.
    base = dict(matrix=torch.zeros(3,4),scalar=torch.zeros(()),zero=torch.zeros(2))
    previous = {k:torch.randn(v.shape,generator=generator) for k,v in base.items()}
    incoming = {k:torch.randn(v.shape,generator=generator) for k,v in base.items()}
    previous['zero'].zero_(); incoming['zero'].zero_()
    geometry = RowGeometry(base, previous, incoming, list(base))
    proposal = torch.randn(geometry.count,generator=generator,dtype=torch.float64)
    collapsed = geometry.collapse(proposal)
    native_gradient = {k:torch.randn(v.shape,generator=generator,dtype=torch.float64) for k,v in base.items()}
    full_gradient = torch.zeros(geometry.count,dtype=torch.float64)
    offset, native_dot, native_squared = 0, 0., 0.
    for j,(name,first,last) in enumerate(geometry.slices):
        rows = last-first
        old = (previous[name]-base[name]).double().reshape(rows,-1)
        new = (incoming[name]-base[name]).double().reshape(rows,-1)
        grad = native_gradient[name].reshape(rows,-1)
        old_g,new_g=(grad*old).sum(1),(grad*new).sum(1)
        full_gradient[j]=old_g.sum();full_gradient[geometry.n+j]=new_g.sum()
        full_gradient[2*geometry.n+first:2*geometry.n+last]=new_g
        full_gradient[2*geometry.n+geometry.rows+first:2*geometry.n+geometry.rows+last]=old_g
        delta = collapsed[first:last,0,None]*old+collapsed[first:last,1,None]*new
        native_dot += float((grad*delta).sum());native_squared += float(delta.square().sum())
    row_grad = geometry.row_gradients(full_gradient[None])[0]
    dot_errors=[abs(float(full_gradient@proposal)-native_dot),abs(float((row_grad*collapsed).sum())-native_dot)]
    gram_squared=float(torch.einsum('ri,rij,rj->',collapsed,geometry.gram,collapsed))
    assert max(dot_errors)<1e-10 and abs(gram_squared-native_squared)<1e-10
    correction=torch.randn(geometry.rows,2,generator=generator,dtype=torch.float64)
    assert torch.equal(geometry.collapse(geometry.lift(correction)),correction)
    metric_value=float(torch.einsum('ri,rij,rj->',collapsed,geometry.native_metric,collapsed))
    assert abs(metric_value-native_squared-float((geometry.ridge[:,None]*collapsed.square()).sum()))<1e-10
    result=dict(status='passed',cases=24,active_set_primal_max_abs_error=max(errors),
        minimum_raw_constraint_margin=min(violations),effective_row_dot_max_abs_error=max(dot_errors),
        native_gram_squared_norm_abs_error=abs(gram_squared-native_squared),
        row_correction_lift_exact=True,zero_and_rank_one_rows_checked=True,
        sha256={f:sha256(Path(f)) for f in ['streaming_step_projection.py','check_step_projection_algebra.py',
            str(root/'design.json'),str(root/'prior_boundary.json')]},
        model_forwards=0,optimizer_updates=0,official_test_read=False)
    with (root/'algebra.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
