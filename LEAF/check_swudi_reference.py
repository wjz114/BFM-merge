"""Numerically compare the local matrix solver with pinned official pure functions."""
import ast
import hashlib
import json
import time
from pathlib import Path

import torch

from build_swudi_control import spectral_delta
from d_downstream_qformer_ce import sha256


def main():
    started = time.time()
    torch.set_num_threads(4)
    root = Path('runs/paper_merge_v2')
    plan_path = root/'inputs/swudi_control_plan.json'
    plan = json.loads(plan_path.read_text())
    source_path = root/'literature/swudi_functional_reference.py'
    raw = source_path.read_bytes()
    # apply_patch may append a final blank line; recover only the exact Git blob.
    candidates = [raw] + [raw[:-i] for i in range(1, 4) if raw.endswith(b'\n'*i)]
    matching = [data for data in candidates if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == plan['reference_functional_blob']]
    assert len(matching) == 1
    raw = matching[0]
    tree = ast.parse(raw.decode())
    rank = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_aswudi_pick_rank')
    outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'aswudi_merge')
    solver = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == '_solve')
    # Execute only these two audited pure numerical functions, not module imports
    # or the repository's loading, CLI, traversal and output code.
    module = ast.fix_missing_locations(ast.Module(body=[rank, solver], type_ignores=[]))
    namespace = {'torch': torch, 'device': None, 'min_eigenvalue': 1e-8,
                 'auto_rank_method': 'participation_sqrt', 'min_rank': 1,
                 'max_rank_ratio': 1.0, 'rank_scale': 1.0, 'filter_type': 'none',
                 'init_mode': 'sum', 'per_layer_ranks': {}, '_NeedsDAware': type('_NeedsDAware', (Exception,), {})}
    exec(compile(module, str(source_path), 'exec'), namespace)
    source = json.loads(Path(plan['source_fit']).read_text())
    base = torch.load(source['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    experts = [torch.load(path, map_location='cpu', weights_only=True, mmap=True) for path in source['experts'].values()]
    built_path = Path(plan['outdir'])/'build.json'
    built = json.loads(built_path.read_text())
    checkpoint = torch.load(built['checkpoints']['all_2d']['path'], map_location='cpu', weights_only=True, mmap=True)
    names = ['qformer.base_queries', 'qformer.output_embed.2.weight']
    names += [next(name for name in base if name.startswith(prefix) and name.endswith('in_proj_weight'))
              for prefix in ['tower.', 'qformer.']]
    records = {}
    with torch.no_grad():
        for name in names:
            vectors = torch.stack([state[name]-base[name] for state in experts])
            assert vectors.flatten(1).norm(dim=1).square().max() >= 1e-12
            local, info = spectral_delta(vectors)
            reference = namespace['_solve'](name, vectors)
            error = float((local-reference).abs().max())
            relative = float((local-reference).double().norm()/reference.double().norm().clamp_min(1e-30))
            assert info['rank'] == namespace['per_layer_ranks'][name]
            assert error < 1e-6 and relative < 1e-4
            assert torch.equal(base[name]+local, checkpoint[name])
            records[name] = {'shape': list(vectors.shape), 'rank': info['rank'],
                             'max_abs_delta_error': error, 'relative_delta_error': relative,
                             'exported_state_recomputed_exact': True}
    result = {'unix': time.time(), 'code_sha256': sha256(Path(__file__)),
              'plan_sha256': sha256(plan_path), 'build_sha256': sha256(built_path),
              'reference_git_blob_exact': plan['reference_functional_blob'],
              'reference_bytes_sha256': hashlib.sha256(raw).hexdigest(), 'checks': records,
              'seconds': time.time()-started,
              'scope': 'Four actual LEAF matrices covering learned queries, both attention families and the largest output matrix. No performance/data inference or complete external-library reproduction. The separate build also checks analytic one-expert/disjoint-support/tiny-update cases.'}
    with (root/'analysis/swudi_reference_verification.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
