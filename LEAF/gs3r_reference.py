"""Adapter for the pinned, optimization-free official GS3R implementation.

The original source and licenses are retained in runs/gs3r_reference_v1/official.
Only the FusionBench typing dependency and two hardcoded CUDA transfers are
adapted. Matrix selection, clustering, restoration, and nonmatrix averaging
are the published implementation, not the historical-sketch candidate.
"""
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import torch


ROOT = Path(__file__).resolve().parent / 'runs/gs3r_reference_v1'
SOURCE = ROOT / 'official/fusion_bench/method/gs3r/utils/gs3r_utils.py'
HELPERS = {
    'canon_svd', 'change_sign', 'choose_R', 'save_symmetric',
    'restore_symmetric', '_kmeans_gpu', 'separate_tasks', 'gs3r_merge',
}


def load_official_core():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    expected = manifest['files']['fusion_bench/method/gs3r/utils/gs3r_utils.py']['local_sha256']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == expected
    tree = ast.parse(SOURCE.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {n.name for n in functions} == HELPERS

    class DevicePort(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 'cuda'):
                node.args[0] = ast.Name(id='accelerator', ctx=ast.Load())
                self.count += 1
            return node

    porter = DevicePort()
    module = ast.fix_missing_locations(porter.visit(ast.Module(body=functions, type_ignores=[])))
    assert porter.count == 2
    namespace = {'torch': torch, 'List': List, 'Optional': Optional, 'StateDictType': dict}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace


class GS3RReference:
    """Keep only the current task vector and its official restoration state."""

    def __init__(self, base, active, device='cpu'):
        self.base = base
        self.active = tuple(active)
        self.device = torch.device(device)
        self.core = load_official_core()
        self.vector = None
        self.restoration = {}
        self.count = 0

    @torch.no_grad()
    def arrive(self, expert):
        incoming = {n: expert[n].detach().cpu().float() - self.base[n].detach().cpu().float()
                    for n in self.active}
        if self.vector is None:
            self.vector = incoming
        else:
            self.vector, self.restoration = self.core['gs3r_merge'](
                [self.vector, incoming], exclude_keys=[], accelerator=self.device,
                procrustes=True, idx=self.count, singular_values=self.restoration)
        self.count += 1
        return {'count': self.count,
                'restoration_bytes': sum(x.numel() * x.element_size()
                                         for pair in self.restoration.values() for x in pair),
                'restored_matrix_count': len(self.restoration)}

    @torch.no_grad()
    def export(self):
        assert self.vector is not None
        return {n: value.detach().cpu().clone() if n not in self.vector
                else value.detach().cpu().float() + self.vector[n].cpu()
                for n, value in self.base.items()}
