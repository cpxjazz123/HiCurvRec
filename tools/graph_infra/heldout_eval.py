"""The strict held-out evaluation interface (issue #249, step 6 and CONTRACT.md §8).

The training side may not read the held-out structure targets. This module serves them to an
evaluator only, and refuses to hand them out through the training view.

It never trains a probe and never reads the validation/test edges of the training graph.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import load_sparse, sha256

#: The banks that count as strict held-out structure targets.
STRICT_PREFIX = 'strict_'
RELAXED_PREFIX = 'relaxed_'


class HeldOutEvaluation:
    """Serves the strict held-out banks and nothing else."""

    def __init__(self, repo: Path, assets_root: Path | None = None):
        self.repo = Path(repo)
        self.root = Path(assets_root or self.repo / 'assets')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.entries = {entry['name']: entry for entry in self.manifest['assets']}

    def strict_banks(self) -> list[str]:
        return sorted(name for name in self.entries
                      if name.startswith(STRICT_PREFIX) and self.entries[name]['use'] == 'evaluation')

    def load_strict(self, name: str) -> dict:
        if not name.startswith(STRICT_PREFIX):
            raise ValueError(f'{name} is not a strict held-out bank')
        entry = self.entries[name]
        if entry['use'] != 'evaluation':
            raise ValueError(f'{name} is not declared as an evaluation asset')
        blob = np.load(self.repo / entry['path'])
        return {key: blob[key] for key in blob.files}

    def strict_pairs(self) -> dict:
        """Every strict pair, keyed by bank, with the bank's own sha256."""
        report = {}
        for name in self.strict_banks():
            blob = self.load_strict(name)
            report[name] = {'pairs': int(len(blob['pairs'])),
                            'sha256': self.entries[name]['sha256'],
                            'pair_sha256': sha256(self.repo / self.entries[name]['path'])}
        return report

    def held_out_support(self):
        entry = self.entries['held_out_support']
        if entry['use'] != 'evaluation':
            raise ValueError('held_out_support is not an evaluation asset')
        return load_sparse(self.repo / entry['path'])

    def struct_split_users(self) -> dict:
        entry = self.entries['split_users']
        if entry['use'] != 'evaluation':
            raise ValueError('split_users is not an evaluation asset')
        blob = np.load(self.repo / entry['path'])
        return {key: blob[key] for key in blob.files}

    def is_isolated_from_training(self) -> dict:
        """The struct-val and struct-test users must not appear in the visible user set."""
        splits = self.struct_split_users()
        visible = set(np.asarray(splits['visible']).tolist())
        struct_val = set(np.asarray(splits['struct_val']).tolist())
        struct_test = set(np.asarray(splits['struct_test']).tolist())
        return {'visible_users': len(visible),
                'struct_val_users': len(struct_val),
                'struct_test_users': len(struct_test),
                'visible_intersect_struct_val': len(visible & struct_val),
                'visible_intersect_struct_test': len(visible & struct_test),
                'struct_val_intersect_struct_test': len(struct_val & struct_test)}


def training_may_read(repo: Path, assets_root: Path | None = None, name: str = 'held_out_support') -> bool:
    """Whether the training view is allowed to read an asset: only `training` assets."""
    root = Path(assets_root or Path(repo) / 'assets')
    manifest = json.loads((root / 'manifest.json').read_text())
    for entry in manifest['assets']:
        if entry['name'] == name:
            return entry['use'] == 'training'
    raise KeyError(name)
