"""Training/evaluation separation for the v4 assets (issue #248).

The training side may only load assets whose declared use is ``training``. Held-out structure
targets are served by a separate class that refuses anything else. The v4 curvature assets are
derived from the training-visible graph only, so they carry the ``training`` use; the strict banks
and the held-out support keep the ``evaluation`` use.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import load_sparse, sha256


class ManifestError(ValueError):
    """Raised when the manifest, a hash or a declared use does not hold."""


def load_manifest(repo: Path, root: Path | None = None) -> dict:
    repo = Path(repo)
    root = Path(root or repo / 'results/research/issue248/assets')
    path = root / 'manifest.json'
    if not path.is_file():
        raise ManifestError(f'missing manifest: {path}')
    payload = json.loads(path.read_text())
    payload['root'] = root
    payload['by_name'] = {entry['name']: entry for entry in payload['assets']}
    return payload


def verify_manifest(payload: dict, repo: Path) -> dict:
    """Recompute every hash; raise on drift, on a missing file or on an escaped path."""
    checked = []
    repo = Path(repo).resolve()
    for entry in payload['assets']:
        path = (repo / entry['path']).resolve()
        if not path.is_relative_to(repo):
            raise ManifestError(f'path escapes the repository: {entry["path"]}')
        if not path.is_file():
            raise ManifestError(f'missing asset: {entry["path"]}')
        if sha256(path) != entry['sha256']:
            raise ManifestError(f'hash mismatch: {entry["name"]}')
        checked.append(entry['name'])
    return {'verified': len(checked), 'names': checked}


def assert_use(payload: dict, name: str, use: str) -> dict:
    entry = payload['by_name'].get(name)
    if entry is None:
        raise ManifestError(f'unknown asset: {name}')
    if entry['use'] != use:
        raise ManifestError(f'{name} is declared "{entry["use"]}", not "{use}"')
    return entry


class TrainingView:
    """Reads only assets declared for training."""

    def __init__(self, repo: Path, root: Path | None = None):
        self.repo = Path(repo)
        self.payload = load_manifest(self.repo, root)
        self.root = self.payload['root']

    def entry(self, name: str) -> dict:
        return assert_use(self.payload, name, 'training')

    def load(self, name: str):
        entry = self.entry(name)
        path = self.repo / entry['path']
        if path.suffix == '.npz':
            blob = np.load(path)
            return {key: blob[key] for key in blob.files}
        if path.suffix == '.npy':
            return np.load(path, mmap_mode='r')
        if path.suffix == '.npz' or entry.get('schema') == 'csr':
            return load_sparse(path)
        return json.loads(path.read_text())

    def curvature_table(self):
        """The full triple table, shard by shard; each shard's rows are grouped by centre."""
        from tools.graph_infra.full_assets import FullAssets
        return FullAssets(self.repo, self.root)

    def visible_graph(self):
        return load_sparse(self.repo / self.entry('visible_forward')['path'])

    def undirected_projection(self):
        return load_sparse(self.repo / self.entry('visible_undirected')['path'])


class EvaluationView:
    """Serves only the strict held-out structure targets."""

    def __init__(self, repo: Path, root: Path | None = None):
        self.repo = Path(repo)
        self.payload = load_manifest(self.repo, root)
        self.root = self.payload['root']

    def entry(self, name: str) -> dict:
        if not name.startswith('strict_'):
            raise ManifestError(f'{name} is not a strict evaluation bank')
        return assert_use(self.payload, name, 'evaluation')

    def load(self, name: str):
        entry = self.entry(name)
        blob = np.load(self.repo / entry['path'])
        return {key: blob[key] for key in blob.files}

    def held_out_support(self):
        entry = assert_use(self.payload, 'held_out_support', 'evaluation')
        return load_sparse(self.repo / entry['path'])

    def split_users(self):
        entry = assert_use(self.payload, 'split_users', 'evaluation')
        blob = np.load(self.repo / entry['path'])
        return {key: blob[key] for key in blob.files}
