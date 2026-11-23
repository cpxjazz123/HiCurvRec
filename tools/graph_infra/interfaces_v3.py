"""Explicitly separated visible training features and held-out evaluation targets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools.graph_infra.assets import load_sparse, sha256
from tools.graph_infra.topology import ego_full


def validate_ids(ids, expected):
    ids = np.asarray(ids)
    if ids.dtype.kind not in 'iu' or ids.ndim != 1 or not np.array_equal(ids, expected) or len(np.unique(ids)) != len(ids):
        raise ValueError('ID alignment mismatch')


def verify(manifest, repo):
    manifest, repo = Path(manifest), Path(repo)
    data = json.loads(manifest.read_text())
    for entry in data['assets']:
        path = (repo / entry['path']).resolve()
        if not path.is_relative_to(repo.resolve()) or sha256(path) != entry['sha256']:
            raise ValueError('manifest hash/path mismatch: ' + entry['name'])
    return data


class TrainingAssets:
    """No test-target or full-description loading method exists on this interface."""
    def __init__(self, repo, root=None):
        self.repo = Path(repo)
        self.root = Path(root or self.repo / 'results/research/issue247/assets')
        manifest = verify(self.root / 'manifest.json', self.repo)
        self.entries = {e['name']:e for e in manifest['assets']}
        self.ids = np.load(self.root / 'item_ids.npy')
        validate_ids(self.ids, np.arange(len(self.ids)))
        self.forward = self.load('visible_forward')
        self.reverse = self.forward.T.tocsr()

    def load(self, name):
        e = self.entries[name]
        if e['use'] != 'training':
            raise ValueError('training cannot read evaluation or description assets')
        p = self.repo / e['path']
        if e['schema'] == 'csr': return load_sparse(p)
        if p.suffix == '.npy': return np.load(p, mmap_mode='r')
        if p.suffix == '.npz': return dict(np.load(p))
        return json.loads(p.read_text())

    def aligned(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(self.ids)):
            raise ValueError('invalid item indices')
        curvature = self.load('sectional_summary')
        topology = self.load('topology_features')
        return {'item_ids':self.ids[indices], 'semantic':self.load('semantic')[indices],
                'curvature':{k:v[indices] for k,v in curvature.items()},
                'topology':{k:v[indices] for k,v in topology.items() if v.ndim and len(v)==len(self.ids)}}

    def ego(self, item, radius):
        if radius not in (1,2) or not 0 <= item < len(self.ids):
            raise ValueError('invalid ego request')
        block = ego_full(item, self.forward, self.reverse, radius)
        block['node_mask'] = np.ones(len(block['nodes']),dtype=bool)
        block['edge_mask'] = np.ones(len(block['edges']),dtype=bool)
        return block


class HeldOutEvaluation:
    def __init__(self, repo, root=None):
        self.repo = Path(repo)
        root = Path(root or self.repo / 'results/research/issue247/assets')
        self.entries = {e['name']:e for e in verify(root/'manifest.json',self.repo)['assets']}

    def load(self, name):
        e = self.entries[name]
        if e['use'] != 'evaluation' or not name.startswith('strict_'):
            raise ValueError('strict evaluation targets required')
        return dict(np.load(self.repo/e['path']))
