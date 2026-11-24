"""Actual learned-token consumer must reject parseable static-layout corruption."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
SPEC = importlib.util.spec_from_file_location('structure252_integrity', ROOT /
    'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/model/structure.py')
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


class RuntimeIntegrityTests(unittest.TestCase):
    def test_parseable_operator_and_ego_mutations_fail_before_any_array_load(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252') as temporary:
            repo = Path(temporary)
            root = repo / 'assets'
            root.mkdir()
            entries = []
            names = ['ego_full_1hop_visible', 'two_hop_nodes', 'two_hop_offsets',
                     'node_features', 'node_features_report', 'ego_operator_metadata']
            names += ['ego%d_%s_%s' % (hops, direction, part)
                      for hops in (1, 2) for direction in ('in', 'out') for part in ('ptr', 'col')]
            for name in names:
                path = root / (name + '.npy')
                np.save(path, np.array([1, 2, 3], dtype=np.int32))
                entries.append({'name': name, 'path': str(path.relative_to(repo)),
                                'use': 'training', 'bytes': path.stat().st_size,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            # The registered metadata name is not a loader implementation detail.
            metadata = next(e for e in entries if e['name'] == 'ego_operator_metadata')
            actual_metadata = root / 'v9_ego_operators/metadata.json'
            actual_metadata.parent.mkdir()
            (repo / metadata['path']).rename(actual_metadata)
            metadata['path'] = str(actual_metadata.relative_to(repo))
            metadata['use'] = 'description'
            (root / 'manifest.json').write_text(json.dumps({'assets': entries}))
            store = type('Store', (), {})()
            store.repo, store.root, store.n = repo, root, 3
            for entry in entries:
                with self.subTest(asset=entry['name']):
                    path = repo / entry['path']
                    original = path.read_bytes()
                    changed = bytearray(original)
                    changed[-1] ^= 1
                    path.write_bytes(changed)
                    try:
                        with patch.object(structure.np, 'load', side_effect=AssertionError('loaded corrupted asset')):
                            with self.assertRaisesRegex(ValueError, 'asset.*(changed|identity)|changed.*asset'):
                                structure.SubgraphTokens()._cached_store(store)
                    finally:
                        path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
