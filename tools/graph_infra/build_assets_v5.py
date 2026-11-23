"""Build and register the issue #249 assets (issue #249, step 8).

Everything genuinely new is written into ``assets/v5_gnn_geometry/``; the existing #246/#247/#248
assets are referenced by immutable hash and never copied.

The builder is idempotent: it verifies the reused assets first, writes only the new files, and
registers every entry in the root ``assets/manifest.json``.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra.assets import sha256
from tools.graph_infra.curvature_stream import CurvatureReader
from tools.graph_infra.ego_gnn import EgoStore, write_batching_index, FEATURE_NAMES
from tools.graph_infra import fidelity, tiger_compare

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'assets'
NEW = ASSETS / 'v5_gnn_geometry'
COMMAND = ('/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m '
           'tools.graph_infra.build_assets_v5')


def dump(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(payload, indent=2, default=convert, allow_nan=False) + '\n')


def verify_reused() -> dict:
    """Every reused asset must still match its hash in the root manifest."""
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    checked = 0
    for entry in manifest['assets']:
        path = REPO / entry['path']
        if not path.is_file():
            raise FileNotFoundError(f'reused asset missing: {entry["path"]}')
        if sha256(path) != entry['sha256']:
            raise ValueError(f'reused asset drifted: {entry["path"]}')
        checked += 1
    return {'verified_assets': checked, 'manifest_version': manifest['version']}


def build_ego_assets(store: EgoStore) -> dict:
    """The batching index, the constant node features and the standardization statistics."""
    NEW.mkdir(parents=True, exist_ok=True)
    report = write_batching_index(store, NEW / 'ego_batching_index.npz')
    features = store.node_features()
    np.savez_compressed(NEW / 'node_features.npz', features=features,
                        feature_names=np.array(FEATURE_NAMES))
    summary = {'feature_names': list(FEATURE_NAMES), 'items': int(store.n),
               'standardization': store.standardization(),
               'forbidden_inputs': ['graph structure labels (family / orbit / graphlet class)',
                                    'validation and test edges',
                                    'the 768-dim semantic embedding as a mandatory input'],
               'source': 'training-visible graph only'}
    dump(NEW / 'node_features_report.json', summary)
    return {'batching_index': report, 'node_features': summary}


def persist_fidelity() -> dict:
    """Copy the persisted fidelity report into the asset tree and register it."""
    source = REPO / 'results/research/issue249/FIDELITY.json'
    if not source.is_file():
        raise FileNotFoundError('the fidelity report has not been produced yet')
    target = NEW / 'fidelity_report.json'
    target.write_text(source.read_text())
    return {'path': str(target.relative_to(REPO)), 'bytes': target.stat().st_size}


def build_evaluation_assets() -> dict:
    """The curvature coverage record and the frozen TIGER comparison, both recomputed."""
    reader = CurvatureReader(REPO, ASSETS)
    traversal = reader.traverse()
    agreement = reader.summary_agreement(sample=256)
    tiger = tiger_compare.compare(REPO)
    evaluation = {
        'curvature_traversal': traversal,
        'curvature_summary_agreement': agreement,
        'curvature_reference_set': 'the centre component, as frozen in #248',
        'fidelity_scales': {
            'graph_side': 'hop^2 (unweighted shortest path on the undirected projection)',
            'quantized_side': 'the metric family\'s own length^2',
            'alignment': 's = sum(kappa_q * kappa_g) / sum(kappa_q^2), a single global factor',
            'headline': 'sign agreement (scale free)',
        },
        'tiger': tiger,
    }
    dump(NEW / 'evaluation_report.json', evaluation)
    return evaluation


def register(entries: list[dict]) -> dict:
    """Add the new assets to the root manifest and keep the old entries untouched."""
    path = ASSETS / 'manifest.json'
    manifest = json.loads(path.read_text())
    existing = {entry['name']: entry for entry in manifest['assets']}
    for entry in entries:
        existing[entry['name']] = entry
    manifest['assets'] = [existing[name] for name in sorted(existing)]
    manifest['version'] = 'v5-gnn-geometry-249'
    manifest.setdefault('history', []).append(
        {'version': 'v4-full-sectional-248',
         'note': 'the #248 curvature table, kept; the v5 entries below are new and additive'})
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return {'assets': len(manifest['assets']), 'version': manifest['version']}


def entry_for(path: Path, use: str, schema: str, rows: int, origin: str, **extra) -> dict:
    payload = {'name': path.stem, 'path': str(path.relative_to(REPO)), 'use': use,
               'schema': schema, 'rows': int(rows), 'bytes': path.stat().st_size,
               'sha256': sha256(path), 'origin': origin, 'command': COMMAND}
    payload.update(extra)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    reused = verify_reused()
    if not args.verify:
        store = EgoStore(REPO, ASSETS)
        built = build_ego_assets(store)
        evaluation = build_evaluation_assets()
        fidelity_entry = persist_fidelity()
        entries = [
            entry_for(NEW / 'ego_batching_index.npz', 'training', 'npz',
                      int(store.n), 'built by issue #249',
                      content='per-item one-hop ego sizes; the node lists are recomputed '
                              'deterministically from the visible CSR'),
            entry_for(NEW / 'node_features.npz', 'training', 'npz',
                      int(store.n), 'built by issue #249',
                      content='constant per-node features: degrees, reciprocal degree, logarithms, '
                              'isolated flag',
                      feature_names=list(FEATURE_NAMES)),
            entry_for(NEW / 'node_features_report.json', 'training', 'json',
                      int(store.n), 'built by issue #249',
                      content='standardization statistics fitted on the training-visible items only'),
            entry_for(NEW / 'evaluation_report.json', 'description', 'json',
                      1, 'built by issue #249',
                      content='curvature traversal coverage, summary agreement and the frozen TIGER '
                              'comparison (geometry metrics NA)'),
            entry_for(NEW / 'fidelity_report.json', 'description', 'json',
                      int(json.loads((NEW / 'fidelity_report.json').read_text())['triples_evaluated']),
                      'built by issue #249',
                      content='quantized geometry fidelity against the frozen graph curvature: '
                              'per-triple error, sign agreement, per-item view and distance '
                              'distortion; the hard SID and the continuous z are reported apart'),
        ]
        registered = register(entries)
    else:
        registered = {'assets': reused['verified_assets'], 'version': reused['manifest_version']}
    report = {'reused': reused, 'registered': registered,
              'seconds': time.monotonic() - start}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
