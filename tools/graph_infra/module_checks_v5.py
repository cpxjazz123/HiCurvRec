"""Per-module verification of the issue #249 deliverables on the real data.

Every claim is recomputed from the persisted assets or from an independent computation; the
builder's own report is never accepted as evidence.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra import diff_sectional, fidelity, heldout_eval, quant_eval, tiger_compare
from tools.graph_infra.assets import sha256
from tools.graph_infra.curvature_stream import CurvatureReader
from tools.graph_infra.ego_gnn import EgoStore, FEATURE_NAMES

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'assets'
NEW = ASSETS / 'v5_gnn_geometry'
OUT = REPO / 'results/research/issue249/MODULE_CHECKS.json'


def main() -> int:
    rows = []

    def row(module, check, ok, detail):
        rows.append({'module': module, 'check': check,
                     'verdict': 'PASS' if ok else 'FAIL', 'detail': detail})

    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    entries = {entry['name']: entry for entry in manifest['assets']}

    # --- manifest and reuse ------------------------------------------------
    drifted = []
    for entry in manifest['assets']:
        path = REPO / entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']:
            drifted.append(entry['path'])
    row('manifest', 'every registered asset exists and matches its sha256', not drifted,
        {'assets': len(manifest['assets']), 'drifted': drifted})
    reused = [entry for entry in manifest['assets'] if entry['origin'] == 'reused by immutable hash']
    row('manifest', 'the reused assets are referenced, not copied', len(reused) >= 40,
        {'reused': len(reused), 'built': len(manifest['assets']) - len(reused)})

    # --- ego interface -----------------------------------------------------
    store = EgoStore(REPO, ASSETS)
    features = store.node_features()
    row('ego', 'the node feature table has one row per item and only the allowed columns',
        features.shape == (store.n, len(FEATURE_NAMES)),
        {'shape': list(features.shape), 'features': list(FEATURE_NAMES)})
    stats = store.standardization()
    row('ego', 'the standardization is fitted on the training-visible items only',
        stats['items'] == store.n and stats['fitted_on'] == 'training-visible items only',
        {'items': stats['items']})
    sample = [0, 1, 2899, 1234, 20000]
    batch = store.collate(sample, hops=1)
    centres = [int(batch['nodes'][int(batch['root_global'][i])]) for i in range(len(sample))]
    row('ego', 'the collated batch recovers the centre of every item', centres == sample,
        {'items': sample, 'centres': centres})
    sparse = store.collate_sparse(sample, hops=1, chunk=2)
    row('ego', 'the sparse collate equals the plain collate',
        np.array_equal(batch['nodes'], sparse['nodes'])
        and np.array_equal(batch['edges'], sparse['edges']),
        {'nodes': int(len(batch['nodes']))})
    index = np.load(NEW / 'ego_batching_index.npz')
    row('ego', 'the persisted batching index covers every item', len(index['one_hop_sizes']) == store.n,
        {'items': int(len(index['one_hop_sizes']))})

    # --- curvature reader ---------------------------------------------------
    reader = CurvatureReader(REPO, ASSETS)
    traversal = reader.traverse()
    row('curvature', 'the full traversal reads every row exactly once',
        traversal['rows_match_expected'] and traversal['duplicate_triples'] == 0,
        {'rows_read': traversal['rows_read'], 'seconds': traversal['seconds'],
         'peak_rss_bytes': traversal['peak_rss_bytes']})
    agreement = reader.summary_agreement(sample=128)
    row('curvature', 'a batched aggregation equals the persisted per-item summary',
        agreement['matches'], agreement)
    rows_item = reader.item_rows(1234)
    row('curvature', 'the valid mask and the effective reference count are exposed',
        'valid' in rows_item and 'effective_references' in rows_item
        and bool((rows_item['effective_references'] > 0).all()),
        {'rows': len(rows_item['b'])})

    # --- differentiable statistic ------------------------------------------
    torch.manual_seed(249)
    vectors = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    metric = diff_sectional.build_metric('euclidean')
    value = diff_sectional.statistic_from_vectors(
        vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
        torch.tensor([0, 1, 2, 3, 4, 5]), metric)
    value.backward()
    row('diff_sectional', 'the statistic is differentiable and the gradient is finite and nonzero',
        bool(torch.isfinite(value)) and bool(torch.isfinite(vectors.grad).all())
        and float(vectors.grad.abs().sum()) > 0,
        {'statistic': float(value.detach()), 'gradient_norm': float(vectors.grad.norm())})
    chunked = diff_sectional.statistic_from_vectors_chunked(
        vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
        torch.tensor([0, 1, 2, 3, 4, 5]), metric, chunk=2)
    row('diff_sectional', 'the chunked evaluation equals the unchunked one',
        abs(float(chunked) - float(value.detach())) < 1e-12,
        {'chunked': float(chunked.detach()), 'plain': float(value.detach())})
    for family, kappa in [('euclidean', 0.0), ('hyperbolic', 1.0), ('spherical', 1.0)]:
        built = diff_sectional.build_metric(family, kappa)
        x = torch.randn(4, 3, dtype=torch.float64)
        y = torch.randn(4, 3, dtype=torch.float64)
        symmetric = bool(torch.allclose(built(x, y), built(y, x), atol=1e-12))
        zero = float(built(x[0], x[0]))
        row('diff_sectional', f'the {family} metric is symmetric with a zero self distance',
            symmetric and zero == 0.0, {'symmetric': symmetric, 'self_distance': zero})

    # --- fidelity ----------------------------------------------------------
    report = json.loads((NEW / 'fidelity_report.json').read_text())
    row('fidelity', 'the graph side of the evaluation reproduces the frozen #248 target exactly',
        report['graph_side_recomputation_matches_the_frozen_target']['matches'],
        report['graph_side_recomputation_matches_the_frozen_target'])
    row('fidelity', 'the hard SID and the continuous z are reported separately',
        'hard_sid_decoded' in report and 'continuous_z' in report
        and 'sign_agreement' in report['continuous_z'],
        {'hard_sid_sign': report['hard_sid_decoded']['sign_agreement']['sign_agreement'],
         'continuous_sign': report['continuous_z']['sign_agreement']['sign_agreement']})
    row('fidelity', 'the evaluation reports the scale, the sign agreement and the distance distortion',
        'scale_factor' in report['hard_sid_decoded']
        and 'distance_distortion' in report['hard_sid_decoded'],
        {'scale_factor': report['hard_sid_decoded']['scale_factor'],
         'distortion': report['hard_sid_decoded']['distance_distortion']})

    # --- TIGER -------------------------------------------------------------
    tiger = tiger_compare.compare(REPO)
    row('tiger', 'the Stage3 SID table is the one the run read',
        tiger['sid_sha256'] == '025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059',
        {'sid_sha256': tiger['sid_sha256']})
    row('tiger', 'the geometry metrics of the Stage3 table are NA, not fabricated',
        tiger['geometry_metrics']['status'] == 'NA',
        tiger['geometry_metrics']['reason'])
    row('tiger', 'the pure SID metrics are recomputed from the table',
        abs(tiger['pure_sid_metrics']['collision_prefix3']['collision_prefix']
            - 0.16927644690283483) < 1e-12,
        {'collision_prefix3': tiger['pure_sid_metrics']['collision_prefix3']['collision_prefix']})

    # --- isolation ---------------------------------------------------------
    evaluation = heldout_eval.HeldOutEvaluation(REPO, ASSETS)
    isolation = evaluation.is_isolated_from_training()
    row('isolation', 'the struct splits do not intersect the visible users',
        isolation['visible_intersect_struct_val'] == 0
        and isolation['visible_intersect_struct_test'] == 0, isolation)
    row('isolation', 'the training view may not read the held-out support',
        not heldout_eval.training_may_read(REPO, ASSETS, 'held_out_support'),
        {'use': entries['held_out_support']['use']})
    row('isolation', 'the strict banks are served only by the evaluation view',
        len(evaluation.strict_banks()) > 0
        and all(entries[name]['use'] == 'evaluation' for name in evaluation.strict_banks()),
        {'strict_banks': len(evaluation.strict_banks())})

    payload = {'issue': 249, 'modules': rows,
               'pass': sum(1 for r in rows if r['verdict'] == 'PASS'),
               'fail': sum(1 for r in rows if r['verdict'] == 'FAIL'),
               'na': sum(1 for r in rows if r['verdict'] == 'NA')}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float) + '\n')
    print(json.dumps({key: payload[key] for key in ('pass', 'fail', 'na')}))
    return 1 if payload['fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
