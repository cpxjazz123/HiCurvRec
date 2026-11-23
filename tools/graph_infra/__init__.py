"""Shared, model-independent graph infrastructure (issue #245).

Layout and rationale
--------------------
``tools/research/`` is reserved for the Stage2/Stage3 launcher (CLAUDE.md §94), and the
experiment directories under ``results/research/issueNNN/`` must not each grow their own
copy of the same geometry or graph code.  This package is the one public home for

    manifold        kappa-stereographic exp/log/projection/distance (Euclidean, hyperbolic,
                    spherical) with a validated reference
    kappa_gcn       directed kappa-GCN layer
    topology        deterministic and budgeted induced ego subgraphs
    pairs           typed pair banks (direct, shared successor/predecessor, two-hop,
                    degree-matched) with per-item verification
    metrics         Spearman with ties, Pearson, Jaccard, prefix sharing, distortion
    graph_distance  sparse BFS over the visible graph, full versus subgraph distances
    assets          asset manifest, freeze/verify, and the strict join interface

Nothing here depends on a model checkpoint, a curvature weight or a training run.
"""
from __future__ import annotations

__all__ = ["manifold", "kappa_gcn", "topology", "pairs", "metrics", "graph_distance", "assets"]
