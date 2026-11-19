"""One undirected behaviour graph: the check that enforces it.

The rule (owner, 2026-11-19): triangles, open wedges, communities/bridges and
AF3 are all defined on a single undirected graph stored in
``results/research/behavior_graph``, the ordered sequences stay in
``train.parquet``, and no experiment keeps its own copy.  The check has two
halves and both are proved here:

* **the stored graph is the one ``metadata.json`` describes** -- a rebuilt or
  hand-edited file whose sha256 no longer matches is refused;
* **a candidate tree does not build its own graph** -- a tree that reads
  ``train.parquet`` and constructs adjacency/CSR is refused, a tree that reads
  the canonical ``graph.npz`` passes, and a tree that touches no graph at all is
  not applicable rather than a failure.
"""
from __future__ import annotations

import hashlib
import json

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _graph_dir(tmp_path: Path, corrupt: bool = False, policy: bool = True) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    (folder / "edges.parquet").write_bytes(b"edge-bytes")
    (folder / "graph.npz").write_bytes(b"npz-bytes")
    (folder / "node_features.npy").write_bytes(b"feature-bytes")
    metadata = {
        "schema": "genrec.behavior_graph.v1",
        "policy": ({"single_undirected_graph": True, "directed_graph_stored": False}
                   if policy else {"single_undirected_graph": True}),
        "outputs": {},
    }
    for name in ("edges.parquet", "graph.npz", "node_features.npy"):
        digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        metadata["outputs"][name] = {"bytes": (folder / name).stat().st_size, "sha256": digest}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if corrupt:
        (folder / "graph.npz").write_bytes(b"edited-after-metadata")
    return folder


def test_matching_hashes_pass(tmp_path):
    from tools.research.stage2 import graph_contract
    verdict = graph_contract.verify_graph(_graph_dir(tmp_path))
    assert verdict["status"] == "PASS", verdict


def test_edited_file_is_refused(tmp_path):
    from tools.research.stage2 import graph_contract
    verdict = graph_contract.verify_graph(_graph_dir(tmp_path, corrupt=True))
    assert verdict["status"] == "FAIL"
    assert "sha256" in verdict["detail"]


def test_missing_policy_is_refused(tmp_path):
    from tools.research.stage2 import graph_contract
    verdict = graph_contract.verify_graph(_graph_dir(tmp_path, policy=False))
    assert verdict["status"] == "FAIL"
    assert "policy" in verdict["detail"]


def test_tree_that_builds_its_own_graph_is_refused(tmp_path):
    from tools.research.stage2 import graph_contract
    tree = tmp_path / "TIGER_own_graph"
    tree.mkdir()
    (tree / "structure.py").write_text(
        "import pandas as pd\nfrom scipy import sparse\n"
        "frame = pd.read_parquet('results/stage0_build_parquet/train.parquet')\n"
        "adjacency = sparse.coo_matrix((weights, (rows, cols)))\n")
    verdict = graph_contract.inspect_tree(tree, _graph_dir(tmp_path))
    assert verdict["status"] == "FAIL"
    assert "builds its own" in verdict["detail"]


def test_tree_that_reads_the_canonical_graph_passes(tmp_path):
    from tools.research.stage2 import graph_contract
    tree = tmp_path / "TIGER_reads_graph"
    tree.mkdir()
    (tree / "structure.py").write_text(
        "import numpy as np\n"
        "GRAPH = 'results/research/behavior_graph/graph.npz'\n"
        "payload = np.load(GRAPH)\n")
    verdict = graph_contract.inspect_tree(tree, _graph_dir(tmp_path))
    assert verdict["status"] == "PASS", verdict


def test_tree_without_graph_use_is_not_applicable(tmp_path):
    from tools.research.stage2 import graph_contract
    tree = tmp_path / "TIGER_plain"
    tree.mkdir()
    (tree / "structure.py").write_text("VALUE = 1\n")
    verdict = graph_contract.inspect_tree(tree, _graph_dir(tmp_path))
    assert verdict["status"] == "NOT_APPLICABLE"


def test_the_repository_graph_verifies():
    from tools.research.stage2 import graph_contract
    verdict = graph_contract.verify_graph(REPO / "results" / "research" / "behavior_graph")
    assert verdict["status"] in ("PASS", "MISSING"), verdict
