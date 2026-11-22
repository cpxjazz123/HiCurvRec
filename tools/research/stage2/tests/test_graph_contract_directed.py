"""Issue #221: the canonical graph is directed; the undirected copy is historical."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.research.stage2 import graph_contract


REPO = Path(__file__).resolve().parents[4]
DIRECTED = REPO / "results" / "research" / "behavior_graph_directed"
UNDIRECTED = REPO / "results" / "research" / "behavior_graph"


def test_the_canonical_graph_is_the_directed_one():
    assert graph_contract.CANONICAL.name == "behavior_graph_directed"
    verdict = graph_contract.verify_graph()
    assert verdict["status"] == "PASS", verdict


def test_a_folder_that_claims_undirected_is_refused(tmp_path):
    copy = tmp_path / "graph"
    shutil.copytree(DIRECTED, copy)
    metadata = json.loads((copy / "metadata.json").read_text())
    metadata["directed"] = False
    (copy / "metadata.json").write_text(json.dumps(metadata))
    verdict = graph_contract.verify_graph(copy)
    assert verdict["status"] == "FAIL"
    assert "directed" in verdict["detail"]


def test_a_tampered_graph_file_is_refused(tmp_path):
    copy = tmp_path / "graph"
    shutil.copytree(DIRECTED, copy)
    payload = (copy / "graph.npz").read_bytes()
    (copy / "graph.npz").write_bytes(payload[:-1] + bytes([payload[-1] ^ 0x01]))
    verdict = graph_contract.verify_graph(copy)
    assert verdict["status"] == "FAIL"
    assert "sha256" in verdict["detail"]


def test_the_undirected_graph_still_verifies_as_history():
    verdict = graph_contract.verify_historical()
    assert verdict["status"] == "PASS", verdict
    assert verdict.get("historical") is True


def test_a_tree_that_builds_its_own_adjacency_is_refused(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "graph.py").write_text(
        "import pandas as pd\n"
        "frame = pd.read_parquet('results/stage0_build_parquet/train.parquet')\n"
        "adjacency = csr_matrix((data, (rows, cols)))\n")
    verdict = graph_contract.inspect_tree(tree)
    assert verdict["status"] == "FAIL"
    assert "builds its own" in verdict["detail"]
