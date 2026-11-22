"""One DIRECTED behaviour graph, verified rather than assumed (Issue #221).

The undirected graph of #218/#220 is kept as a historical reference; it is no
longer the canonical structure.  Everything below verifies the directed copy and
refuses a folder whose metadata still claims an undirected single graph.

Owner decision (2026-11-19): triangles, open wedges, communities/bridges and AF3
are defined on a single undirected graph under ``results/research/behavior_graph``;
the ordered sequences stay in ``train.parquet`` and no experiment keeps its own
copy.  This module is the machine half of that rule:

* :func:`verify_graph` recomputes the sha256 of the stored files and compares them
  with ``metadata.json`` -- a rebuilt or hand-edited graph is refused, and so is a
  metadata file whose policy block does not say "one undirected graph, no directed
  copy";
* :func:`inspect_tree` reads a candidate tree's source: a tree that builds its own
  graph (reads ``train.parquet`` and constructs adjacency) is refused, a tree that
  reads the canonical ``graph.npz`` passes only if the stored graph verifies, and a
  tree with no graph use is ``NOT_APPLICABLE`` rather than a failure.

    python3 -m tools.research.stage2.graph_contract            # verify the stored graph
    python3 -m tools.research.stage2.graph_contract <tree>     # plus the tree scan
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA_PREFIX = "genrec.behavior_graph"
SCHEMA_DIRECTED = "genrec.behavior_graph.directed.v1"
CANONICAL = Path("results") / "research" / "behavior_graph_directed"
HISTORICAL = Path("results") / "research" / "behavior_graph"
BUILD_TOKENS = re.compile(r"adjacency|coo_matrix|csr_matrix|csgraph|neighbou?rs?\b|edge_index|torch_geometric|networkx")


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_graph(folder: Path | str | None = None) -> dict:
    """The stored graph is exactly the one metadata.json describes."""
    folder = Path(folder) if folder is not None else repo_root() / CANONICAL
    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        return {"status": "MISSING", "graph": str(folder),
                "detail": "no metadata.json; the graph has not been built"}
    try:
        metadata = json.loads(metadata_path.read_text())
    except ValueError as exc:
        return {"status": "FAIL", "graph": str(folder), "detail": "unreadable metadata: " + repr(exc)}
    schema = str(metadata.get("schema", ""))
    if not schema.startswith(SCHEMA_PREFIX):
        return {"status": "FAIL", "graph": str(folder),
                "detail": "schema %r is not %s.*" % (schema, SCHEMA_PREFIX)}
    policy = metadata.get("policy") or {}
    if schema != SCHEMA_DIRECTED or metadata.get("directed") is not True:
        return {"status": "FAIL", "graph": str(folder),
                "detail": "expected a directed graph (%s, directed=true); got schema %r, directed %r"
                          % (SCHEMA_DIRECTED, schema, metadata.get("directed"))}
    if "symmetry" not in policy:
        return {"status": "FAIL", "graph": str(folder),
                "detail": "the policy block must state the symmetry rule; got %r" % (policy,)}
    outputs = metadata.get("outputs") or {}
    if not outputs:
        return {"status": "FAIL", "graph": str(folder), "detail": "metadata lists no outputs to verify"}
    problems = []
    for name, record in outputs.items():
        path = folder / name
        if not path.is_file():
            problems.append(name + ": missing")
            continue
        digest = _sha256(path)
        if digest != record.get("sha256"):
            problems.append("%s: sha256 %s != metadata %s" % (name, digest[:16], str(record.get("sha256"))[:16]))
        elif path.stat().st_size != record.get("bytes"):
            problems.append("%s: size %d != metadata %s" % (name, path.stat().st_size, record.get("bytes")))
    if problems:
        return {"status": "FAIL", "graph": str(folder), "detail": "; ".join(problems)}
    return {"status": "PASS", "graph": str(folder),
            "detail": "%d files match metadata.json" % len(outputs)}


def verify_historical(folder: Path | str | None = None) -> dict:
    """The undirected graph of #218/#220: still readable, explicitly historical."""
    folder = Path(folder) if folder is not None else repo_root() / HISTORICAL
    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        return {"status": "MISSING", "graph": str(folder), "detail": "no historical metadata.json"}
    metadata = json.loads(metadata_path.read_text())
    policy = metadata.get("policy") or {}
    if policy.get("single_undirected_graph") is not True:
        return {"status": "FAIL", "graph": str(folder),
                "detail": "the historical folder does not declare the undirected rule"}
    problems = [name for name, record in (metadata.get("outputs") or {}).items()
                if not (folder / name).is_file() or _sha256(folder / name) != record.get("sha256")]
    if problems:
        return {"status": "FAIL", "graph": str(folder), "detail": "; ".join(problems)}
    return {"status": "PASS", "graph": str(folder), "historical": True,
            "detail": "undirected reference verified; not the canonical graph"}


def _sources(tree: Path):
    for path in tree.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def inspect_tree(tree: Path | str, folder: Path | str | None = None) -> dict:
    """A candidate may read the canonical graph; it may not build its own."""
    tree = Path(tree)
    folder = Path(folder) if folder is not None else repo_root() / CANONICAL
    builds, reads, foreign = [], [], []
    for path in _sources(tree):
        text = path.read_text(errors="ignore")
        if "train.parquet" in text and BUILD_TOKENS.search(text):
            builds.append(path.relative_to(tree).as_posix())
        if "graph.npz" in text:
            mentions_canonical = "behavior_graph/graph.npz" in text or "behavior_graph\"" in text
            (reads if mentions_canonical else foreign).append(path.relative_to(tree).as_posix())
    if builds:
        return {"status": "FAIL", "tree": str(tree),
                "detail": "builds its own behaviour graph from train.parquet in " + ", ".join(sorted(builds)[:4])
                          + "; read the canonical " + str(CANONICAL / "graph.npz") + " instead",
                "builds": builds}
    if foreign:
        return {"status": "FAIL", "tree": str(tree),
                "detail": "reads a graph.npz that is not the canonical one in " + ", ".join(sorted(foreign)[:4]),
                "foreign": foreign}
    if not reads:
        return {"status": "NOT_APPLICABLE", "tree": str(tree),
                "detail": "the tree does not use a behaviour graph"}
    stored = verify_graph(folder)
    if stored["status"] != "PASS":
        return {"status": "FAIL", "tree": str(tree),
                "detail": "reads the canonical graph but the stored graph does not verify: " + stored["detail"],
                "graph": stored}
    return {"status": "PASS", "tree": str(tree),
            "detail": "reads the canonical graph; " + stored["detail"], "graph": stored}


def main(argv: list) -> int:
    verdict = {"graph": verify_graph()}
    if len(argv) > 1:
        verdict["tree"] = inspect_tree(argv[1])
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
