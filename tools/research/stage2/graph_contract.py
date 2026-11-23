"""One DIRECTED behaviour graph, verified rather than assumed (Issue #221).

The undirected graph of #218/#220 is kept as a historical reference; it is no
longer the canonical structure.  Everything below verifies the directed copy and
refuses a folder whose metadata still claims an undirected single graph.

The default directed canonical contract remains in force for undeclared arms.
Issue252 additionally registers ``infra246-visible-v2``: its independently
reconstructed raw-train user split, graph values, training dependencies and audit
algorithm are pinned in ``assets/v7_graph_contract/contract.json``. Only an
explicit literal ``BEHAVIOR_GRAPH_NAME`` opts into this contract; it is not a
per-arm exception. Correct existing graph/target binaries are reused unchanged.

``verify_named_graph`` rehashes the receipt, sources, inputs and relevant manifest
leaves on every invocation. Added graph-derived training caches require exact
visible-graph identity and training-only input ancestry; they participate in the
returned cache identity. Unrelated nontraining manifest extensions do not.
``inspect_tree`` refuses candidate graph construction, unregistered/mixed graph
reads and evaluation inputs. A graph-free undeclared tree remains
``NOT_APPLICABLE``; an explicit but invalid declaration always fails.

    python3 -m tools.research.stage2.graph_contract            # verify the stored graph
    python3 -m tools.research.stage2.graph_contract <tree>     # plus the tree scan
"""
from __future__ import annotations

import ast
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
NAMED_GRAPH = "infra246-visible-v2"
NAMED_CONTRACT = Path("assets/v7_graph_contract/contract.json")
NAMED_CONTRACT_SHA256 = "d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687"


def verify_named_graph(name: str) -> dict:
    """Validate a pinned raw-data audit and each relevant manifest leaf (stdlib only)."""
    try:
        if name != NAMED_GRAPH:
            raise ValueError("unknown named graph: " + str(name))
        root = repo_root()
        path = root / NAMED_CONTRACT
        if _sha256(path) != NAMED_CONTRACT_SHA256:
            raise ValueError("named contract digest changed")
        contract = json.loads(path.read_text())
        if (contract["name"] != name or contract["schema"] != "genrec.named_graph.v1"
                or contract["audit_version"] != "issue252.raw-transitions.v1"):
            raise ValueError("named contract name/schema/audit-version drift")
        audit_path = root / contract["audit"]
        if _sha256(audit_path) != contract["audit_sha256"]:
            raise ValueError("audit receipt mutated")
        audit = json.loads(audit_path.read_text())
        if (audit["status"] != "PASS" or audit["schema"] != contract["audit_version"]
                or audit["bindings"] != contract["bindings"]
                or audit["manifest_leaves"] != contract["manifest_leaves"]):
            raise ValueError("audit receipt does not bind this registration")
        for relative, expected in contract["bindings"].items():
            bound = (root / relative).resolve()
            if root.resolve() not in bound.parents:
                raise ValueError("binding escapes repository: " + relative)
            if (bound.stat().st_size != expected["bytes"]
                    or _sha256(bound) != expected["sha256"]):
                raise ValueError("named graph input/source mutated: " + relative)
        manifest = json.loads((root / "assets/manifest.json").read_text())
        entries = {}
        for entry in manifest["assets"]:
            if entry["name"] in entries:
                raise ValueError("duplicate manifest asset: " + entry["name"])
            entries[entry["name"]] = entry
        for name, expected in contract["manifest_leaves"].items():
            entry = entries.get(name, {})
            if any(entry.get(key) != value for key, value in expected.items()):
                raise ValueError("manifest leaf identity/role changed: " + name)
        # New performance caches may extend the manifest, but cannot extend the
        # graph family or turn an evaluation bank into a training dependency.
        by_path = {entry["path"]: entry for entry in entries.values()}
        if len(by_path) != len(entries):
            raise ValueError("duplicate manifest asset path")
        extensions = {}
        metadata_paths = {str(NAMED_CONTRACT): NAMED_CONTRACT_SHA256,
                          contract["audit"]: contract["audit_sha256"]}

        def check_path(relative, digest):
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("non-repository cache binding: " + relative)
            actual = (root / relative_path).resolve()
            if root.resolve() not in actual.parents or _sha256(actual) != digest:
                raise ValueError("cache input/source identity changed: " + relative)

        for asset_name, entry in entries.items():
            if asset_name in contract["manifest_leaves"]:
                continue
            relative = entry["path"]
            if relative in metadata_paths:
                if entry.get("sha256") != metadata_paths[relative]:
                    raise ValueError("metadata registration digest changed")
                check_path(relative, metadata_paths[relative])
                continue
            if entry.get("use") != "training" and not any(
                    key in entry for key in ("graph_name", "graph_sha256")):
                continue
            if (entry.get("use") != "training" or entry.get("graph_name") != NAMED_GRAPH
                    or entry.get("graph_sha256") != contract["manifest_leaves"]["visible_forward"]["sha256"]):
                raise ValueError("unregistered training cache graph: " + asset_name)
            check_path(relative, entry["sha256"])
            if (root / relative).stat().st_size != entry.get("bytes"):
                raise ValueError("cache size changed: " + asset_name)
            for field in ("inputs", "source_hashes"):
                values = entry.get(field)
                if not isinstance(values, dict) or not values:
                    raise ValueError("cache lacks " + field + ": " + asset_name)
                for dependency, digest in values.items():
                    check_path(dependency, digest)
                    if field == "inputs":
                        source = by_path.get(dependency, {})
                        if source.get("use") != "training" or source.get("sha256") != digest:
                            raise ValueError("cache uses nontraining input: " + dependency)
                    elif Path(dependency).suffix != ".py":
                        raise ValueError("cache source binding is not Python: " + dependency)
            extensions[asset_name] = entry
        # A dependency cycle cannot establish visible-graph ancestry.
        visiting, visited = set(), set()

        def visit(asset_name):
            if asset_name in visiting:
                raise ValueError("cyclic training cache ancestry")
            if asset_name in visited or asset_name not in extensions:
                return
            visiting.add(asset_name)
            for relative in extensions[asset_name]["inputs"]:
                visit(by_path[relative]["name"])
            visiting.remove(asset_name)
            visited.add(asset_name)

        for asset_name in extensions:
            visit(asset_name)
        identity = NAMED_CONTRACT_SHA256
        if extensions:
            identity = hashlib.sha256(json.dumps(
                {"contract": identity, "training_extensions": extensions},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"status": "PASS", "name": NAMED_GRAPH,
                "identity": identity, "contract_sha256": NAMED_CONTRACT_SHA256,
                "detail": "raw train, split, graph, audit source and training dependencies verified"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "FAIL", "name": name, "detail": str(exc)}


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
    declarations = []
    for source in _sources(tree):
        try:
            parsed = ast.parse(source.read_text(errors="ignore"))
        except SyntaxError:
            continue  # The ordinary scanner below reports the exact syntax error.
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == "BEHAVIOR_GRAPH_NAME"
                       for target in targets):
                    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                        return {"status": "FAIL", "detail": "BEHAVIOR_GRAPH_NAME must be an explicit literal"}
                    declarations.append(node.value.value)
    if declarations:
        if set(declarations) != {NAMED_GRAPH}:
            return {"status": "FAIL", "detail": "unknown or conflicting BEHAVIOR_GRAPH_NAME"}
        return _inspect_named_tree(tree)
    builds, reads, foreign = [], [], []
    for path in _sources(tree):
        text = path.read_text(errors="ignore")
        try:
            parsed=ast.parse(text)
        except SyntaxError as exc:
            return {"status":"FAIL","tree":str(tree),
                    "detail":"cannot inspect invalid Python source %s: %s" %
                             (path.relative_to(tree).as_posix(),exc.msg)}
        for node in ast.walk(parsed):
            # EgoStore reads assets/train/visible_*.npz, not the canonical graph.
            # Imported loaders must not disappear behind NOT_APPLICABLE merely
            # because their implementation and graph filename live outside the tree.
            direct=(isinstance(node,ast.ImportFrom) and
                    node.module=="tools.graph_infra.ego_gnn" and
                    any(alias.name in ("EgoStore","*") for alias in node.names))
            module=(isinstance(node,ast.Import) and
                    any(alias.name=="tools.graph_infra.ego_gnn" for alias in node.names))
            parent=(isinstance(node,ast.ImportFrom) and node.module=="tools.graph_infra" and
                    any(alias.name=="ego_gnn" for alias in node.names))
            if direct or module or parent:
                foreign.append(path.relative_to(tree).as_posix())
        if "train.parquet" in text and BUILD_TOKENS.search(text):
            builds.append(path.relative_to(tree).as_posix())
        if "graph.npz" in text:
            mentions_canonical = "behavior_graph/graph.npz" in text or "behavior_graph\"" in text
            (reads if mentions_canonical else foreign).append(path.relative_to(tree).as_posix())
    foreign=sorted(set(foreign))
    if builds:
        return {"status": "FAIL", "tree": str(tree),
                "detail": "builds its own behaviour graph from train.parquet in " + ", ".join(sorted(builds)[:4])
                          + "; read the canonical " + str(CANONICAL / "graph.npz") + " instead",
                "builds": builds}
    if foreign:
        return {"status": "FAIL", "tree": str(tree),
                "detail": "reads a graph outside the canonical source in " + ", ".join(foreign[:4]),
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


def _inspect_named_tree(tree: Path) -> dict:
    """Only registered visible consumers are admitted, never a per-arm exemption."""
    stored = verify_named_graph(NAMED_GRAPH)
    if stored["status"] != "PASS":
        return {"status": "FAIL", "tree": str(tree), "detail": stored["detail"], "graph": stored}
    reads = []
    for path in _sources(tree):
        text = path.read_text(errors="ignore")
        relative = path.relative_to(tree).as_posix()
        try:
            parsed = ast.parse(text)
        except SyntaxError as exc:
            return {"status": "FAIL", "detail": "invalid Python source: " + str(exc)}
        if "train.parquet" in text and BUILD_TOKENS.search(text):
            return {"status": "FAIL", "detail": "candidate-local graph construction: " + relative}
        # A named graph never permits a second canonical/full/evaluation graph.
        if re.search(r"behavior_graph|full_description|struct_val|struct_test", text):
            return {"status": "FAIL", "detail": "mixed/evaluation graph in " + relative}
        if "graph.npz" in text:
            return {"status": "FAIL", "detail": "unregistered graph path in " + relative}
        ego_aliases = {"EgoStore"}
        for node in ast.walk(parsed):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.module == "tools.graph_infra":
                    modules += ["tools.graph_infra." + alias.name for alias in node.names]
            if any(module in ("tools.graph_infra.graph_build", "tools.graph_infra.splits")
                   or module.startswith("tools.graph_infra.build_assets") for module in modules):
                return {"status": "FAIL", "detail": "candidate imports a graph builder in " + relative}
            if "tests" not in path.relative_to(tree).parts and any(
                    module.startswith(("scipy.sparse", "networkx", "torch_geometric")) for module in modules):
                return {"status": "FAIL", "detail": "candidate-local graph construction import in " + relative}
            if isinstance(node, ast.ImportFrom) and node.module == "tools.graph_infra.ego_gnn":
                ego_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "EgoStore")
            if isinstance(node, ast.Import) and any("ego_gnn" in alias.name for alias in node.names):
                return {"status": "FAIL", "detail": "use the explicit registered EgoStore import"}
            if isinstance(node, ast.ImportFrom) and node.module == "tools.graph_infra" and any(
                    alias.name == "ego_gnn" for alias in node.names):
                return {"status": "FAIL", "detail": "use the explicit registered EgoStore import"}
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if re.search(r"visible_(forward|reverse|undirected)\.npz$", value):
                    return {"status": "FAIL", "detail": "unregistered visible graph path in " + relative}
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom) and node.module == "tools.graph_infra.ego_gnn":
                if any(alias.name == "*" for alias in node.names):
                    return {"status": "FAIL", "detail": "ambiguous wildcard graph consumer"}
                reads.append(relative)
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in ego_aliases:
                roots = list(node.args[1:]) + [kw.value for kw in node.keywords if kw.arg == "assets_root"]
                if any(kw.arg is None for kw in node.keywords) or any(isinstance(arg, ast.Starred) for arg in node.args):
                    return {"status": "FAIL", "detail": "dynamic EgoStore graph arguments in " + relative}
                for value in roots:
                    # Existing registered interface: EgoStore(ROOT, ROOT/'assets').
                    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
                            and isinstance(value.left, ast.Name)
                            and isinstance(value.right, ast.Constant) and value.right.value == "assets"):
                        return {"status": "FAIL", "detail": "unregistered EgoStore root in " + relative}
    if not reads:
        return {"status": "FAIL", "detail": "named graph declares no registered visible consumer"}
    return {"status": "PASS", "tree": str(tree), "detail": stored["detail"],
            "identity": stored["identity"], "graph": stored, "reads": sorted(set(reads))}


def main(argv: list) -> int:
    verdict = {"graph": verify_graph()}
    if len(argv) > 1:
        verdict["tree"] = inspect_tree(argv[1])
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
