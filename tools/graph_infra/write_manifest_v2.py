"""Write the v2 manifest from the frozen files on disk (issue #246 C34).

The build script writes the manifest itself; this entry point rebuilds it from the same
files and the same metadata rules, so a manifest can always be regenerated without
recomputing the assets.  It never overwrites an asset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets, config  # noqa: E402

ROOT = config.ASSET_ROOT
MANIFEST = ROOT / "manifest.json"
COMMAND = "python3 tools/graph_infra/build_assets_v2.py"
GENERALISATION = "generalisation"
DESCRIPTION = "description"
EVALUATION = "evaluation"

GRAPH_GROUPS = {
    "full_description": DESCRIPTION,
    "visible": GENERALISATION,
    "struct_val": EVALUATION,
    "struct_test": EVALUATION,
}


def _shape_dtype(path: Path):
    if path.suffix == ".npz":
        blob = np.load(path, allow_pickle=False)
        names = [name for name in blob.files if name not in ("names", "family_names", "kind_names")]
        if "indptr" in blob.files:
            return [int(x) for x in blob["shape"]], "int64"
        if "nodes" in blob.files:
            return [int(blob["nodes"].shape[0])], "mixed"
        if "counts" in blob.files:
            return [int(x) for x in blob["counts"].shape], "int64"
        return [int(blob[names[0]].shape[0])] if names else [0], "mixed"
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [int(len(payload))], "json"
    return [path.stat().st_size], "binary"


def build_entries() -> list:
    entries = []
    graph_dir = ROOT / "graph"
    for name, use in GRAPH_GROUPS.items():
        for kind in ("forward", "reverse", "undirected"):
            path = graph_dir / f"{name}_{kind}.npz"
            if not path.exists():
                continue
            shape, dtype = _shape_dtype(path)
            entries.append(assets.entry_v2(
                f"graph_{name}_{kind}", path, repo=REPO, schema="csr.int64", shape=shape,
                dtype=dtype, graph_version=f"{config.VERSION}/{name}", legal_use=use,
                version=config.VERSION,
                purpose=f"{kind} graph of the {name} split",
                command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
                code_sha256="see manifest.code_sha256", dependencies=["train.parquet"]))
    for name, purpose, use in (
            ("held_out_support",
             "undirected support of edges held out from the visible graph (struct_val ∪ struct_test "
             "minus visible), both directions", GENERALISATION),
            ("split_users", "real user ids of the three train splits", GENERALISATION)):
        path = graph_dir / f"{name}.npz"
        if path.exists():
            shape, dtype = _shape_dtype(path)
            entries.append(assets.entry_v2(
                name, path, repo=REPO, schema="npz.v2", shape=shape, dtype=dtype,
                graph_version=f"{config.VERSION}/visible", legal_use=use, version=config.VERSION,
                purpose=purpose, command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
                code_sha256="see manifest.code_sha256", dependencies=["train.parquet"]))
    stats = graph_dir / "stats.json"
    if stats.exists():
        entries.append(assets.entry_v2(
            "graph_stats", stats, repo=REPO, schema="json", shape=[1], dtype="json",
            graph_version=f"{config.VERSION}/visible", legal_use=GENERALISATION,
            version=config.VERSION,
            purpose="counts, split audit, self loops and the independent cross-check",
            command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
            code_sha256="see manifest.code_sha256", dependencies=["train.parquet"]))
    for name, purpose, use in (
            ("ego_full_1hop_visible", "complete one-hop induced ego blocks", GENERALISATION),
            ("ego_sampled_visible", "budgeted (cap 32) ego blocks with hop tags", GENERALISATION),
            ("ego_sampled_full_description", "ego blocks of the full train graph",
             DESCRIPTION)):
        path = ROOT / "ego" / f"{name}.npz"
        if path.exists():
            shape, dtype = _shape_dtype(path)
            entries.append(assets.entry_v2(
                name, path, repo=REPO, schema="ego.offset.v2", shape=shape, dtype=dtype,
                graph_version=f"{config.VERSION}/visible" if "visible" in name
                else f"{config.VERSION}/full_description",
                legal_use=use, version=config.VERSION, purpose=purpose, command=COMMAND,
                source_sha256=assets.sha256(config.TRAIN), code_sha256="see manifest.code_sha256",
                dependencies=["graph_visible_forward"]))
    for name, purpose, use in (
            ("motif_counts", "exact 3-node family counts per centre", GENERALISATION),
            ("motif_instances_visible", "sampled 3-node instances with orbits", GENERALISATION),
            ("motif_instances_4node_visible", "sampled 4-node instances (declared sampling)",
             GENERALISATION)):
        path = ROOT / "motifs" / f"{name}.npz"
        if path.exists():
            shape, dtype = _shape_dtype(path)
            entries.append(assets.entry_v2(
                name, path, repo=REPO, schema="npz.v2", shape=shape, dtype=dtype,
                graph_version=f"{config.VERSION}/visible", legal_use=use, version=config.VERSION,
                purpose=purpose, command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
                code_sha256="see manifest.code_sha256", dependencies=["graph_visible_forward"]))
    for path in sorted((ROOT / "pairs").glob("*.npz")):
        if path.name.startswith("_"):
            continue
        name = path.stem
        use = GENERALISATION if name.startswith("train") or name == "probe" else EVALUATION
        shape, dtype = _shape_dtype(path)
        entries.append(assets.entry_v2(
            f"bank_{name}", path, repo=REPO, schema="bank.v2", shape=shape, dtype=dtype,
            graph_version=f"{config.VERSION}/visible", legal_use=use, version=config.VERSION,
            purpose=f"{name} pair bank with per-pair graph features and witnesses",
            command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
            code_sha256="see manifest.code_sha256", dependencies=["graph_visible_forward",
                                                                 "held_out_support"]))
    for path in sorted((ROOT / "curvature").glob("*")) if (ROOT / "curvature").exists() else []:
        if path.is_file():
            shape, dtype = _shape_dtype(path)
            entries.append(assets.entry_v2(
                path.stem, path, repo=REPO, schema="npz.v2" if path.suffix == ".npz" else "json",
                shape=shape, dtype=dtype, graph_version=f"{config.VERSION}/visible",
                legal_use=GENERALISATION, version=config.VERSION,
                purpose="exact Ollivier-Ricci on the visible graph for frozen reference nodes",
                command=COMMAND, source_sha256=assets.sha256(config.TRAIN),
                code_sha256="see manifest.code_sha256", dependencies=["graph_visible_undirected"]))
    return entries


def main() -> int:
    if not ROOT.exists():
        raise SystemExit(f"no v2 assets at {ROOT}")
    entries = build_entries()
    payload = {"issue": 246, "version": config.VERSION,
               "notes": "v2 assets built by tools/graph_infra/build_assets_v2.py; this manifest was "
                        "regenerated from the frozen files without recomputing them",
               "assets": entries}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit = assets.audit_manifest(MANIFEST)
    verified = assets.verify_manifest(MANIFEST)
    print(json.dumps({"assets": len(entries), "schema_ok": audit["ok"],
                      "verified": len(verified["verified"]),
                      "issues": audit["issues"][:5]}, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
