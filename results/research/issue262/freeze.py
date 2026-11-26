"""Issue #262: freeze public infrastructure and asset identities before analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: str) -> dict:
    target = ROOT / path
    if not target.is_file():
        return {"path": path, "status": "MISSING", "sha256": None, "bytes": None}
    return {"path": path, "status": "PASS",
            "sha256": sha256(target), "bytes": target.stat().st_size}


def main() -> None:
    frozen = [
        "results/research/issue262/ISSUE.json",
        "CLAUDE.md",
        ".agent/skills/complete-open-issue/SKILL.md",
        "assets/manifest.json",
        "results/stage0_build_parquet/train.parquet",
        "results/stage0_build_parquet/valid.parquet",
        "results/stage0_build_parquet/test.parquet",
        "stage1_GeneEmbedding/output/sentence_t5.npy",
        "results/research/behavior_graph/edges.parquet",
        "results/research/behavior_graph/graph.npz",
        "results/research/behavior_graph/node_features.npy",
        "results/research/behavior_graph/metadata.json",
        "results/research/behavior_graph_directed/edges.parquet",
        "results/research/behavior_graph_directed/graph.npz",
        "results/research/behavior_graph_directed/node_features.npy",
        "results/research/behavior_graph_directed/metadata.json",
        "assets/v8_runtime_layout/targets.npy",
        "assets/v8_runtime_layout/target_offsets.npy",
        "assets/node_moments.npz",
        "assets/index.npz",
        "tools/graph_infra/assets.py",
        "tools/graph_infra/bank_features.py",
        "tools/graph_infra/build_assets_v2.py",
        "tools/graph_infra/heldout_eval.py",
        "tools/graph_infra/isolation.py",
        "tools/graph_infra/metrics.py",
        "tools/graph_infra/pairs.py",
        "tools/graph_infra/splits.py",
        "tools/issue251/evaluation.py",
        "tools/issue251/full_evaluation.py",
        "tools/issue252/final_evaluation.py",
        "tools/issue252/archive_audit.py",
        "tools/issue253/evaluation_reference.py",
        "tools/iteration/analysis.py",
        "tools/iteration/execution.py",
        "tools/iteration/depgraph.py",
        "tools/iteration/report.py",
        "tools/iteration/replay.py",
        "tools/research/stage2/contract.py",
        "tools/research/stage2/launcher.py",
        "stage3_T5Train/train_HG-Rec.py",
    ]
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    for entry in manifest["assets"]:
        frozen.append(entry["path"])
    # The strict pair banks and their source graph are part of the held-out contract.
    for path in sorted((ROOT / "results/research/infra246/assets/pairs").glob("*.npz")):
        frozen.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "results/research/infra246/assets/graph").glob("*.npz")):
        frozen.append(str(path.relative_to(ROOT)))
    frozen = sorted(set(frozen))
    files = {path: record(path) for path in frozen}
    issue = json.loads((ROOT / "results/research/issue262/ISSUE.json").read_text())
    body = issue["body"]
    snapshot = {
        "schema": "genrec.issue262.infra_snapshot/1",
        "issue": issue["number"],
        "issue_state": issue["state"],
        "issue_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "issue_title": issue["title"],
        "mode": "FROZEN",
        "files": files,
        "missing": sorted(path for path, value in files.items() if value["status"] != "PASS"),
        "policy": "hashes are compared before and after analysis; no file in this list may change",
    }
    (OUT / "INFRA_SNAPSHOT_START.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Issue #262 public-infrastructure freeze",
        "",
        f"Issue body sha256: `{snapshot['issue_body_sha256']}`  ",
        f"Files hashed: {len(files)}; missing: {len(snapshot['missing'])}  ",
        "Mode: `FROZEN`; no public file or frozen asset is modified by this issue.",
        "",
        "| path | sha256 |",
        "|---|---|",
    ]
    for path in frozen:
        value = files[path]
        lines.append(f"| `{path}` | `{value['sha256']}` |")
    if snapshot["missing"]:
        lines += ["", "## Missing", ""] + [f"- `{p}`" for p in snapshot["missing"]]
    (OUT / "INFRA.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"files": len(files), "missing": snapshot["missing"],
                      "issue_body_sha256": snapshot["issue_body_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
