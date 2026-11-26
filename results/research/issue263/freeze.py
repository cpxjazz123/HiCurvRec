"""Issue #263: freeze public infrastructure and the inputs used by the candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue263"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: str) -> dict:
    p = ROOT / path
    return {"path": path, "status": "PASS" if p.is_file() else "MISSING",
            "sha256": sha(p) if p.is_file() else None,
            "bytes": p.stat().st_size if p.is_file() else None}


def main() -> None:
    paths = [
        "results/research/issue263/ISSUE.json",
        "CLAUDE.md",
        ".agent/skills/complete-open-issue/SKILL.md",
        "assets/manifest.json",
        "results/stage0_build_parquet/train.parquet",
        "results/stage0_build_parquet/valid.parquet",
        "results/stage0_build_parquet/test.parquet",
        "stage1_GeneEmbedding/output/sentence_t5.npy",
        "results/research/behavior_graph_directed/graph.npz",
        "results/research/behavior_graph_directed/edges.parquet",
        "results/research/behavior_graph_directed/metadata.json",
        "results/research/directed_graphlet_role/dgdv.npz",
        "results/research/directed_graphlet_role/coverage.json",
        "results/research/directed_graphlet_role/instances.npz",
        "results/research/directed_graphlet_role/subtypes.json",
        "results/research/directed_graphlet_teacher/teacher/structure/v3/structure_targets_v3.npz",
        "results/research/directed_graphlet_teacher/teacher/structure/v3/teacher_structure_contract_v3.json",
        "assets/v8_runtime_layout/targets.npy",
        "assets/v8_runtime_layout/target_offsets.npy",
        "assets/node_moments.npz",
        "assets/index.npz",
        "assets/v10_direction_ego/metadata.json",
        "assets/v10_direction_ego/audit.json",
        "assets/v9_ego_operators/metadata.json",
        "assets/v5_gnn_geometry/node_features.npz",
        "assets/v5_gnn_geometry/node_features_report.json",
        "results/stage2_RQ-VAE/_runs/issue256_r01_formal/TIGER_module_budget_prefix_rq/result/item_sids.json",
        "results/stage2_RQ-VAE/_runs/issue256_r01_formal/TIGER_module_budget_prefix_rq/result/rqvae_best.pth",
        "results/stage2_RQ-VAE/_runs/issue256_r01_formal/TIGER_module_budget_prefix_rq/result/export_manifest.json",
        "results/stage3_T5Train/TIGER_module_budget_prefix_rq/logs/Amazon_2023_Instruments/Nov-25-2026_06-36-15/test_final.json",
        "stage3_T5Train/train_HG-Rec.py",
        "tools/research/stage2/contract.py",
        "tools/research/stage2/launcher.py",
        "tools/research/stage2/checks.py",
        "tools/research/stage2/sid_quality.py",
        "tools/iteration/depgraph.py",
        "tools/iteration/execution.py",
        "tools/iteration/analysis.py",
        "tools/iteration/replay.py",
        "tools/issue252/final_evaluation.py",
        "tools/issue253/evaluation_reference.py",
        "tools/graph_infra/ego_gnn.py",
    ]
    # Frozen direction-aware ego arrays are explicit assets.
    for variant in ("in_only", "out_only"):
        for name in ("nodes.npy", "offsets.npy", "node_counts.npy", "in_ptr.npy",
                     "in_col.npy", "out_ptr.npy", "out_col.npy",
                     "in_edge_counts.npy", "out_edge_counts.npy"):
            paths.append(f"assets/v10_direction_ego/{variant}/{name}")
    paths = sorted(set(paths))
    files = {p: record(p) for p in paths}
    issue = json.loads((OUT / "ISSUE.json").read_text())
    snap = {"schema": "genrec.issue263.infra_snapshot/1", "issue": issue["number"],
            "issue_state": issue["state"],
            "issue_body_sha256": hashlib.sha256(issue["body"].encode()).hexdigest(),
            "mode": "FROZEN", "files": files,
            "missing": sorted(p for p, v in files.items() if v["status"] != "PASS"),
            "policy": "public files and frozen assets must keep path:sha256; candidate source/cache is separate"}
    (OUT / "INFRA_SNAPSHOT_START.json").write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    lines = ["# Issue #263 public infrastructure freeze", "",
             f"Issue body sha256: `{snap['issue_body_sha256']}`  ",
             "Mode: `FROZEN`; public tools/assets are read-only for this candidate.",
             "", "| path | sha256 |", "|---|---|"]
    for p in paths:
        lines.append(f"| `{p}` | `{files[p]['sha256']}` |")
    if snap["missing"]:
        lines += ["", "## Missing", ""] + [f"- `{p}`" for p in snap["missing"]]
    (OUT / "INFRA.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"files": len(files), "missing": snap["missing"],
                      "issue_body_sha256": snap["issue_body_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
