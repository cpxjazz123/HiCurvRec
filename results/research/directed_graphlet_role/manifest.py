"""Issue #221 deliverable: MANIFEST.json with every input and output digest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ("D0", "D1", "D2", "D3", "D4", "U", "S")
INPUTS = ("results/research/behavior_graph_directed/graph.npz",
          "results/research/behavior_graph_directed/metadata.json",
          "results/research/directed_graphlet_role/instances.npz",
          "results/research/directed_graphlet_role/coverage.json",
          "results/research/directed_graphlet_role/subtypes.json",
          "stage1_GeneEmbedding/output/sentence_t5.npy",
          "results/stage0_build_parquet/train.parquet")


ANALYSIS = (
    "REPORT_DIRECTED_GRAPHLET_ROLE.md",
    "directed_graphlets.py", "directed_role.py", "evaluate.py",
    "probe_panel.py", "probe_panel.json", "probe_depth.py", "probe_depth.json",
    "diagnose_alignment.py", "diagnose_z.py", "diagnose_joint.json",
    "diagnose_structure.py", "diagnose_structure.json",
    "dgdv_features.py", "dgdv_meta.json", "dgdv_directed_keys.json",
    "dgdv_undirected_keys.json", "dgdv_shuffled_keys.json",
    "dirgnn_predictor.py", "dirgnn_predictor.json",
    "subgraph_predictor.py", "subgraph_predictor.json",
    "quantisation_gap.py", "quantisation_gap.json",
    "relation_gap.py", "relation_gap.json",
    "directed_edge_predictor.py", "directed_edge_predictor.json",
    "head_pretrained_directed.json",
    "tests/test_subgraph_controls.py", "tests/test_dgdv_leakage.py",
)


def digest(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main() -> None:
    payload = {"issue": 221, "inputs": {}, "arms": {}, "analysis": {}}
    for relative in INPUTS:
        path = REPO / relative
        if path.is_file():
            payload["inputs"][relative] = digest(path)
    for arm in ARMS:
        tree = REPO / "stage2_RQ-VAE" / f"TIGER_drq_{arm}"
        result = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}"
        if not tree.is_dir():
            continue
        record = {"tree": str(tree.relative_to(REPO)), "source": {}, "artifacts": {}}
        for path in sorted(tree.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                record["source"][str(path.relative_to(tree))] = digest(path)
        for name in ("rqvae_best.pth", "item_sids.json", "trq_metrics.json"):
            target = result / name
            if target.is_file():
                record["artifacts"][name] = digest(target)
        payload["arms"][arm] = record
    for name in ANALYSIS:
        path = HERE / name
        if path.is_file():
            payload["analysis"][name] = digest(path)
    (HERE / "MANIFEST.json").write_text(json.dumps(payload, indent=1))
    print("manifest written:", {arm: len(row["artifacts"]) for arm, row in payload["arms"].items()},
          "analysis files:", len(payload["analysis"]))


if __name__ == "__main__":
    main()
