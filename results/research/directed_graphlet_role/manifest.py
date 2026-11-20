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


def digest(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main() -> None:
    payload = {"issue": 221, "inputs": {}, "arms": {}}
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
    (HERE / "MANIFEST.json").write_text(json.dumps(payload, indent=1))
    print("manifest written:", {arm: len(row["artifacts"]) for arm, row in payload["arms"].items()})


if __name__ == "__main__":
    main()
