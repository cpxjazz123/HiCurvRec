"""Category-conditioned SID structure metrics from real Amazon metadata.

The stage-0 parquet kept a ``categories`` column but every value is null, so the
real category path is recovered from the raw Amazon-2023 metadata dump that
still carries it, joined on an exact title match. Nothing here uses user
co-occurrence as a stand-in for a parent-child relation.

Metrics split in two:

* reused verbatim from the frozen ``sid_diag`` analysis - codeword usage and
  prefix collisions, so the readings stay comparable with the accepted model;
* added here - coarse/fine category agreement with L1 and L1+L2, in both
  directions, next to a label-permutation control that fixes the chance level.

All paths are constants and the script takes no arguments (CLAUDE.md §1). The
metric functions are importable so a comparison across training arms can reuse
them without duplicating the definition.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
CONFIG_PATH = ROOT / "stage2_RQ-VAE/curvature_RQ-VAE/curvature_config.py"
ITEMS_PARQUET = ROOT / "results/stage0_build_parquet/items.parquet"
RAW_METADATA = Path(
    "/home/wlia0047/ar57_scratch/wenyu/dataset/Amazon_2023_Instruments"
    "/raw/meta_categories/meta_Musical_Instruments.jsonl"
)
SUPERVISION_PATH = (
    ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments"
    / "category_supervision.npz"
)
OUTPUT_DIR = ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE/category_structure"

# Category path levels used as the coarse and fine supervision. Level 0 is the
# single marketplace name ("Musical Instruments") and carries no information.
COARSE_LEVEL = 1
FINE_LEVEL = 2
# Fallback used when a run ships 256 codes at every level, which is the
# case for the accepted model. Runs that change a level's codebook size must
# pass the real sizes: scoring a 32-code level against 256 codes would count
# 224 codes that do not exist, which inflates Gini and deflates the normalised
# entropy of that level.
N_CODES = 256


def codebook_sizes_from_checkpoint(arm_dir: Path) -> list[int]:
    """Per-level codebook sizes taken from the run's own checkpoint."""
    import torch

    checkpoint_path = arm_dir / "out/rqvae/instruments/rqvae_best.pth"
    if not checkpoint_path.is_file():
        return [N_CODES] * 3
    state = torch.load(checkpoint_path, map_location="cpu").get("state_dict", {})
    sizes: list[int] = []
    for level in range(3):
        weight = state.get(f"rq.vq_layers.{level}.embed.weight")
        if weight is None:
            return [N_CODES] * 3
        sizes.append(int(weight.shape[0]))
    return sizes
CONTROL_SEED = 2026


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment():
    return _load_module("curvature_config", CONFIG_PATH)


def _sid_diag():
    return _load_module("sid_diag", ROOT / "stage2_RQ-VAE/curvature_RQ-VAE/scripts/sid_diag.py")


def _raw_signature() -> dict[str, Any]:
    stat = RAW_METADATA.stat()
    items = ITEMS_PARQUET.stat()
    return {
        "raw_size": stat.st_size,
        "raw_mtime_ns": stat.st_mtime_ns,
        "items_size": items.st_size,
        "items_mtime_ns": items.st_mtime_ns,
        "coarse_level": COARSE_LEVEL,
        "fine_level": FINE_LEVEL,
    }


def _read_titles_to_paths() -> dict[str, list[tuple[str, ...]]]:
    """Stream the 632 MB metadata dump once, keeping only usable paths."""
    by_title: dict[str, list[tuple[str, ...]]] = {}
    with RAW_METADATA.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            title = record.get("title")
            path = record.get("categories")
            if not title or not isinstance(path, list) or not path:
                continue
            by_title.setdefault(title, []).append(tuple(path))
    return by_title


def build_category_supervision() -> dict[str, Any]:
    """Join items to their real category path and cache the labels.

    Returns the audit block; the arrays land in ``SUPERVISION_PATH``.
    """
    if not RAW_METADATA.is_file():
        raise FileNotFoundError(f"Raw Amazon metadata missing: {RAW_METADATA}")
    items = pd.read_parquet(ITEMS_PARQUET, columns=["item_id", "title"])
    items = items.sort_values("item_id")
    by_title = _read_titles_to_paths()

    coarse_names: dict[str, int] = {}
    fine_names: dict[str, int] = {}
    coarse = np.full(len(items), -1, dtype=np.int64)
    fine = np.full(len(items), -1, dtype=np.int64)
    matched = 0
    ambiguous = 0
    for row, title in enumerate(items["title"].tolist()):
        paths = by_title.get(title)
        if not paths:
            continue
        matched += 1
        distinct = set(paths)
        if len(distinct) > 1:
            ambiguous += 1
        path = paths[0]
        if len(path) > COARSE_LEVEL:
            name = path[COARSE_LEVEL]
            coarse[row] = coarse_names.setdefault(name, len(coarse_names))
        if len(path) > FINE_LEVEL:
            name = path[FINE_LEVEL]
            fine[row] = fine_names.setdefault(name, len(fine_names))

    audit = {
        "items": int(len(items)),
        "matched_titles": int(matched),
        "matched_fraction": float(matched / len(items)),
        "ambiguous_titles": int(ambiguous),
        "ambiguous_fraction": float(ambiguous / max(matched, 1)),
        "coarse_nodes": len(coarse_names),
        "fine_nodes": len(fine_names),
        "coarse_labelled_items": int((coarse >= 0).sum()),
        "fine_labelled_items": int((fine >= 0).sum()),
        "signature": _raw_signature(),
    }
    SUPERVISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SUPERVISION_PATH,
        item_id=items["item_id"].to_numpy(dtype=np.int64),
        coarse=coarse,
        fine=fine,
        coarse_names=np.asarray(sorted(coarse_names, key=coarse_names.get), dtype=object),
        fine_names=np.asarray(sorted(fine_names, key=fine_names.get), dtype=object),
        signature=np.asarray([json.dumps(audit["signature"], sort_keys=True)]),
    )
    return audit


def load_category_supervision() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the cached labels, rebuilding them when the inputs moved."""
    rebuild = not SUPERVISION_PATH.is_file()
    if not rebuild:
        with np.load(SUPERVISION_PATH, allow_pickle=True) as saved:
            rebuild = json.loads(str(saved["signature"][0])) != _raw_signature()
    if rebuild:
        build_category_supervision()
    with np.load(SUPERVISION_PATH, allow_pickle=True) as saved:
        return (
            saved["coarse"].copy(),
            saved["fine"].copy(),
            json.loads(str(saved["signature"][0])),
        )


def _purity(cluster: np.ndarray, label: np.ndarray) -> float:
    """Share of items whose cluster carries the cluster's majority label."""
    hits = 0
    for value in np.unique(cluster):
        members = cluster == value
        labels, counts = np.unique(label[members], return_counts=True)
        hits += int(counts.max())
    return float(hits / len(cluster))


def _concentration(cluster: np.ndarray, label: np.ndarray) -> float:
    """Item-weighted share of each label held by its single dominant cluster.

    This is the "do items of one category share a code" direction: 1.0 means
    every category is confined to one cluster, 1/clusters means the label is
    spread evenly.
    """
    hits = 0
    for value in np.unique(label):
        members = label == value
        _, counts = np.unique(cluster[members], return_counts=True)
        hits += int(counts.max())
    return float(hits / len(cluster))


def _agreement(
    cluster: np.ndarray, label: np.ndarray, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shuffled = label[rng.permutation(len(label))]
    return {
        "ami": float(adjusted_mutual_info_score(label, cluster)),
        "ami_control": float(adjusted_mutual_info_score(shuffled, cluster)),
        "purity_cluster_to_label": _purity(cluster, label),
        "concentration_label_to_cluster": _concentration(cluster, label),
        "n_items": int(len(label)),
        "n_labels": int(len(np.unique(label))),
        "n_clusters": int(len(np.unique(cluster))),
    }


def prefix_ids(tokens: np.ndarray, depth: int) -> np.ndarray:
    _, inverse = np.unique(tokens[:, :depth], axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def structure_metrics(
    tokens: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    code_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Category agreement plus the frozen usage/collision readings."""
    if tokens.shape[0] != len(coarse) or tokens.shape[0] != len(fine):
        raise ValueError("Token rows and supervision rows disagree")
    diag = _sid_diag()
    if code_sizes is None:
        code_sizes = [N_CODES] * tokens.shape[1]
    if len(code_sizes) != tokens.shape[1]:
        raise ValueError("code_sizes must have one entry per quantization level")
    usage_rows, usage = diag.code_usage(tokens, n_codes=N_CODES)
    for level, size in enumerate(code_sizes):
        # Re-score this level against its own codebook size.
        block = usage[f"L{level + 1}"]
        block["codewords"] = int(size)
        block["used_codewords"] = min(int(block["used_codewords"]), int(size))
        block["unused_codewords"] = int(size) - int(block["used_codewords"])
        counts = np.bincount(tokens[:, level], minlength=size).astype(np.float64)
        block["gini"] = diag.gini(counts)
        block["normalized_entropy"] = (
            diag.entropy_bits(counts) / np.log2(size) if size > 1 else 0.0
        )
    _, _, prefixes = diag.prefix_audit(tokens)
    del usage_rows

    coarse_mask = coarse >= 0
    fine_mask = fine >= 0
    cluster_sizes = np.unique(prefix_ids(tokens, 2), return_counts=True)[1]
    metrics: dict[str, Any] = {
        "l1l2_prefix_singleton_fraction": float((cluster_sizes == 1).mean()),
        "l1l2_prefix_clusters": int(len(cluster_sizes)),
        "coarse_vs_L1": _agreement(
            prefix_ids(tokens[coarse_mask], 1), coarse[coarse_mask], CONTROL_SEED
        ),
        "fine_vs_L1L2": _agreement(
            prefix_ids(tokens[fine_mask], 2), fine[fine_mask], CONTROL_SEED + 1
        ),
        "codeword_usage": usage,
        "prefix_structure": prefixes,
        "full_sid_unique": int(len(np.unique(tokens, axis=0))),
        "full_sid_collision_rate": float(
            1.0 - len(np.unique(tokens, axis=0)) / len(tokens)
        ),
    }
    return metrics


def main() -> None:
    audit = build_category_supervision()
    coarse, fine, stored = load_category_supervision()
    if stored != audit["signature"]:
        raise RuntimeError("Supervision cache signature mismatch after rebuild")
    tokens = np.load(_experiment().RAW_SIDS_NPY)
    metrics = structure_metrics(tokens, coarse, fine)
    report = {
        "sid_path": str(_experiment().RAW_SIDS_NPY),
        "supervision": audit,
        "metrics": metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "category_structure.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(
        f"[category] supervision matched={audit['matched_fraction']:.3f} "
        f"coarse_nodes={audit['coarse_nodes']} fine_nodes={audit['fine_nodes']} "
        f"ambiguous={audit['ambiguous_fraction']:.4f}",
        flush=True,
    )
    for name in ("coarse_vs_L1", "fine_vs_L1L2"):
        block = metrics[name]
        print(
            f"[category] {name:<12} AMI={block['ami']:.4f} "
            f"(control {block['ami_control']:.4f}) "
            f"purity={block['purity_cluster_to_label']:.4f} "
            f"concentration={block['concentration_label_to_cluster']:.4f} "
            f"labels={block['n_labels']} clusters={block['n_clusters']}",
            flush=True,
        )
    for level, block in metrics["codeword_usage"].items():
        print(
            f"[category] {level} used={block['used_codewords']}/{block['codewords']} "
            f"gini={block['gini']:.4f} norm_entropy={block['normalized_entropy']:.4f}",
            flush=True,
        )
    print(
        f"[category] unique SIDs={metrics['full_sid_unique']}/{len(tokens)} "
        f"collision_rate={metrics['full_sid_collision_rate']:.6f}",
        flush=True,
    )
    print(f"[category] wrote {OUTPUT_DIR / 'category_structure.json'}", flush=True)
    if not math.isfinite(metrics["coarse_vs_L1"]["ami"]):
        raise RuntimeError("Category agreement is not finite")


if __name__ == "__main__":
    main()
