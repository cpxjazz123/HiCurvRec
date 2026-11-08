"""Correlate the AF3 structural distortion with recall on the pairs that survive.

The repository audited 190 SID+recall pairs (118 VERIFIED). Measuring the
distortion of a SID needs three things at once: the SID table, the codebook that
turns its three integers into a 32-number readout, and the Stage 3 result. Only a
few pairs have all three, because binaries are kept out of git by policy
(CLAUDE.md section 46). This script measures the ones that do, reports the
correlation with its bootstrap interval, and states exactly what it had to skip
and why.

Usage:
    python3 analysis_sid_recall/af3_vs_recall.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "results" / "research" / "issue139"
sys.path.insert(0, str(RESEARCH))

import measure_structure as M  # noqa: E402
from measure_baseline import baseline_phi  # noqa: E402


def sid_tokens(sid_path: Path) -> torch.Tensor:
    payload = json.loads(sid_path.read_text())
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    keys = sorted(items, key=lambda key: int(key))
    return torch.tensor([[(items[k]["sid"] if isinstance(items[k], dict) else items[k])[level]
                          for level in range(3)] for k in keys], dtype=torch.long)


def readout_of(sid_path: Path) -> tuple[torch.Tensor, str]:
    """The 32-number readout of one SID table, from that arm's own codebook."""
    tree = REPO / "stage2_RQ-VAE" / sid_path.parent.name
    checkpoint = sid_path.parent / "rqvae_best.pth"
    tokens = sid_tokens(sid_path)
    embeddings = torch.from_numpy(np.load(M.EMBEDDINGS).astype(np.float32))
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    sys.path.insert(0, str(tree))
    import train_rqvae as trainer
    config = trainer._tokenizer_config()
    if (tree / "model" / "mixed_rq.py").is_file():
        from model.mixed_rq import MixedRQVAE, quantized_readout, split
        from model.poincare_contract import expmap0
        model = MixedRQVAE(config, in_dim=768, curvature=1.0)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            lefts, rights = [], []
            for level, layer in enumerate(model.rq.levels):
                left, right = split(layer.embed.weight[tokens[:, level]])
                lefts.append(left)
                rights.append(expmap0(right, model.rq.curvature)
                              if model.rq.curvature > 0 else right)
            return quantized_readout(lefts, rights, model.rq.curvature), "mixed"
    from model.model import RQVAE
    model = RQVAE(config, in_dim=768)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = torch.zeros(embeddings.shape[0], int(config.codebook_dim))
        for level, layer in enumerate(model.rq.vq_layers):
            out = out + layer.get_code_embs()[tokens[:, level]]
    return out, "euclidean"


def recall_at_10(path: Path) -> float | None:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    for key in ("test_recall@10", "test_recall_10", "recall@10"):
        if key in payload:
            return float(payload[key])
    return None


def rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap(x: np.ndarray, y: np.ndarray, draws: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(x), len(x))
        rho = spearman(x[pick], y[pick])
        if not math.isnan(rho):
            values.append(rho)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    graph = M.structure.load_graph(M.AF3_CACHE)
    kappa = M.structure.target_kappa(graph)
    import pandas as pd
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    support = M.support_for(graph, kappa, np.load(M.EMBEDDINGS)[train_ids])
    targets = M.per_pair_targets(kappa, support["pairs"])
    readout = json.loads(M.READOUT.read_text())
    index = torch.as_tensor(train_ids, dtype=torch.long)
    print("support %d pairs | calibration tau %.6f b %.6f"
          % (len(support["pairs"]), readout["tau"], readout["intercept"]), flush=True)

    rows = list(csv.DictReader(open(REPO / "analysis_sid_recall" / "pair_audit.csv")))
    measured, skipped = [], Counter()
    for row in rows:
        sid_path = REPO / row["sid_path"]
        if not sid_path.is_file():
            skipped["no SID table on disk"] += 1
            continue
        if not (sid_path.parent / "rqvae_best.pth").is_file():
            skipped["no codebook: the 32 numbers behind the codes are gone (CLAUDE.md 46)"] += 1
            continue
        recall = recall_at_10(Path(row["recall_path"])) if row.get("recall_path") else None
        if recall is None:
            skipped["no Stage 3 result"] += 1
            continue
        try:
            discrete, kind = readout_of(sid_path)
            value = M.distortion(discrete[index], support, kappa, targets,
                                 readout["tau"], readout["intercept"],
                                 graph["n_nodes"])["distortion"]
        except Exception as exc:  # noqa: BLE001
            skipped["measurement failed: " + type(exc).__name__] += 1
            continue
        measured.append({"variant": row["variant"], "stage2_rel": row["stage2_rel"],
                         "verdict": row["verdict"], "commit_tier": row.get("commit_tier"),
                         "kind": kind, "D_af3": value, "recall@10": recall,
                         "family": row["stage2_rel"].split("/")[0]})
        print("  %-46s %-9s D %.6f  R@10 %.6f  [%s]"
              % (row["variant"][:46], kind, value, recall, row["verdict"]), flush=True)

    out = {"pairs_total": len(rows), "measured": measured, "skipped": dict(skipped),
           "calibration": {"tau": readout["tau"], "intercept": readout["intercept"]},
           "why_so_few": "a distortion needs the SID table, the codebook and a Stage 3 "
                         "result at once; binaries are kept out of git by CLAUDE.md 46"}
    d = np.array([m["D_af3"] for m in measured]) if measured else np.array([])
    r = np.array([m["recall@10"] for m in measured]) if measured else np.array([])
    if len(d) >= 5:
        rho = spearman(d, r)
        low, high = bootstrap(d, r)
        out["spearman"] = {"rho": rho, "ci_low": low, "ci_high": high, "n": len(d)}
        print("\nSpearman(D_AF3, recall@10) = %+.4f  95%% CI [%+.4f, %+.4f]  n=%d"
              % (rho, low, high, len(d)))
        groups = {}
        for key in ("verdict", "kind"):
            for value_name in sorted({m[key] for m in measured}):
                subset = [m for m in measured if m[key] == value_name]
                if len(subset) >= 5:
                    groups[key + ":" + value_name] = {
                        "n": len(subset),
                        "rho": spearman(np.array([m["D_af3"] for m in subset]),
                                        np.array([m["recall@10"] for m in subset]))}
        out["groups"] = groups
        for name, value in groups.items():
            print("  within %-24s rho %+.4f (n=%d)" % (name, value["rho"], value["n"]))
        out["families"] = dict(Counter(m["family"] for m in measured))
        print("families:", out["families"])
        print("recall spread: %.6f .. %.6f (%.2f%% of the mean)"
              % (r.min(), r.max(), (r.max() - r.min()) / r.mean() * 100))
    print("skipped:", dict(skipped))
    path = RESEARCH / "af3_vs_recall.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
