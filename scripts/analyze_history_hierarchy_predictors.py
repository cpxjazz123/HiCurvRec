"""Issue #48 driver: history-conditioned predictability, conditional refinement and
target ambiguity as out-of-family predictors of Stage 3 Recall@10 / NDCG@10.

Read-only with respect to every frozen arm. Reads the audited 38-model inventory,
the #47 manifest, the held-out valid pairs and each arm's verified SID export, runs
small from-scratch probes on train.parquet only, and writes exclusively into
``results/stage3_T5Train/monotonicity/history_hierarchy_predictors/``.

Five prespecified predictors, all on the same folds, sharing the #47 M0 baseline:

    M0  log(Hard L1 ER valid)
    M4  M0 + G_H1            history-conditioned L1 predictability gain
    M5  M0 + D2              conditional L2 refinement discrimination
    M6  M0 + A_target        target-weighted three-token ambiguity
    M7  M0 + G_H1 + D2       minimal two-new-feature complement

Usage:
    python3 scripts/analyze_history_hierarchy_predictors.py --stage features
    python3 scripts/analyze_history_hierarchy_predictors.py --stage probes --shard 0 --num-shards 4
    python0 scripts/analyze_history_hierarchy_predictors.py --stage predict
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history_hierarchy_features as H  # noqa: E402
import joint_prefix_lofo as L47  # noqa: E402

OUT_DIR = H.REPO_ROOT / "results/stage3_T5Train/monotonicity/history_hierarchy_predictors"
FEATURES_CSV = OUT_DIR / "per_model_features.csv"
PROBES_CSV = OUT_DIR / "probe_scores.csv"

FEATURE_BOOTSTRAP_SEED = 20240720
FEATURE_BOOTSTRAP_REPLICATES = 2000
CV_SEED = L47.CV_SEED
ALPHA_GRID = L47.ALPHA_GRID

FEATURE_SETS: dict[str, list[str]] = {
    "M0": ["log_er_l1"],
    "M4": ["log_er_l1", "G_H1"],
    "M5": ["log_er_l1", "D2"],
    "M6": ["log_er_l1", "A_target"],
    "M7": ["log_er_l1", "G_H1", "D2"],
}

PRIMARY_SEEDS = (42, 43, 44)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def stage_features() -> None:
    """F2 and F3 for every arm. No GPU, no probe training."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = H.load_inventory()
    anchors, positives, negatives, _ = H.load_valid_pairs()
    valid_targets = H.load_valid_target_ids()
    catalogue = H.load_catalogue_ids()

    prior = json.loads(H.PRIOR_MANIFEST.read_text())
    prior_rows = {
        r["model"]: r
        for r in csv.DictReader(H.PRIOR_FEATURES.open())
    }

    rows: list[dict[str, Any]] = []
    for m in inv["models"]:
        path = H.resolve_sid_path(m["sid_path"])
        sid, sha = H.load_sid(path)
        base = prior_rows[m["model"]]

        # Independent recheck of the #47 reference on this arm's own SIDs.
        pos_share = float((sid[positives, 0] == sid[anchors, 0]).mean())
        neg_share = float((sid[negatives, 0] == sid[anchors, 0][:, None]).mean())
        l1_er = pos_share / neg_share

        chrd = H.chrd_features(sid, anchors, positives, negatives)
        boot = H.chrd_cluster_bootstrap(
            sid, anchors, positives, negatives, anchors,
            FEATURE_BOOTSTRAP_REPLICATES, FEATURE_BOOTSTRAP_SEED,
        )
        amb = H.target_ambiguity(sid, valid_targets, catalogue)

        row: dict[str, Any] = {
            "model": m["model"],
            "family": m["family"],
            "sid_path": m["sid_path"],
            "sid_sha256": sha,
            "sid_sha_match_inventory": sha.startswith(m["sid_sha256"]),
            "sid_source": "repo" if (H.REPO_ROOT / m["sid_path"]).is_file() else "git_history",
            "stage3_recall@10": m["recall@10"],
            "stage3_ndcg@10": m["ndcg@10"],
            "stage3_n_eval": m["n_eval"],
            "pos_share": pos_share,
            "neg_share": neg_share,
            "log_er_l1": float(np.log(l1_er)),
            "er_l1_recheck": l1_er,
            "er_l1_from_issue47": float(base["er"]),
            **chrd,
            "D2_ci95_low": boot["ci95"][0],
            "D2_ci95_high": boot["ci95"][1],
            "D2_boot_replicates_defined": boot["replicates_defined"],
            **amb,
        }
        rows.append(row)

    cols = list(rows[0].keys())
    with FEATURES_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                json.dumps(r[c]) if isinstance(r[c], (list, dict)) else r[c] for c in cols
            ])
    print(f"wrote {FEATURES_CSV}  ({len(rows)} models)")
    ref = next(r for r in rows if r["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    print(f"  Resample recheck: L1 ER {ref['er_l1_recheck']!r} (expected {H.BASELINE_L1_ER!r})")


def stage_probes(shard: int, num_shards: int, seq_lens: tuple[int, ...] = (20, 1)) -> None:
    """F1 probes for a shard of the arms. Writes per-shard CSV fragments."""
    import torch

    from history_probe import ProbeConfig
    from history_probe_runner import run_arm

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = H.load_inventory()
    device = torch.device(f"cuda:{shard % torch.cuda.device_count()}")
    cfg = ProbeConfig()

    rows: list[dict[str, Any]] = []
    for i, m in enumerate(inv["models"]):
        if i % num_shards != shard:
            continue
        sid, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
        for seq_len in seq_lens:
            arm_cfg = ProbeConfig(**{**cfg.to_dict(), "seq_len": seq_len})
            for rec in run_arm(sid, arm_cfg, device, seeds=PRIMARY_SEEDS):
                rows.append({"model": m["model"], "family": m["family"], **rec})
        print(f"  shard {shard}: {m['model']} done", flush=True)

    frag = OUT_DIR / f"_probe_shard{shard}.csv"
    cols = sorted({k for r in rows for k in r})
    with frag.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"wrote {frag} ({len(rows)} probe runs)")


def merge_probe_shards() -> list[dict[str, Any]]:
    frags = sorted(OUT_DIR.glob("_probe_shard*.csv"))
    rows: list[dict[str, Any]] = []
    for frag in frags:
        with frag.open() as f:
            rows.extend(list(csv.DictReader(f)))
    cols = sorted({k for r in rows for k in r})
    with PROBES_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    for frag in frags:
        frag.unlink()
    return rows


def aggregate_probe_features(probe_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean over the three fixed seeds for the primary features; seed spread kept."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in probe_rows:
        key = (r["model"], int(r["seq_len"]))
        by.setdefault(key, []).append(r)

    out: dict[str, dict[str, Any]] = {}
    for (model, seq_len), recs in sorted(by.items()):
        def mean(field: str) -> float:
            vals = [float(r[field]) for r in recs if r.get(field) not in (None, "")]
            return float(np.mean(vals)) if vals else float("nan")

        def std(field: str) -> float:
            vals = [float(r[field]) for r in recs if r.get(field) not in (None, "")]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        entry = {
            "probe_runs": len(recs),
            "probe_seeds": sorted(int(r["seed"]) for r in recs),
            "G_H1": mean("rel_gain_l1"),
            "G_H1_seed_std": std("rel_gain_l1"),
            "gain_H1_nats": mean("gain_l1"),
            "probe_ce_l1": mean("probe_ce_l1"),
            "prior_ce_l1": mean("prior_ce_l1"),
            "probe_top1_l1": mean("probe_top1_acc_l1"),
            "probe_top20_l1": mean("probe_top20_acc_l1"),
            "prior_top1_l1": mean("prior_top1_acc_l1"),
            "prior_top20_l1": mean("prior_top20_acc_l1"),
            "rel_gain_l2": mean("rel_gain_l2"),
            "rel_gain_l3": mean("rel_gain_l3"),
            "gain_l2_nats": mean("gain_l2"),
            "gain_l3_nats": mean("gain_l3"),
            "probe_top1_l2": mean("probe_top1_acc_l2"),
            "probe_top1_l3": mean("probe_top1_acc_l3"),
            "train_rows": mean("train_rows"),
            "dev_rows": mean("dev_rows"),
            "fit_seconds_total": sum(float(r["fit_seconds"]) for r in recs),
        }
        out.setdefault(model, {})[f"seq{seq_len}"] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "probes", "merge", "predict"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seq-lens", default="20,1")
    args = ap.parse_args()

    if args.stage == "features":
        stage_features()
    elif args.stage == "probes":
        stage_probes(
            args.shard, args.num_shards,
            tuple(int(x) for x in args.seq_lens.split(",")),
        )
    elif args.stage == "merge":
        rows = merge_probe_shards()
        print(f"wrote {PROBES_CSV} ({len(rows)} probe runs)")
    elif args.stage == "predict":
        run_prediction()


def run_prediction() -> None:
    from history_hierarchy_report import build_outputs  # local import

    build_outputs(git_sha(), FEATURE_SETS, CV_SEED)


if __name__ == "__main__":
    main()
