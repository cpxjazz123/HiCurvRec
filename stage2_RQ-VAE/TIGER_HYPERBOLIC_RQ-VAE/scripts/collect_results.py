"""Collect every arm's Stage2, Stage3, SID-structure and behaviour metrics.

Run from the tree root with no arguments. Reads only native artifacts the runs
already produced, plus the frozen test split, and prints one JSON object with a
row per arm. Writes no files.

Behaviour metrics are fitted on TRAIN transitions and scored on TEST
transitions, which no Stage2 run ever read, so no arm can be favoured by having
seen its evaluation events.

This is the Exp6 analysis tool: it is meant to be re-run after every arm so the
comparison table is always generated the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(ROOT))

# arm label -> (stage2 results dir, stage3 results dir)
ARMS: dict[str, tuple[str, str]] = {
    "A_tiger": (
        "results/stage2_RQ-VAE/TIGER_RQ-VAE",
        "results/stage3_T5Train/tiger/logs/Amazon_2023_Instruments/Oct-01-2026_18-07-15",
    ),
    "C_iter48_R48": (
        "results/stage2_RQ-VAE/TIGER_BEHAVIOR_RQ-VAE",
        "results/stage3_T5Train/tiger_behavior/logs/Amazon_2023_Instruments/Oct-10-2026_16-25-18",
    ),
    "exp1a_resample": (
        "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample",
        "results/stage3_T5Train/tiger_sample_resample/logs/Amazon_2023_Instruments/Oct-10-2026_18-16-53",
    ),
    "exp1b_resample_fn_mask": (
        "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_fn_mask",
        "results/stage3_T5Train/tiger_sample_resample_fn_mask",
    ),
    "exp2_poincare_c1": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp2_poincare_c1",
        "results/stage3_T5Train/tiger_hyp_exp2_poincare_c1",
    ),
    "exp2_euclid_matched": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp2_euclid_matched",
        "results/stage3_T5Train/tiger_hyp_exp2_euclid_matched",
    ),
    "exp3_hyp_c0p1": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp3_hyp_c0p1",
        "results/stage3_T5Train/tiger_hyp_exp3_hyp_c0p1",
    ),
    "exp3_hyp_c0p5": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp3_hyp_c0p5",
        "results/stage3_T5Train/tiger_hyp_exp3_hyp_c0p5",
    ),
    "exp3_hyp_c5": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp3_hyp_c5",
        "results/stage3_T5Train/tiger_hyp_exp3_hyp_c5",
    ),
    "exp4_euclid_delta": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp4_euclid_delta",
        "results/stage3_T5Train/tiger_hyp_exp4_euclid_delta",
    ),
    "exp5_mobius_add": (
        "results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/exp5_mobius_add",
        "results/stage3_T5Train/tiger_hyp_exp5_mobius_add",
    ),
}

ALPHA = 5.0
SEED = 42


def transitions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sources, targets = [], []
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy(dtype=np.int64)
    ):
        if history is None or len(history) == 0:
            continue
        sources.append(int(history[-1]))
        targets.append(int(target))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def sid_structure(tokens: np.ndarray) -> dict:
    """Per-level cardinality, entropy, collision; plus conditional entropies."""
    n_items = len(tokens)
    out: dict = {"unique_3tuples": int(len(np.unique(tokens, axis=0)))}
    out["collision"] = 1.0 - out["unique_3tuples"] / n_items
    for level in range(tokens.shape[1]):
        # `codes` counts the level's own alphabet, which is always full at 256.
        # What changes across arms is how many distinct PREFIXES the level
        # induces, so report both.
        codes, counts = np.unique(tokens[:, level], return_counts=True)
        p = counts / counts.sum()
        out[f"L{level + 1}_codes_used"] = int(len(codes))
        out[f"L{level + 1}_entropy"] = float(-(p * np.log(p)).sum())
        out[f"L{level + 1}_top1_share"] = float(p.max())
        _, prefix_inverse = np.unique(
            tokens[:, : level + 1], axis=0, return_inverse=True
        )
        out[f"L{level + 1}_prefixes"] = int(prefix_inverse.max()) + 1
    # H(L2|L1) and H(L3|L2): how much uncertainty the next level adds.
    for parent, child in ((0, 1), (1, 2)):
        joint = pd.crosstab(tokens[:, parent], tokens[:, child]).to_numpy()
        p_joint = joint / joint.sum()
        p_child = p_joint.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(
                p_joint > 0,
                p_joint * np.log(p_joint / p_child[None, :]),
                np.nan,
            )
        out[f"H_L{child + 1}|L{parent + 1}"] = float(-np.nansum(cond))
    return out


def behaviour_metrics(tokens: np.ndarray, fit_pairs, eval_pairs) -> dict:
    """Transition prefix hit@10/100 and matched AUC, fitted train / scored test."""
    from bench_hier_behavior.behavior_metrics import evaluate_next_item_ranking

    frame = pd.read_parquet(ROOT / "results/stage0_build_parquet/test.parquet")
    metrics = evaluate_next_item_ranking(
        frame, tokens, fit_pairs[0], fit_pairs[1],
        alpha=ALPHA, seed=SEED, batch_size=256,
    )["metrics"]
    keep = {
        k: v for k, v in metrics.items()
        if k.startswith(("full_hit@10_", "full_hit@100_", "matched_auc_", "matched_hit@10_"))
    }
    return keep


def stage2_summary(results_dir: Path) -> dict:
    """Final-epoch collision/recon/behaviour figures from the arm's own metrics."""
    metrics = results_dir / "logs" / "training_metrics.jsonl"
    if not metrics.is_file():
        return {}
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    train = [r for r in rows if r.get("event") == "train"]
    starts = [r for r in rows if r.get("event") == "train_start"]
    end = [r for r in rows if r.get("event") == "train_end"]
    out: dict = {}
    if starts:
        s = starts[0]
        for key in (
            "resolved_temperature", "target_logit_spread", "geometry", "curvature",
            "tangent_scale", "target_radius_fraction", "transition",
            "distance_normalization", "resample_pairs", "false_negative_mask",
        ):
            if key in s:
                out[key] = s[key]
    if end:
        out["stage2_final_collision"] = end[0].get("final_collision")
        out["stage2_wall_time_s"] = end[0].get("total_wall_time_s")
    if train:
        last = train[-1]
        out["stage2_final_loss"] = last.get("loss")
        out["stage2_final_recon"] = last.get("recon")
        if "behavior_contrastive" in last:
            out["stage2_final_behavior_loss"] = last.get("behavior_contrastive")
        if "behavior_weight" in last:
            out["stage2_final_behavior_weight"] = last.get("behavior_weight")
    return out


def stage3_summary(log_dir: Path | None) -> dict:
    if log_dir is None or not log_dir.is_dir():
        return {}
    found = sorted(log_dir.rglob("test_final.json"))
    if not found:
        return {}
    d = json.loads(found[-1].read_text())
    return {
        "test_recall@5": d["test_recall@5"],
        "test_recall@10": d["test_recall@10"],
        "test_ndcg@5": d["test_ndcg@5"],
        "test_ndcg@10": d["test_ndcg@10"],
        "n_eval": d["n_eval"],
    }


def main() -> None:
    train_frame = pd.read_parquet(ROOT / "results/stage0_build_parquet/train.parquet")
    fit_pairs = transitions(train_frame)
    report: dict = {"fit_transitions": int(len(fit_pairs[0])), "arms": {}}
    for label, (sid_dir, s3_dir) in ARMS.items():
        sid_path = ROOT / sid_dir / "sids_for_hgrec.npy"
        row: dict = {"stage2_results_dir": sid_dir}
        if sid_path.is_file():
            tokens = np.load(sid_path)[:, :3].astype(np.int64)
            row.update(sid_structure(tokens))
            row.update(behaviour_metrics(tokens, fit_pairs, None))
            row.update(stage2_summary(ROOT / sid_dir))
        else:
            row["missing"] = str(sid_path)
        row.update(stage3_summary(ROOT / s3_dir if s3_dir else None))
        report["arms"][label] = row
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
