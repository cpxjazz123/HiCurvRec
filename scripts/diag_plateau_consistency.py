#!/usr/bin/env python3
"""Diag: Cross-algorithm plateau consistency analysis.

Goal: Determine whether the seed43 early-training plateau is an algorithm-specific
issue (only baseline A) or a universal pattern (occurs in HRQ / AQ / baseline).

We sweep all available training logs under logs/train/runs/ and detect plateau
windows based on val_R@10 and val_loss stagnation.

Plateau definition (user-specified):
- Window of length >= MIN_LEN (default 200) steps where:
  - val_R@10 does NOT improve by more than R_THRESH (default 0.005)
  - val_loss does NOT decrease by more than L_THRESH (default 0.5)
- After window ends, did val_R@10 recover within next 200 steps? -> Y / N.

Outputs:
- result/diag_plateau_consistency/plateau_window_per_seed.csv
- result/diag_plateau_consistency/per_algo_summary.csv
- result/diag_plateau_consistency/verdict.md
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs")
OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_plateau_consistency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_LEN = 200  # min plateau window length (steps)
R_THRESH = 0.005  # max val_R@10 improvement allowed
L_THRESH = 0.5  # max val_loss decrease allowed
RECOVERY_LOOKAHEAD = 200  # steps to check for recovery after plateau

# Algorithm classification: (run_dir_name, algo_label, seed)
# Naming convention chosen by the user task spec.
TARGETS = [
    ("p4_seed42", "A_baseline", 42),
    ("p4_seed43", "A_baseline", 43),
    ("p4_seed44", "A_baseline", 44),
    ("p4_seed43_patience8", "A_baseline_p8", 43),
    ("task18_hrq_s3", "HRQ", 42),
    ("task18_hrq_s3_seed43", "HRQ", 43),
    ("task19_aq_s3", "AQ", 42),
    ("task19_aq_s3_seed43", "AQ", 43),
    ("task19_aq_s3_seed44", "AQ", 44),
]

# CSV metrics files we care about
VAL_R10 = "val/recall@10"
VAL_LOSS = "val/loss"


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
def load_metrics(csv_path: Path) -> dict:
    """Load val_R@10, val_loss, val_NDCG@10 as {col: [(step, value)]}."""
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_s = row.get("step", "").strip()
            if not step_s:
                continue
            try:
                step = int(step_s)
            except ValueError:
                continue
            for k in (VAL_R10, VAL_LOSS, "val/ndcg@10"):
                v = row.get(k, "").strip()
                if not v:
                    continue
                try:
                    out[k].append((step, float(v)))
                except ValueError:
                    continue
    return out


def sorted_series(d: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    arr = sorted(d.get(key, []), key=lambda x: x[0])
    if not arr:
        return np.array([]), np.array([])
    steps, vals = zip(*arr)
    return np.array(steps, dtype=int), np.array(vals, dtype=float)


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------
def find_plateau(
    steps: np.ndarray,
    vals_r10: np.ndarray,
    vals_loss: np.ndarray,
    min_len: int = MIN_LEN,
    r_thresh: float = R_THRESH,
    l_thresh: float = L_THRESH,
    recovery_lookahead: int = RECOVERY_LOOKAHEAD,
) -> dict:
    """Find first plateau window meeting definition.

    A plateau is a contiguous segment of length >= min_len where:
      - max(val_R@10) - min(val_R@10) <= r_thresh (no big improvement)
      - max(val_loss) - min(val_loss) <= l_thresh (no big drop)

    We also test recovery: does val_R@10 improve by >= r_thresh in the next
    recovery_lookahead steps? If yes -> recovered=Y, else N.

    Returns dict with keys:
      plateau_start, plateau_end, plateau_length, recovered, best_step,
      best_r10, n_eval_points
    """
    if len(steps) < 3:
        return {
            "plateau_start": "",
            "plateau_end": "",
            "plateau_length": 0,
            "recovered": "",
            "best_step": int(steps[-1]) if len(steps) else "",
            "best_r10": float(vals_r10.max()) if len(vals_r10) else 0.0,
            "n_eval_points": int(len(steps)),
            "no_plateau": True,
        }

    # Best step & best R@10 across full run
    best_idx = int(np.argmax(vals_r10))
    best_step = int(steps[best_idx])
    best_r10 = float(vals_r10[best_idx])

    n = len(steps)
    plateau_start = None
    plateau_end = None
    recovered = "N"

    # Sliding window: find FIRST window of length >= min_len meeting thresholds
    # We anchor on consecutive evaluation rows.
    for i in range(n):
        # need to cover min_len steps (not rows); use steps[i] .. last row whose
        # step <= steps[i] + min_len
        end = i
        while end + 1 < n and steps[end + 1] - steps[i] <= min_len:
            end += 1
        if end - i + 1 < 2:
            continue
        seg_r10 = vals_r10[i : end + 1]
        seg_loss = vals_loss[i : end + 1]
        if seg_r10.max() - seg_r10.min() <= r_thresh and seg_loss.max() - seg_loss.min() <= l_thresh:
            plateau_start = int(steps[i])
            plateau_end = int(steps[end])
            # Recovery: does any post-window eval (within recovery_lookahead) exceed
            # the plateau's max R@10 by >= r_thresh?
            plateau_max_r10 = float(seg_r10.max())
            for j in range(end + 1, n):
                if steps[j] - plateau_end > recovery_lookahead:
                    break
                if vals_r10[j] - plateau_max_r10 >= r_thresh:
                    recovered = "Y"
                    break
            break

    if plateau_start is None:
        return {
            "plateau_start": "",
            "plateau_end": "",
            "plateau_length": 0,
            "recovered": "",
            "best_step": best_step,
            "best_r10": best_r10,
            "n_eval_points": n,
            "no_plateau": True,
        }

    return {
        "plateau_start": plateau_start,
        "plateau_end": plateau_end,
        "plateau_length": int(plateau_end - plateau_start),
        "recovered": recovered,
        "best_step": best_step,
        "best_r10": best_r10,
        "n_eval_points": n,
        "no_plateau": False,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze() -> tuple[list[dict], list[dict]]:
    per_seed: list[dict] = []
    for run_dir, algo, seed in TARGETS:
        csv_path = RUNS_ROOT / run_dir / "csv" / "version_0" / "metrics.csv"
        # Try version_0 first; fall back to version_4 (legacy) if not present.
        if not csv_path.exists():
            for v in ("version_4", "version_3", "version_2", "version_1"):
                alt = RUNS_ROOT / run_dir / "csv" / v / "metrics.csv"
                if alt.exists():
                    csv_path = alt
                    break
        if not csv_path.exists():
            per_seed.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "run_dir": run_dir,
                    "csv_path": "",
                    "plateau_start": "",
                    "plateau_end": "",
                    "plateau_length": 0,
                    "recovered": "",
                    "best_step": "",
                    "best_r10": "",
                    "n_eval_points": 0,
                    "no_plateau": True,
                    "missing": True,
                }
            )
            continue

        d = load_metrics(csv_path)
        steps, vals_r10 = sorted_series(d, VAL_R10)
        _, vals_loss = sorted_series(d, VAL_LOSS)
        if len(vals_loss) < len(vals_r10):
            # val_loss may have fewer rows if some epochs lack val; pad with last.
            pass
        # Align lengths: in practice they match in the same row but val_loss may be
        # logged at end-of-epoch only. We use the same length as vals_r10.
        n = min(len(vals_r10), len(vals_loss))
        steps = steps[:n]
        vals_r10 = vals_r10[:n]
        vals_loss = vals_loss[:n]
        info = find_plateau(steps, vals_r10, vals_loss)
        row = {
            "algo": algo,
            "seed": seed,
            "run_dir": run_dir,
            "csv_path": str(csv_path),
            **{k: v for k, v in info.items()},
        }
        row["missing"] = False
        per_seed.append(row)
    # per-algorithm summary
    agg: dict[str, dict] = defaultdict(
        lambda: {"n_seeds": 0, "n_plateau": 0, "n_recovered": 0, "lens": [], "best_r10s": []}
    )
    for r in per_seed:
        if r.get("missing"):
            continue
        a = r["algo"]
        agg[a]["n_seeds"] += 1
        if not r.get("no_plateau"):
            agg[a]["n_plateau"] += 1
            agg[a]["lens"].append(r["plateau_length"])
            if r["recovered"] == "Y":
                agg[a]["n_recovered"] += 1
        if isinstance(r["best_r10"], float):
            agg[a]["best_r10s"].append(r["best_r10"])
    per_algo: list[dict] = []
    for algo, s in sorted(agg.items()):
        n = s["n_seeds"]
        np_ = s["n_plateau"]
        avg_len = float(np.mean(s["lens"])) if s["lens"] else 0.0
        rec_rate = (s["n_recovered"] / np_) if np_ else 0.0
        max_r10 = float(max(s["best_r10s"])) if s["best_r10s"] else 0.0
        per_algo.append(
            {
                "algo": algo,
                "n_seeds": n,
                "n_plateau": np_,
                "plateau_rate": np_ / n if n else 0.0,
                "avg_plateau_len": avg_len,
                "recovery_rate": rec_rate,
                "max_best_r10": max_r10,
            }
        )
    return per_seed, per_algo


# ---------------------------------------------------------------------------
# Kill-line evaluation & verdict writing
# ---------------------------------------------------------------------------
def evaluate_kill_line(per_algo: list[dict]) -> str:
    """Per user-spec kill line:
      - if all baseline seeds have plateau AND >= 50% of HRQ/AQ seeds have plateau
        -> "universal".
      - if only baseline/seed43 -> "baseline-specific".
      - else -> "mixed".
    """
    base = next((x for x in per_algo if x["algo"] == "A_baseline"), None)
    hrq = next((x for x in per_algo if x["algo"] == "HRQ"), None)
    aq = next((x for x in per_algo if x["algo"] == "AQ"), None)

    def p(x):
        return x["plateau_rate"] if x else 0.0

    base_rate = p(base)
    hrq_rate = p(hrq)
    aq_rate = p(aq)

    all_base = base_rate == 1.0
    h_or_a_pool = []
    for x in (hrq, aq):
        if x:
            h_or_a_pool.append(x["plateau_rate"])
    avg_ha = float(np.mean(h_or_a_pool)) if h_or_a_pool else 0.0

    if all_base and avg_ha >= 0.5:
        return "universal"
    # baseline-specific: only baseline (or only baseline/seed43)
    non_base_with_plateau = sum(1 for x in (hrq, aq) if x and x["n_plateau"] > 0)
    if non_base_with_plateau == 0 and base_rate >= 0.5:
        return "baseline-specific"
    if hrq_rate >= 0.5 or aq_rate >= 0.5:
        return "mixed"
    return "weak/unclear"


def write_outputs(per_seed: list[dict], per_algo: list[dict], kill_line: str) -> None:
    ps_csv = OUT_DIR / "plateau_window_per_seed.csv"
    with ps_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "seed",
                "run_dir",
                "plateau_start",
                "plateau_end",
                "plateau_length",
                "recovered",
                "best_step",
                "best_r10",
                "n_eval_points",
                "no_plateau",
                "missing",
                "csv_path",
            ],
        )
        w.writeheader()
        for r in per_seed:
            w.writerow(r)

    pa_csv = OUT_DIR / "per_algo_summary.csv"
    with pa_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "n_seeds",
                "n_plateau",
                "plateau_rate",
                "avg_plateau_len",
                "recovery_rate",
                "max_best_r10",
            ],
        )
        w.writeheader()
        for r in per_algo:
            w.writerow(r)

    summary_json = OUT_DIR / "summary.json"
    summary_json.write_text(
        json.dumps(
            {"per_seed": per_seed, "per_algo": per_algo, "kill_line": kill_line},
            indent=2,
            default=str,
        )
    )

    verdict = OUT_DIR / "verdict.md"
    lines = [
        "# Diag #4: 训练平台期跨算法一致性",
        "",
        "## 数据来源",
        "",
        "扫取 `logs/train/runs/` 下 9 个目标 run 的 `metrics.csv`：",
        "",
        "| run_dir | algo | seed |",
        "|---|---|---|",
    ]
    for r in per_seed:
        lines.append(f"| {r['run_dir']} | {r['algo']} | {r['seed']} |")
    lines += [
        "",
        "## 平台期定义",
        "",
        f"- window step 跨度 ≥ {MIN_LEN}",
        f"- 窗内 val_R@10 极差 ≤ {R_THRESH}",
        f"- 窗内 val_loss 极差 ≤ {L_THRESH}",
        f"- 平台期结束后 {RECOVERY_LOOKAHEAD} 步内 val_R@10 是否再上涨 ≥ {R_THRESH} -> Y/N",
        "",
        "## Per-seed 平台期窗口",
        "",
        "| algo | seed | plateau_start | plateau_end | length | recovered | best_step | best_r10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_seed:
        lines.append(
            f"| {r['algo']} | {r['seed']} | {r['plateau_start']} | {r['plateau_end']} | "
            f"{r['plateau_length']} | {r['recovered']} | {r['best_step']} | "
            f"{r['best_r10'] if r['best_r10'] != '' else '-'} |"
        )
    lines += [
        "",
        "## Per-algo 聚合",
        "",
        "| algo | n_seeds | n_plateau | plateau_rate | avg_plateau_len | recovery_rate | max_best_r10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_algo:
        lines.append(
            f"| {r['algo']} | {r['n_seeds']} | {r['n_plateau']} | "
            f"{r['plateau_rate']:.3f} | {r['avg_plateau_len']:.0f} | "
            f"{r['recovery_rate']:.3f} | {r['max_best_r10']:.5f} |"
        )
    lines += [
        "",
        "## Kill Line 判定",
        "",
        f"- 规则: 若 A_baseline plateau_rate = 100% **且** (HRQ/AQ 平均) plateau_rate ≥ 50% -> `universal`；若仅 baseline 出现 -> `baseline-specific`。",
        f"- 实际结果: **{kill_line}**",
        "",
        "## 解读",
        "",
    ]
    if kill_line == "universal":
        lines += [
            "- 平台期是跨算法的普遍现象，seed43 的早期平台不能归因于 baseline 算法的特有问题。",
            "- 后续若要确认根因，应聚焦于训练通用机制（数据规模/优化器步长/early-step 初始化），而非 baseline 算法本身。",
        ]
    elif kill_line == "baseline-specific":
        lines += [
            "- 仅 baseline (A) 算法出现平台期，HRQ/AQ 都未触发判定窗口。",
            "- seed43 的早期平台是 A_baseline 的特有问题，可被 HRQ/AQ 规避。",
        ]
    elif kill_line == "mixed":
        lines += [
            "- 平台期在 baseline 与 HRQ/AQ 中都出现，但比例/长度不一。",
            "- 不能简单归类为 universal 或 baseline-specific；建议查看具体平台窗口是否都发生在训练早期（< 500 步）。",
        ]
    else:
        lines += [
            "- 没有任何算法触发平台期判定窗口，或触发比例很低。",
            "- 用户报告的 seed43 早期平台不在本次扫描条件内（min_len=200 且 r_thresh=0.005 / l_thresh=0.5），需调参或扩大扫描。",
        ]
    verdict.write_text("\n".join(lines) + "\n")


def main():
    per_seed, per_algo = analyze()
    kill_line = evaluate_kill_line(per_algo)
    write_outputs(per_seed, per_algo, kill_line)
    print("== per-seed ==")
    for r in per_seed:
        print(r)
    print("== per-algo ==")
    for r in per_algo:
        print(r)
    print(f"KILL_LINE: {kill_line}")
    print(f"OUT_DIR: {OUT_DIR}")


if __name__ == "__main__":
    main()