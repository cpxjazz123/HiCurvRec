"""Is the short probe's export a faithful stand-in for the cold run's export?

The screening phase runs every candidate for a bounded number of steps and reads the export
it produces. If that export predicted the cold run's balance or its enrichment ratio, the
screening could refuse a candidate that would concentrate the codes, and could rank the
portfolio by the enrichment ratio it expects -- without spending a cold run.

This measures the question on the arms that have already run, by comparing each arm's probe
export with the export of its cold run:

* the balance: the level-0 usage's Gini and the number of level-0 codes in use, against the
  archived control's;
* the enrichment ratio: the train-side positive/negative share ratio of the export, against
  the same ratio for the probe's own baseline -- the probe runs the arm's mechanism off as
  well, so the difference is paired and the short export's own scale does not matter.

Run it with the project's environment (it needs numpy):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \\
        tools/genrec_agents/probe_fidelity.py

The answer is a number, and the system is supposed to act on it either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools" / "mvg") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "mvg"))

import numpy as np  # noqa: E402

from export_sensitivity import (  # noqa: E402
    load_sids,
    train_pairs,
    train_side_discrimination,
)

ROUNDS = REPO_ROOT / "results" / "harness3_runs" / "controller" / "rounds"
AUDIT = REPO_ROOT / "results" / "harness3_runs" / "controller" / "audit.jsonl"
STAGE2 = REPO_ROOT / "results" / "stage2_RQ-VAE"
CONTROL_SID = STAGE2 / "r8_control" / "sids_for_hgrec.npy"
TRAIN_PARQUET = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"


def gini(counts: np.ndarray) -> float:
    """The Gini coefficient of a count vector."""
    values = np.sort(np.asarray(counts, dtype="float64"))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n)


def balance(sid_path: Path) -> Dict[str, Any]:
    """The level-0 usage of an export: how many codes are in use, and how unevenly."""
    counts = np.bincount(load_sids(sid_path)[:, 0])
    used = counts[counts > 0]
    return {
        "distinct_codes": int((counts > 0).sum()),
        "gini": gini(used),
        "largest_share": float(used.max() / used.sum()) if used.size else 0.0,
    }


def spearman(first: List[float], second: List[float]) -> Optional[float]:
    """The rank correlation of two sequences."""
    if len(first) < 3:
        return None
    a = np.asarray(first, dtype="float64")
    b = np.asarray(second, dtype="float64")
    rank_a = np.argsort(np.argsort(a)).astype("float64")
    rank_b = np.argsort(np.argsort(b)).astype("float64")
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    if denominator == 0:
        return None
    return float((rank_a * rank_b).sum() / denominator)


def final_ers() -> Dict[str, float]:
    """The frozen ER of every arm the controller has measured, from the audit log."""
    if not AUDIT.is_file():
        return {}
    out: Dict[str, float] = {}
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "evaluate" and event.get("er") is not None:
            out[event["mechanism_id"]] = float(event["er"])
    return out


def measure() -> Dict[str, Any]:
    """Compare every arm's probe export with its cold run's export."""
    if not CONTROL_SID.is_file():
        return {"ok": False, "reason": f"the archived control's export is missing: {CONTROL_SID}"}
    control = balance(CONTROL_SID)
    ers = final_ers()
    parquet = TRAIN_PARQUET
    if not parquet.is_file():
        found = sorted((REPO_ROOT / "results").rglob("train.parquet"))
        if not found:
            return {"ok": False, "reason": "the training parquet is missing"}
        parquet = found[0]
    sources, targets = train_pairs(parquet, limit=200000)

    rows: List[Dict[str, Any]] = []
    for probe in sorted(ROUNDS.glob("round_*/directed_*/probe/redirected/results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy")):
        arm = probe.parents[5].name
        baseline = (
            probe.parents[4]
            / "baseline"
            / "redirected"
            / "results"
            / "stage2_RQ-VAE"
            / "r8_control"
            / "sids_for_hgrec.npy"
        )
        cold = STAGE2 / f"TIGER_c107_{arm}" / "r8_control" / "sids_for_hgrec.npy"
        if not baseline.is_file() or not cold.is_file():
            continue
        probe_balance = balance(probe)
        baseline_balance = balance(baseline)
        cold_balance = balance(cold)
        probe_discrimination = train_side_discrimination(load_sids(probe), sources, targets)
        baseline_discrimination = train_side_discrimination(load_sids(baseline), sources, targets)
        probe_ratio = (
            probe_discrimination["discrimination"] / baseline_discrimination["discrimination"]
            if probe_discrimination["discrimination"] and baseline_discrimination["discrimination"]
            else None
        )
        rows.append(
            {
                "arm": arm,
                "probe_gini_delta": probe_balance["gini"] - baseline_balance["gini"],
                "cold_gini_delta": cold_balance["gini"] - control["gini"],
                "probe_distinct_delta": probe_balance["distinct_codes"] - baseline_balance["distinct_codes"],
                "cold_distinct_delta": cold_balance["distinct_codes"] - control["distinct_codes"],
                "probe_er_ratio": probe_ratio,
                "final_er": ers.get(arm),
            }
        )

    paired_gini = [row["probe_gini_delta"] for row in rows]
    cold_gini = [row["cold_gini_delta"] for row in rows]
    paired_distinct = [row["probe_distinct_delta"] for row in rows]
    cold_distinct = [row["cold_distinct_delta"] for row in rows]
    paired_er = [
        (row["probe_er_ratio"], row["final_er"])
        for row in rows
        if row["probe_er_ratio"] is not None and row["final_er"] is not None
    ]
    probe_ratio = [pair[0] for pair in paired_er]
    final_er = [pair[1] for pair in paired_er]
    return {
        "ok": True,
        "arms": len(rows),
        "control": control,
        "spearman_probe_vs_cold_gini": spearman(paired_gini, cold_gini),
        "spearman_probe_vs_cold_distinct_codes": spearman(paired_distinct, cold_distinct),
        "spearman_probe_er_ratio_vs_final_er": spearman(probe_ratio, final_er),
        "rows": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.genrec_agents.probe_fidelity")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = measure()
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
