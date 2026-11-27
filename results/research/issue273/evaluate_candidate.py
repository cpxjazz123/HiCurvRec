#!/usr/bin/env python3
"""Evaluate one Stage2 candidate on the frozen issue-272/#273 criteria and emit a PASS table.

The evaluation reuses only frozen components:

* the common-ruler global J and the 1-hop/2-hop strata from the candidate's own
  ``model/global_constraint.py`` (the unchanged #269/#271 module) via ``gc.full_j``;
* the SID structure metrics from ``results/research/issue271/collect_metrics.py``;
* the local geometry and the within-centre statistics from
  ``results/research/issue273/within_centre_metrics.py`` (same assets/scale/algorithm as the
  frozen #271 evaluator; the self-check reproduces the frozen MAE/Spearman/sign).

Usage::

    python3 evaluate_candidate.py --arm NAME --tree stage2_RQ-VAE/<ARM> --result-dir <run result dir> \\
        --out-prefix results/research/issue273/<NAME>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.iteration import execution  # noqa: E402

PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

# Frozen thresholds (see results/research/issue273/RUN_LOG.md, the C-list).
P1_J = 0.338335882630658
P10_1HOP = 0.798295
P11_2HOP = 0.278106
P1_MAE = 1.8935063111536612
P2_SPEARMAN = 0.015850898303185326
P3_SIGN = 0.4324533490974272
P12_L2 = 10918
P13_RANK = 8
P4_RECON = 8.589e-5


def run_check(unit: str, argv: list[str], out_dir: Path, note: str):
    result = execution.run(unit, argv, out_dir=out_dir, cwd=REPO, timeout=3600, note=note)
    print(unit, result.status, result.exit_code, result.duration_s, result.log, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--baseline", default=None,
                        help="within-centre baseline json for the C6/C7 comparison")
    args = parser.parse_args()

    tree = REPO / args.tree
    result_dir = REPO / args.result_dir
    out_dir = REPO / "results/research/issue273/executions"

    sys.path.insert(0, str(tree))
    from model import global_constraint as gc  # noqa: E402

    q = gc.load_hard_q(result_dir / "item_sids.json", result_dir / "rqvae_best.pth")
    full_j = gc.full_j(q, gc.load_graph())

    sid_metrics = run_check(
        f"{args.arm}_sid_metrics",
        [PY, "results/research/issue271/collect_metrics.py", str(result_dir)],
        out_dir, "frozen issue-271 SID collector")
    local = run_check(
        f"{args.arm}_within_centre",
        [PY, "results/research/issue273/within_centre_metrics.py", "--arm", args.arm,
         "--sid", str(result_dir / "item_sids.json"),
         "--checkpoint", str(result_dir / "rqvae_best.pth"),
         "--out", f"{args.out_prefix}_within_centre.json"],
        out_dir, "issue-273 within-centre metrics")

    sid_payload = json.loads((result_dir / "issue271_sid_metrics.json").read_text())
    local_payload = json.loads(Path(f"{args.out_prefix}_within_centre.json").read_text())

    metrics_path = result_dir / "logs" / "training_metrics.jsonl"
    final = None
    mu_epochs = 0
    epochs = 0
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            final = record
            epochs += 1
            if float(record.get("global_mu", 0.0)) > 0:
                mu_epochs += 1
    recon = float(final["recon"]) if final else None

    strata = full_j["strata"]
    one_hop = float(strata["1-1"]["mean_j"])
    two_hop = float(strata["2-2"]["mean_j"])
    l2 = int(sid_payload["unique_two"])
    rank = float(sid_payload.get("sid_quality", {}).get("effective_rank", float("nan")))

    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    table = {
        "schema": "genrec.issue273.pass_table/1",
        "arm": args.arm,
        "C1_local_mae": {"value": local_payload["mean_abs_error"], "threshold": P1_MAE,
                          "pass": local_payload["mean_abs_error"] <= P1_MAE},
        "C2_spearman": {"value": local_payload["spearman"], "threshold": P2_SPEARMAN,
                        "pass": local_payload["spearman"] > P2_SPEARMAN},
        "C3_sign": {"value": local_payload["sign_accuracy"], "threshold": P3_SIGN,
                    "pass": local_payload["sign_accuracy"] > P3_SIGN},
        "C4_variance_ratio": {"value": local_payload["variance_ratio"], "threshold": 0.5,
                              "pass": local_payload["variance_ratio"] >= 0.5},
        "C5_slope": {"value": local_payload["slope"], "threshold": 0.1,
                     "pass": local_payload["slope"] >= 0.1},
        "C6_difference_rmse": {"value": local_payload["difference_rmse"],
                               "baseline": baseline["difference_rmse"] if baseline else None,
                               "pass": None if not baseline else
                               local_payload["difference_rmse"] < baseline["difference_rmse"]},
        "C7_within_centre_ordering": {"value": local_payload["within_centre_spearman"],
                                       "baseline": baseline["within_centre_spearman"] if baseline else None,
                                       "pass": None if not baseline else
                                       local_payload["within_centre_spearman"] > baseline["within_centre_spearman"]},
        "C9_full_j": {"value": float(full_j["mean_j"]), "threshold": P1_J,
                      "pass": float(full_j["mean_j"]) <= P1_J},
        "C10_one_hop_j": {"value": one_hop, "threshold": P10_1HOP, "pass": one_hop <= P10_1HOP},
        "C11_two_hop_j": {"value": two_hop, "threshold": P11_2HOP, "pass": two_hop <= P11_2HOP},
        "C12_l2_prefixes": {"value": l2, "threshold": P12_L2, "pass": l2 >= P12_L2},
        "C13_effective_rank": {"value": rank, "threshold": P13_RANK, "pass": rank >= P13_RANK},
        "C15_reconstruction": {"value": recon, "threshold": P4_RECON,
                               "pass": None if recon is None else recon <= P4_RECON},
        "collision_recorded": sid_payload["collision"],
        "mu_active_epochs": {"value": mu_epochs, "epochs": epochs,
                             "fraction": mu_epochs / max(epochs, 1)},
        "sid_metrics": sid_payload,
        "local_payload": local_payload,
        "full_j": full_j,
        "executions": {"sid_metrics": sid_metrics.evidence(), "within_centre": local.evidence()},
    }
    table["all_pass"] = all(v.get("pass") for k, v in table.items()
                            if isinstance(v, dict) and v.get("pass") is not None
                            and k.startswith("C") and k not in ("C16",))
    Path(f"{args.out_prefix}_pass_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in table.items() if k.startswith("C")}, indent=2, default=str))
    print("all_pass", table["all_pass"])


if __name__ == "__main__":
    main()
