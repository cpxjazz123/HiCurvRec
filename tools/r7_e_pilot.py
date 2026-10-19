"""Track E pilot (issue #84): do real deep updates cross L1 boundaries, and can projection stop them?

Two questions, answered separately because they have different answers.

**Q1, measurement.** With control parity established, the question issue #79 could not answer is
now answerable: when a real optimizer step is taken, how many items actually change their L1 code,
and how does that change when the deep term is removed? The margins are recomputed through the
quantizer before and after each branch; nothing is inferred from a gradient difference.

**Q2, the guarantee.** A margin constraint stated on the gradient is NOT a constraint on the
parameter update, because AdamW applies exponential moments, weight decay and clipping. This pilot
measures that rather than asserting it: it reports the realised violation rate after the real
update, which is the only quantity a guarantee can honestly be about.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_replay_engine as G  # noqa: E402
import r7_e_voronoi_projection as E  # noqa: E402

ARMS = ("r6_02", "r6_03", "r6_04", "r4_08")
STEPS = (96, 160)
BUNDLE_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r7_a_parity"
OUTPUT = REPO_ROOT / "results/stage2_RQ-VAE/r7_e_voronoi"
DELTA = 0.0


def branch_margin(branch, bundle) -> dict:
    return E.exact_margins(branch["model"], bundle["item_batch"].to(G.DEVICE))


def crossing(before: dict, after: dict, *, delta: float = DELTA) -> dict:
    """Items whose L1 winner identity changed, and items that fell below the guaranteed band."""
    changed = before["winner"] != after["winner"]
    below = after["margin"] < delta
    return {
        "items": int(before["margin"].numel()),
        "l1_identity_changed": int(changed.sum()),
        "l1_change_rate": float(changed.float().mean()),
        "below_delta_after": int(below.sum()),
        "below_delta_rate": float(below.float().mean()),
        "margin_before_mean": float(before["margin"].mean()),
        "margin_after_mean": float(after["margin"].mean()),
        "margin_after_min": float(after["margin"].min()),
    }


def run_arm(arm: str) -> dict:
    started = time.time()
    tree = REPO_ROOT / f"stage2_RQ-VAE/TIGER_{arm}"
    report: dict = {"arm": arm, "kind": "measurement_plus_mechanism_probe", "steps": {},
                    "delta": DELTA, "blockers": []}
    for step in STEPS:
        path = BUNDLE_ROOT / arm / f"step_{step:04d}" / "bundle.pt"
        if not path.is_file():
            report["blockers"].append(f"no captured bundle at step {step}")
            continue
        bundle = torch.load(path, weights_only=False)
        entry: dict = {}
        states = {}
        for intervention in ("FULL", "NO-DEEP", "BASE-ONLY"):
            branch = G.build_frozen_branch(tree, bundle,
                                            module_name=f"r7e_{arm}_{step}_{intervention}")
            before = branch_margin(branch, bundle)
            result = G.assemble_and_step(branch, bundle, intervention=intervention)
            after = branch_margin(branch, bundle)
            states[intervention] = before
            entry[intervention] = {"crossing": crossing(before, after),
                                   "update_sha256": result["update_sha256"],
                                   "no_learning_signal": result["no_learning_signal"]}
            G.release_branch(branch)
        full, nodeep = states["FULL"], states["NO-DEEP"]
        entry["deep_attributable_crossings"] = {
            "definition": "items that change their L1 code under FULL but not under NO-DEEP, which "
                          "is the only attribution a same-state counterfactual supports",
            "full_changed": entry["FULL"]["crossing"]["l1_identity_changed"],
            "no_deep_changed": entry["NO-DEEP"]["crossing"]["l1_identity_changed"],
            "delta": (entry["FULL"]["crossing"]["l1_identity_changed"]
                      - entry["NO-DEEP"]["crossing"]["l1_identity_changed"]),
        }
        report["steps"][str(step)] = entry
    report["status"] = "supported" if report["steps"] and not report["blockers"] else "blocked"
    report["elapsed_s"] = round(time.time() - started, 2)
    return report


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for arm in ARMS:
        print(f"=== {arm} ===", flush=True)
        report = run_arm(arm)
        reports.append(report)
        (OUTPUT / arm).mkdir(parents=True, exist_ok=True)
        (OUTPUT / arm / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"{arm}: {report['status']}", flush=True)
    summary = {
        "arms": reports,
        "all_supported": all(report["status"] == "supported" for report in reports),
        "guarantee_status": "NOT ESTABLISHED",
        "guarantee_reason": "a constraint stated on the gradient is not a constraint on the "
                            "parameter update: AdamW applies exponential moments, weight decay and "
                            "clipping, so the realised step is not a scalar multiple of the "
                            "constrained gradient. This pilot measures the realised crossings "
                            "instead of claiming a gradient-side guarantee.",
    }
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
