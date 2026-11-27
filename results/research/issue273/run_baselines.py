#!/usr/bin/env python3
"""Issue #273 round 1: measure the frozen baselines on the within-centre statistics.

The C6/C7 thresholds are "below/above the #269 value", so the #269 baseline (and the TIGER and
#271 references for the other thresholds) must be measured **before** the formal run.  The runs go
through the public persistent interface ``tools/iteration/execution.py::run`` (command.json, log,
exit code, duration).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/iteration"))
import execution  # noqa: E402

PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [
    ("TIGER_RQ-VAE",
     "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
     "results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth",
     "results/research/issue271/assets/geometry/TIGER_RQ-VAE.json"),
    ("TIGER_calibrated_kappa_rq",
     "results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/item_sids.json",
     "results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/rqvae_best.pth",
     "results/research/issue271/assets/geometry/TIGER_calibrated_kappa_rq.json"),
    ("TIGER_global_j_rq",
     "results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result/item_sids.json",
     "results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result/rqvae_best.pth",
     "results/research/issue271/assets/geometry/TIGER_global_j_rq.json"),
]


def main() -> None:
    out_dir = REPO / "results/research/issue273/executions"
    summaries = []
    for arm, sid, checkpoint, reference in ARMS:
        argv = [PY, "results/research/issue273/within_centre_metrics.py",
                "--arm", arm, "--sid", sid, "--checkpoint", checkpoint,
                "--out", f"results/research/issue273/within_centre_{arm}.json",
                "--reference", reference]
        result = execution.run(f"issue273_within_centre_{arm}", argv, out_dir=out_dir, cwd=REPO,
                               timeout=3600, note="issue273 round-1 frozen baseline")
        print(arm, result.status, result.exit_code, result.duration_s, result.log, flush=True)
        summaries.append({"arm": arm, "status": result.status, "exit_code": result.exit_code,
                          "duration_s": result.duration_s, "log": result.log})
    (REPO / "results/research/issue273/baseline_runs.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
