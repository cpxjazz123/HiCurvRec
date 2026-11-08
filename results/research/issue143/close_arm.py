"""Close one Stage 2-only arm: recover its run, write the record, write the conclusion.

The shared-metrics fault of 2026-11-08 (a duplicate run truncating the collector
and the earlier run appending its tail) is repaired first: the delivered run is
the one run in the file whose 1..3000 train sequence is complete, and it is
written to ``logs/run_metrics.jsonl`` with a per-run audit. Then the mandatory
layered SID record and the conclusion are written from that stream.

    python3 results/research/issue143/close_arm.py M4_post_sid_af3
    python3 results/research/issue143/close_arm.py M1_learned_kappa --no-conclude
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TREE = REPO / "stage2_RQ-VAE"
sys.path.insert(0, str(REPO))

from tools.research.stage2 import run_metrics  # noqa: E402

ARMS = ("M1_learned_kappa", "M2_radial_manifold", "M3_role_profiles",
        "M4_post_sid_af3")
EPOCHS = 3000


def recover(arm: str) -> dict:
    logs = TREE / ("TIGER_" + arm) / "logs"
    stream = logs / "training_metrics.jsonl"
    if not stream.is_file():
        raise SystemExit("%s has no metrics stream" % arm)
    split = run_metrics.write_run_stream(stream, logs / "run_metrics.jsonl", EPOCHS)
    audit = {"source": str(stream), "epochs": EPOCHS,
             "runs": {str(start): body for start, body in
                      sorted(run_metrics.summarize(stream).items())},
             "delivered": split}
    (logs / "run_split.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--no-conclude", action="store_true")
    args = parser.parse_args()
    audit = recover(args.arm)
    print(json.dumps({"arm": args.arm, "delivered": audit["delivered"],
                      "runs": audit["runs"]}, indent=2, sort_keys=True))
    done = subprocess.run([sys.executable, str(HERE / "finish.py"), args.arm],
                          cwd=str(REPO), capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("finish failed: %s" % (done.stderr or done.stdout)[-800:])
    if not args.no_conclude:
        written = subprocess.run([sys.executable, str(HERE / "conclude.py"), args.arm],
                                 cwd=str(REPO), capture_output=True, text=True)
        if written.returncode != 0:
            raise SystemExit("conclude failed: %s" % (written.stderr or written.stdout)[-800:])
        print(written.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
