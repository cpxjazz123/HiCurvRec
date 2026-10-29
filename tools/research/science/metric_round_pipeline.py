#!/usr/bin/env python3
"""One command that reproduces the metric-round result end to end.

    python3 tools/research/science/metric_round_pipeline.py --all

does, in order:

1. **collect** the confirmed SID+recall pairs (the collector, which pairs a Stage 3
   run with the SID table it read and checks the table against the vocabulary the
   run itself recorded);
2. **score** the metrics of a module on every one of those pairs;
3. **report** the Spearman correlation of each metric with ``test_recall@10``, its
   bootstrap interval, and whether it passes the selection rule fixed before any
   number was seen.

Nothing is trained: no Stage 2, no Stage 3, no GPU. The reference is Issue
#126's frozen behaviour model and the SID tables are the ones already exported.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COLLECTOR = REPO / "analysis_sid_recall" / "collect.py"
BENCH = REPO / "tools" / "research" / "science" / "sid_metric_bench.py"
DEFAULT_MODULE = REPO / "results" / "research" / "issue126" / "metric_round" / "metrics_all_fast.py"
DEFAULT_OUT = REPO / "results" / "research" / "issue126" / "metric_round" / "pipeline_report.json"


def run(command: list, label: str) -> None:
    print(f"\n=== {label}\n$ {' '.join(str(part) for part in command)}", flush=True)
    done = subprocess.run([str(part) for part in command], cwd=str(REPO))
    if done.returncode != 0:
        raise SystemExit(f"{label} failed with {done.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=str(DEFAULT_MODULE),
                        help="a python file defining METRICS = {name: fn}")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--collect", action="store_true",
                        help="re-collect the pairs first (the collector rewrites "
                             "analysis_sid_recall/)")
    parser.add_argument("--all", action="store_true",
                        help="collect, score and report in one go")
    args = parser.parse_args()

    if args.collect or args.all:
        run([sys.executable, COLLECTOR, "--paired-only"], "collect the confirmed pairs")

    pairs = REPO / "analysis_sid_recall" / "sid_recall.csv"
    if not pairs.is_file():
        raise SystemExit("no collection: run with --collect first")
    run([sys.executable, BENCH, "--module", args.module, "--out", args.out,
         "--workers", str(args.workers)], "score the metrics on every confirmed pair")

    report = json.loads(Path(args.out).read_text())
    print("\n=== the result")
    print(f"pairs: {report['pairs']}")
    for result in sorted(report["results"], key=lambda item: -item["spearman"]):
        interval = result["bootstrap_ci95"]
        print(f"  {result['metric']:28s} rho {result['spearman']:+.4f} "
              f"p {result['p']:.3g} CI [{interval[0]:+.3f}, {interval[1]:+.3f}] "
              f"{'ELIGIBLE' if result['eligible'] else 'not eligible'}")
    print(f"\nchosen by the fixed rule: {report['chosen']}")
    out = Path(args.out).resolve()
    try:
        shown = out.relative_to(REPO.resolve())
    except ValueError:
        shown = out
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
