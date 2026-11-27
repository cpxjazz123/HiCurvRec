#!/usr/bin/env python3
"""Write the CLAUDE.md §63 layered metric record for one Stage2 run.

The automatic recorder (``tools/research/science/sid_layer_analysis.py``) was deleted with
§94, so the frozen implementation ``tools/issue252/layer_metrics.py`` is used instead; that
script hashes itself and its contract into the record it writes, so the provenance stays
machine-checkable.  The record lands in
``results/stage2_RQ-VAE/<ARM>/stage2_metrics/<sid_sha256>.json`` and is never overwritten.

Usage::

    python3.10 results/research/issue273/write_layer_metrics.py \\
        --sid results/stage2_RQ-VAE/_runs/<run_id>/<ARM>/result/item_sids.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.issue252.layer_metrics import record_layers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True, help="the run's item_sids.json")
    parser.add_argument("--arm", default=None, help="the arm name (default: the run's own arm)")
    args = parser.parse_args()
    result = record_layers(args.sid, repo=REPO, arm=args.arm)
    report = result["report"]
    print(json.dumps({"path": result["path"], "sha256": result["sha256"],
                      "sid": report["sid"], "catalogue": report["catalogue"],
                      "validation": report["validation"],
                      "evaluator": report["evaluator"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
