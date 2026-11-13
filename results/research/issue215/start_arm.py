"""Start one issue-215 arm on one card, through the framework's own Stage 2 entry point.

    python3 results/research/issue215/start_arm.py 0 TIGER_role_geom_rq

The card is pinned through ``CUDA_VISIBLE_DEVICES`` before the server module is imported, so the
framework probes the layout the way it does for every other arm: one visible card means the
trainer's single-card branch, which keeps the global batch at 4096 and the frozen 3000 epochs and
the frozen seed.  That is the parallel layout the owner's rule asks for -- several arms at once,
each on its own card -- and it is the same code path the framework uses when it spreads a round's
jobs over the cards.

Nothing here decides anything: the framework's smoke runs first and no training starts unless it
passes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOGICAL_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def load_server():
    spec = importlib.util.spec_from_file_location("genrec_server", REPO / "tools" / "genrec-mcp" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    card, tree_name = sys.argv[1], sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = card
    server = load_server()
    tree = LOGICAL_ROOT / "stage2_RQ-VAE" / tree_name
    outcome = server.tool_stage2_start(str(tree))
    print(json.dumps({"card": card, "tree": tree_name,
                      "started": outcome.get("started"),
                      "refused": outcome.get("refused"),
                      "status": outcome.get("status"),
                      "result_dir": outcome.get("result_dir"),
                      "worker_pid": outcome.get("worker_pid"),
                      "gpu_probe": outcome.get("gpu_probe")}, indent=1, default=str))
    return 0 if outcome.get("started") or outcome.get("status") == "smoke_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
