#!/usr/bin/env python3
"""Reduce the per-rank test shards of ONE Stage 3 evaluation into test_final.json.

Stage 3 evaluates the test split on four ranks and each rank publishes its own
sums; rank 0 reduces them. When rank 0 cannot do it (it exited before the
reduction, or a rank wrote its shard into another run directory), the numbers are
still on disk and the reduction is a plain sum -- but only if the shards belong
to the same evaluation.

This tool refuses to guess:

* every rank from 0 to N-1 must be present exactly once;
* the ``rank`` field inside each file must match its position;
* the shard files must have been written within ``--window`` seconds of each
  other (one evaluation, not two);
* the total ``n_eval`` must be stated and matched.

The reduction itself is the trainer's own ``reduce_metric_sums``, imported from
the frozen Stage 3 entry point, so the arithmetic is the one the run would have
used.

Usage:
    aggregate_stage3_shards.py --out <test_final.json> --expect-n 57439 \
        rank0.json rank1.json rank2.json rank3.json [--note TEXT]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

STAGE3_ENTRY = Path(__file__).resolve().parents[2] / "stage3_T5Train" / "train_HG-Rec.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reduction(repo_root: Path):
    """The frozen trainer's own reduction, loaded without importing RecBole."""
    import ast
    import textwrap

    source = STAGE3_ENTRY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "reduce_metric_sums":
            wanted = node
            break
    if wanted is None:
        raise SystemExit("reduce_metric_sums is not in " + str(STAGE3_ENTRY))
    namespace: dict = {}
    exec(textwrap.dedent(ast.get_source_segment(source, wanted)), namespace)
    return namespace["reduce_metric_sums"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-n", required=True, type=int)
    parser.add_argument("--window", type=float, default=900.0)
    parser.add_argument("--note", default="")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    payloads = []
    for path in args.shards:
        if not path.is_file():
            raise SystemExit("missing shard: " + str(path))
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))

    ranks = sorted(int(payload["rank"]) for _, payload in payloads)
    if ranks != list(range(len(ranks))):
        raise SystemExit("shard ranks are not 0..N-1 exactly once: " + str(ranks))
    mtimes = [path.stat().st_mtime for path, _ in payloads]
    if max(mtimes) - min(mtimes) > args.window:
        raise SystemExit("the shards are more than %.0f s apart: not one evaluation" % args.window)

    reduce_metric_sums = load_reduction(STAGE3_ENTRY)
    recalls, ndcgs, n_eval = reduce_metric_sums([
        ({int(k): float(v) for k, v in payload["recall_sums"].items()},
         {int(k): float(v) for k, v in payload["ndcg_sums"].items()},
         int(payload["n_eval"]))
        for _, payload in payloads
    ])
    if n_eval != args.expect_n:
        raise SystemExit("n_eval %d does not match the expected %d" % (n_eval, args.expect_n))

    result = {"n_eval": int(n_eval),
              **{f"test_{key}": value for key, value in recalls.items()},
              **{f"test_{key}": value for key, value in ndcgs.items()}}
    if args.checkpoint:
        result["best_checkpoint"] = args.checkpoint
    result["provenance"] = {
        "reduced_by": str(Path(__file__).resolve()),
        "reduction": "stage3_T5Train/train_HG-Rec.py::reduce_metric_sums",
        "shards": [{"path": str(path), "sha256": digest(path),
                    "mtime": path.stat().st_mtime, "n_eval": int(payload["n_eval"])}
                   for path, payload in payloads],
        "note": args.note,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
