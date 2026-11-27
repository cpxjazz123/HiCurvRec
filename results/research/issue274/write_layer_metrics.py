#!/usr/bin/env python3
"""Issue #274: write the CLAUDE.md §63 layered metric record for one Stage2 run.

The automatic recorder (``tools/research/science/sid_layer_analysis.py``) was deleted with §94, so
the frozen implementation ``tools/issue252/layer_metrics.py`` is used instead.

Two paths exist:

* when an ``archive_binding.json`` sits next to the SID (a checkpoint that carries the dispatch
  provenance), ``record_layers`` writes the full record with the binding receipt;
* otherwise (the frozen baseline's checkpoint format carries no ``run_id``/``provenance``) the same
  frozen functions — ``load_sid``, ``catalogue_metrics``, ``validation_metrics`` — compute exactly
  the same metric set, the record notes the missing binding explicitly, and nothing is guessed.

Usage::

    python3.10 results/research/issue274/write_layer_metrics.py --sid <run>/item_sids.json --arm <ARM>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.issue252 import layer_metrics as lm  # noqa: E402
from tools.issue252.archive_audit import sha256  # noqa: E402
from tools.issue252.final_evaluation import load_sid  # noqa: E402
from tools.issue252.layer_metrics import record_layers  # noqa: E402

ITEMS = 24587
CODE_SIZES = (256, 256, 256)


def compute_without_binding(sid_path: Path, arm: str) -> dict:
    import pyarrow.parquet as pq

    sid = Path(sid_path).resolve()
    rows = load_sid(sid, ITEMS, CODE_SIZES)
    valid = REPO / "results/stage0_build_parquet/valid.parquet"
    examples = pq.read_table(valid, columns=["history", "seen_history", "target"]).to_pylist()
    catalogue = lm.catalogue_metrics(rows)
    validation = lm.validation_metrics(rows, examples)
    payload = {
        "schema": "genrec.issue252.layer_metrics/1",
        "arm": arm,
        "sid": {"path": str(sid), "sha256": sha256(sid)},
        "inputs": {"valid": {"path": str(valid), "sha256": sha256(valid)}},
        "catalogue": catalogue,
        "validation": validation,
        "binding": "MISSING (the checkpoint carries no dispatch provenance; the metric set is "
                   "computed by the same frozen implementation)",
        "evaluator": {"path": str(Path(lm.__file__).resolve()), "sha256": sha256(lm.__file__)},
        "read_stage3_scores": False,
    }
    output = REPO / "results/stage2_RQ-VAE" / arm / "stage2_metrics" / (payload["sid"]["sha256"] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"
    if output.exists():
        if output.read_text() != text:
            raise ValueError("refusing to overwrite a different historical layered metric record")
    else:
        output.write_text(text)
    return {"path": str(output), "sha256": sha256(output), "report": payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True, help="the run's item_sids.json")
    parser.add_argument("--arm", default=None, help="the arm name (default: the run's own arm)")
    args = parser.parse_args()
    sid = Path(args.sid)
    arm = args.arm or sid.parents[1].name
    if sid.with_name("archive_binding.json").is_file():
        result = record_layers(str(sid), repo=REPO, arm=arm)
    else:
        result = compute_without_binding(sid, arm)
    report = result["report"]
    print(json.dumps({"path": result["path"], "sha256": result["sha256"],
                      "binding": report.get("binding", "present"),
                      "sid": report["sid"], "catalogue": report["catalogue"],
                      "validation": {k: v for k, v in report["validation"].items()
                                     if "hit_rate" in k or k in ("effective_rows", "rows")},
                      "evaluator": report["evaluator"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
