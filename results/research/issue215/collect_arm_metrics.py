"""Collect each arm's SID evidence: the layer metrics and the three-code collision.

    python3 results/research/issue215/collect_arm_metrics.py

For every arm of issue #215 this records the §63 layer metrics (through the framework's own
recorder) and computes the §89 gate metric -- the collision of the FIRST THREE codes, which is
what decides whether the arm may enter Stage 3. The fourth token of the exported table is the
RecBole collision extension and is not part of the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.research.science.sid_layer_analysis import record_single_sid  # noqa: E402

ARMS = ["TIGER_role_geom_rq", "TIGER_role_geom_global_rq", "TIGER_role_geom_shuffled_rq",
        "TIGER_mixed_rq_F0"]
OUT = REPO / "results" / "research" / "issue215" / "arm_metrics.json"


def collision_of_first_three(path: Path) -> dict:
    payload = json.loads(path.read_text())
    triples = [tuple(int(value) for value in row[:3]) for row in payload.values()]
    items = len(triples)
    distinct = len(set(triples))
    return {"items": items, "distinct_first_three": distinct,
            "collision": 1.0 - distinct / max(items, 1),
            "distinct_four": len({tuple(int(v) for v in row[:4]) for row in payload.values()})}


def main() -> int:
    report = {}
    for arm in ARMS:
        sid_path = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
        if not sid_path.is_file():
            print(f"{arm}: no SID yet ({sid_path})", flush=True)
            continue
        block = {"sid_path": str(sid_path.relative_to(REPO))}
        block.update(collision_of_first_three(sid_path))
        try:
            recorded = record_single_sid(sid_path)
            block["layer_metrics"] = str(recorded.relative_to(REPO))
        except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
            block["layer_metrics_error"] = f"{type(exc).__name__}: {exc}"
        report[arm] = block
        print(f"{arm}: items {block['items']} | distinct(first 3) {block['distinct_first_three']} "
              f"| collision {block['collision']:.6f} | gate(<0.40) "
              f"{'PASS' if block['collision'] < 0.40 else 'FAIL'}", flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(f"written: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
