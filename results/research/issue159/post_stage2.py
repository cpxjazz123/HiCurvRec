"""The post-Stage2 steps of issue #159, runnable the moment the run finishes.

Records the section 63 layer metrics first, then reports the entry collision (first three codes)
and the section 63 l3_collision side by side, and states whether Stage3 is admissible.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARM = "TIGER_plan_rounded_rq"
SID = ROOT / "results" / "stage2_RQ-VAE" / ARM / "item_sids.json"


def main() -> int:
    if not SID.exists():
        print(f"no export yet at {SID}")
        return 1
    sys.path.insert(0, str(ROOT))
    from tools.research.science.sid_layer_analysis import record_single_sid
    report_path = record_single_sid(SID)
    report = json.loads(Path(report_path).read_text())
    rows = {int(key): value for key, value in json.loads(SID.read_text()).items()}
    triples = Counter(tuple(value[:3]) for value in rows.values())
    entry_collision = 1.0 - len(triples) / len(rows)
    widest = max(triples.values()) / len(rows)
    payload = {
        "items": len(rows),
        "distinct_three_token_sids": len(triples),
        "entry_collision": entry_collision,
        "largest_share": widest,
        "l3_collision_section63": report["diagnostics"]["l3_collision"],
        "level2_distinct_prefixes": report["diagnostics"]["level2_distinct_prefixes"],
        "level3_distinct_triples": report["diagnostics"]["level3_distinct_triples"],
        "valid_l1_hard_er": report["diagnostics"]["valid_l1_hard_er"],
        "l3_refinement_gain_bits": report["diagnostics"]["l3_refinement_gain_bits"],
        "conditional_l3_gain": report["diagnostics"]["conditional_l3_gain"],
        "layer_report": str(report_path),
        "stage3_admissible": bool(entry_collision < 0.40),
    }
    out = ROOT / "results" / "research" / "issue159" / "stage2_result.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
