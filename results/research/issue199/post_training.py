"""Issue #194 acceptance 6, second half: the export's statistics and the §89 gate, computed the way the gate itself computes them.

The gate in ``tools/genrec-mcp/server.py`` counts ``1 - distinct(first three codes) / items`` over ``item_sids.json`` and refuses at 0.40.  #63's layered record
is a different statistic and is produced next to it, never instead of it, by ``tools/research/science/sid_layer_analysis.record_single_sid``.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARM = "TIGER_l2k_rq"
SID = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM / "item_sids.json"


def main() -> int:
    payload = json.loads(SID.read_text())
    items = len(payload)
    triples = {tuple(value[:3]) for value in payload.values()}
    gate = 1.0 - len(triples) / max(items, 1)

    level1 = collections.Counter(value[0] for value in payload.values())
    level2 = collections.Counter(tuple(value[:2]) for value in payload.values())
    level3 = collections.Counter(tuple(value[:3]) for value in payload.values())
    report = {
        "arm": ARM,
        "sid": str(SID.relative_to(REPO_ROOT)),
        "items": items,
        "row_widths": sorted({len(v) for v in payload.values()}),
        "gate_collision": gate,
        "gate_refuses_stage3": gate >= 0.40,
        "distinct_triples": len(triples),
        "distinct_prefixes_l2": len(level2),
        "l1_codes_used": len(level1),
        "l1_largest_share": max(level1.values()) / items,
        "l3_largest_share": max(level3.values()) / items,
        "l3_entropy_share_of_max": None,
    }
    import math
    total = sum(level3.values())
    entropy = -sum((c / total) * math.log(c / total, 2) for c in level3.values())
    report["l3_entropy_share_of_max"] = entropy / math.log(min(len(level3), total), 2) if len(level3) > 1 else 0.0
    (HERE / "post_training_export.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    from tools.research.science.sid_layer_analysis import record_single_sid
    path = record_single_sid(sid_path=SID)
    print("layered record:", path)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
