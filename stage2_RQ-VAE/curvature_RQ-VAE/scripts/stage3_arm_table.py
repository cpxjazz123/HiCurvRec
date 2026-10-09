"""Every Stage3 arm this program has run, in one table.

Reads each arm's native ``test_final.json`` next to its metrics file, so the
numbers are the evaluator's own and not a transcription. Stage2 settings come
from the arm's ``train_start`` event, which records the code path it consumed,
and the held-out containment reading comes from the last ``cone_containment``
event when the arm had a cone at all.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS = Path("/home/wlia0047/ar57/wenyu/GeneRec/results")
STAGE3 = RESULTS / "stage3_T5Train"


def _last_event(path: Path, event: str) -> dict:
    found: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == event:
            found = row
    return found


def main() -> None:
    rows = []
    for metrics in sorted(glob.glob(str(STAGE3 / "sidarm_*" / "logs" / "*" / "*" / "training_metrics.jsonl"))):
        path = Path(metrics)
        variant = path.parts[-5]
        start = _last_event(path, "train_start")
        test = _last_event(path, "test")
        containment = _last_event(path, "cone_containment")
        if not test:
            continue
        held = containment.get("heldout", {}) if containment else {}
        rows.append(
            {
                "variant": variant,
                "code_path": start.get("code_path", "").split("curvature_RQ-VAE/")[-1],
                "r10": test.get("test_recall_at_10"),
                "ndcg10": test.get("test_ndcg_at_10"),
                "n_eval": test.get("n_eval"),
                "heldout_pos": held.get("coarse_q2_containment"),
                "heldout_neg": held.get("coarse_q2_negative_containment"),
            }
        )
    rows.sort(key=lambda row: -(row["r10"] or 0))
    print(f"{'variant':<44} {'R@10':>9} {'NDCG@10':>9} {'n_eval':>7} {'pos':>7} {'neg':>7}  code path")
    for row in rows:
        pos = f"{row['heldout_pos']:.4f}" if row["heldout_pos"] is not None else "   -  "
        neg = f"{row['heldout_neg']:.4f}" if row["heldout_neg"] is not None else "   -  "
        print(
            f"{row['variant']:<44} {row['r10']:>9.6f} {row['ndcg10']:>9.6f} "
            f"{row['n_eval']:>7} {pos:>7} {neg:>7}  {row['code_path']}"
        )
    print(f"\n{len(rows)} arms; the bar is the euclidean no-cone arm at 0.059158")


if __name__ == "__main__":
    main()