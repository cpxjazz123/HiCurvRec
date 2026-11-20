"""Does any SID depth carry the directed orbit? (issue #221)

L1 alone sits below the majority baseline.  The code has three levels; this
checks L1, L1+L2 and the full three-token code on the same capped pool, so a
negative result is not an artefact of stopping at the first level.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directed_role import load_pool, load_vocabulary  # noqa: E402
from evaluate import load_sids, role_probe  # noqa: E402

REPO = HERE.parents[2]
ARMS = ("D0", "D1")
DEPTHS = (1, 2, 3)
CAP = 20000


def main() -> None:
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab, max_instances=CAP)
    panel = {}
    for arm in ARMS:
        path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}" / "item_sids.json"
        if not path.is_file():
            continue
        codes = load_sids(arm)
        panel[arm] = {}
        for depth in DEPTHS:
            for split in ("node", "instance"):
                record = role_probe(codes, families, vocab, split=split, depth=depth)
                panel[arm][f"{split}_d{depth}"] = record
                print(f"{arm} depth={depth} split={split}: mean={record['mean']:.4f} "
                      f"floor={record['majority_floor_mean']:.4f}", flush=True)
    (HERE / "probe_depth.json").write_text(json.dumps(panel, indent=1))


if __name__ == "__main__":
    main()
