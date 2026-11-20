"""Role-probe panel for the directed arms (issue #221).

Two readouts per arm: the linear probe from an item's L1 SID code to that
item's directed orbit, and the L3 collision.  The probe is reported on both
splits -- instance-level (an item may appear on both sides) and node-disjoint
(a held-out item must be placed by its code alone) -- because only the second
measures generalisation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directed_role import load_pool, load_vocabulary  # noqa: E402
from evaluate import load_sids, role_probe  # noqa: E402

REPO = HERE.parents[2]
ARMS = ("D0", "D1", "D2", "D3", "D4")


def main() -> None:
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    panel = {}
    for arm in ARMS:
        path = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_drq_{arm}" / "item_sids.json"
        if not path.is_file():
            continue
        codes = load_sids(arm)
        unique = len(np.unique(codes[:, :3], axis=0))
        panel[arm] = {
            "instance_split": role_probe(codes, families, vocab, split="instance"),
            "node_split": role_probe(codes, families, vocab, split="node"),
            "collision": float(1.0 - unique / len(codes)),
            "sid_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{arm}: instance={panel[arm]['instance_split']['mean']:.4f} "
              f"node={panel[arm]['node_split']['mean']:.4f} "
              f"floor={panel[arm]['node_split']['majority_floor_mean']:.4f} "
              f"collision={panel[arm]['collision']:.4f}", flush=True)
    (HERE / "probe_panel.json").write_text(json.dumps(panel, indent=1))


if __name__ == "__main__":
    main()
