"""Build the frozen local-structure table of issue #149, once, before training.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 build_structure_table.py

The builder reads the TRAIN split only, rebuilds the behaviour graph the frozen
AF3 cache was measured on, verifies the rebuild edge for edge, and writes

    results/research/issue149/structure_table.npz    the descriptor and the sets
    results/research/issue149/structure_table.json   its SHA-256 and its counts

Nothing is written outside the repository (CLAUDE.md §86), and the trainer
refuses to start without the manifest-verified table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import structure_table  # noqa: E402


def _repository_root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(TREE))


def main() -> int:
    root = _repository_root()
    manifest = structure_table.write_table(root)
    metadata = manifest["metadata"]
    print(json.dumps({
        "table": manifest["table"],
        "table_sha256": manifest["table_sha256"],
        "n_items": metadata["n_items"],
        "n_nodes": metadata["n_nodes"],
        "n_edges": metadata["n_edges"],
        "n_communities": metadata["n_communities"],
        "items_without_structure": metadata["items_without_structure"],
        "triangles": metadata["triangles"],
        "cross_pairs": metadata["cross_pairs"],
        "same_pairs": metadata["same_pairs"],
        "af3_cache_status": metadata["af3_cache_status"],
        "undefined_dimensions": metadata["undefined_dimensions"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
