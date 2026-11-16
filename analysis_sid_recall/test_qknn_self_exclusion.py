"""Regression: q-kNN must never return the query item itself.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/test_qknn_self_exclusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qknn_community_correlation import qknn_edges  # noqa: E402


def main() -> int:
    # Duplicate coordinates make the self entry non-unique; removal must be by index.
    q = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]], dtype=np.float32)
    edges = qknn_edges(q, k=2)
    assert not any(int(a) == int(b) for a, b in edges.tolist()), edges
    assert len(edges) > 0
    print("qknn self-exclusion ok; edges:", len(edges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
