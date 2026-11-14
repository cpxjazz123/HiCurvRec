"""How the cached Role 3 is cut at other granularities: no NMF setting changes, only K.

Reported because the K-sensitivity section claims the sparse group survives K = 8 and K = 10 rather
than being split.  The cached assignment is the K = 6 fit of section 5.24; this script refits the same
feature block at K = 6, 8 and 10 and counts where the cached Role 3's items land.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nmf_k_sensitivity as nk  # noqa: E402
import orca5_role_preservation as m  # noqa: E402

OUT = HERE / "role3_split_by_k.json"
TARGET = 2                    # the cached role the report calls Role 3
KS = (6, 8, 10)


def main() -> int:
    n, adjacency, _ = m.load_graph()
    print("building the ORCA feature block ...", flush=True)
    matrix = nk.build_features(n, adjacency)

    cached = np.full(n, -1, dtype=np.int64)
    for node, role in zip(np.load(m.ROLE_CACHE5)["nodes"], np.load(m.ROLE_CACHE5)["role"]):
        cached[int(node)] = int(role)
    base = np.flatnonzero(cached == TARGET)
    print(f"cached Role 3 holds {len(base)} items", flush=True)

    rows = {"cached_size": int(len(base)), "splits": {}}
    for k in KS:
        assignment = nk.fit_roles(matrix, k)
        sizes = np.bincount(assignment[base], minlength=k)
        shares = sizes / max(len(base), 1)
        rows["splits"][str(k)] = {"sizes": sizes.tolist(), "shares": shares.tolist()}
        print(f"K={k:<3} the cached Role 3 splits as {sizes.tolist()} "
              f"(shares {[round(float(value), 3) for value in shares]})", flush=True)

    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"written {OUT.relative_to(m.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
