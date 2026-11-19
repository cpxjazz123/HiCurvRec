"""Issue #217 CGV item 5: two fresh processes must reproduce the archived SID exactly."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
G17 = REPO / "results" / "research" / "graphlet_roles"
OUT = G17 / "cgv" / "reencode.json"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def main() -> None:
    (G17 / "cgv").mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in sys.argv[1:]:
        archive = REPO / f"results/stage2_RQ-VAE/TIGER_graphlet_role_{arm}/item_sids.json"
        if not archive.is_file():
            continue
        # Parse the archive once: re-reading it per row is quadratic in 24587 rows.
        payload = json.loads(archive.read_text())
        stored = np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)
        runs = []
        for tag in ("a", "b"):
            target = G17 / "cgv" / f"reencode_{arm}_{tag}.npy"
            subprocess.run([PYTHON, str(G17 / "reencode_one.py"), arm, str(target)],
                           check=True, cwd=REPO, capture_output=True)
            runs.append(np.load(target))
        results[arm] = {
            "runs_agree": bool(np.array_equal(runs[0], runs[1])),
            "mismatches_vs_archive": int((runs[0] != stored).any(axis=1).sum()),
            "rows": int(len(stored)),
            "sha256": hashlib.sha256(runs[0].tobytes()).hexdigest(),
        }
        print(arm, results[arm])
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
