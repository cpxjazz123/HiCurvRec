"""Run the 20k-sample distortion J for every usable cohort arm (8 workers, 1 thread)."""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.iteration.execution import run
OUT = REPO / "results/research/issue270"
PY3 = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"


def one(arm: str) -> dict:
    out = OUT / "assets/geometry" / (arm + ".distortion.json")
    if out.is_file():
        return {"arm": arm, "status": "REUSED"}
    r = run("issue270_distortion_" + arm,
            [PY3, "-u", "results/research/issue270/distortion.py", arm],
            out_dir="results/research/issue270/receipts", timeout=900)
    return {"arm": arm, "status": r.status, "exit": r.exit_code, "duration_s": r.duration_s}


def main() -> None:
    cohort = json.loads((OUT / "COHORT.json").read_text())
    arms = [r["arm"] for r in cohort["rows"] if r["usable_geometry"]]
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, arm): arm for arm in arms}
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(row, flush=True)
    OUT.joinpath("DISTORTION_BATCH.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    bad = [r for r in results if r["status"] not in ("PASS", "REUSED")]
    print("done", len(results), "bad", len(bad))


if __name__ == "__main__":
    main()
