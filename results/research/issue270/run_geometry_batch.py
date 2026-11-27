"""Run the full-geometry evaluation for every usable cohort arm (4 workers)."""
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


def one(arm: str) -> tuple[str, str, int, float]:
    out = OUT / "assets/geometry" / (arm + ".npz")
    if out.is_file():
        return arm, "REUSED", 0, 0.0
    r = run("issue270_geometry_" + arm, [PY3, "-u", "results/research/issue270/geometry_arm.py", arm],
            out_dir="results/research/issue270/receipts", timeout=1800)
    return arm, r.status, r.exit_code, r.duration_s


def main() -> None:
    cohort = json.loads((OUT / "COHORT.json").read_text())
    arms = [r["arm"] for r in cohort["rows"] if r["usable_geometry"]]
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, arm): arm for arm in arms}
        for future in as_completed(futures):
            arm, status, code, duration = future.result()
            results.append({"arm": arm, "status": status, "exit": code, "duration_s": duration})
            print(arm, status, code, round(duration, 1), flush=True)
    OUT.joinpath("GEOMETRY_BATCH.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("done", len(results), "failed", sum(r["status"] != "PASS" and r["status"] != "REUSED" for r in results))


if __name__ == "__main__":
    main()
