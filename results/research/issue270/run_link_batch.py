"""Run SID linkage and prediction pairing for every usable cohort arm."""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.iteration.execution import run
OUT = REPO / "results/research/issue270"
PY3 = "/usr/bin/python3"


def one(kind: str, arm: str) -> dict:
    out = OUT / "assets" / kind / (arm + ".npz")
    script = "sid_link.py" if kind == "sid" else "pair_predictions.py"
    if out.is_file():
        return {"arm": arm, "kind": kind, "status": "REUSED"}
    r = run("issue270_%s_%s" % (kind, arm), [PY3, "-u", "results/research/issue270/" + script, arm],
            out_dir="results/research/issue270/receipts", timeout=900)
    return {"arm": arm, "kind": kind, "status": r.status, "exit": r.exit_code,
            "duration_s": r.duration_s}


def main() -> None:
    cohort = json.loads((OUT / "COHORT.json").read_text())
    arms = [r["arm"] for r in cohort["rows"] if r["usable_geometry"]]
    jobs = [(kind, arm) for kind in ("sid", "pred") for arm in arms]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(row, flush=True)
    OUT.joinpath("LINK_BATCH.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    bad = [r for r in results if r["status"] not in ("PASS", "REUSED")]
    print("done", len(results), "bad", len(bad))
    for r in bad:
        print("BAD", r)


if __name__ == "__main__":
    main()
