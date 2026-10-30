"""Run the Lorentz HBAA check on the 107 pairs, eight workers."""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "lorentz_hbaa_107.json"
CURVATURE = 1.0


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _module("lh", HERE / "lorentz_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    poincare = module.gromov_alignment(entry["table"], context,
                                       curvature=CURVATURE, lorentz=False)
    lorentz = module.gromov_alignment(entry["table"], context,
                                      curvature=CURVATURE, lorentz=True)
    return index, float(poincare), float(lorentz), float(entry["test_recall@10"])


def main() -> int:
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(work, range(107)))
    rows.sort()
    poincare = np.array([row[1] for row in rows])
    lorentz = np.array([row[2] for row in rows])
    recall = np.array([row[3] for row in rows])
    rho_p, p_p = stats.spearmanr(poincare, recall)
    rho_l, p_l = stats.spearmanr(lorentz, recall)
    report = {
        "pairs": len(rows),
        "curvature": CURVATURE,
        "poincare": {"spearman": round(float(rho_p), 6), "p_value": float(p_p)},
        "lorentz": {"spearman": round(float(rho_l), 6), "p_value": float(p_l)},
        "max_abs_value_difference": float(np.max(np.abs(poincare - lorentz))),
        "median_abs_value_difference": float(np.median(np.abs(poincare - lorentz))),
        "values": {"poincare": poincare.tolist(), "lorentz": lorentz.tolist()},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("poincare %+.4f | lorentz %+.4f | max |diff| %.3e | median %.3e" % (
        rho_p, rho_l, report["max_abs_value_difference"],
        report["median_abs_value_difference"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
