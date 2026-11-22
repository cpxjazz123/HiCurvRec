"""Issue #231 §5: score the three arms on the frozen, globally isolated banks.

C40 asks for six numbers per structure, and C41/C43/C45 compare them arm by arm. The comparison
uses the same instances, so the difference of two arms is paired; C44 requires robust intervals
and explicitly forbids claiming a noise floor from a single half-split, so the interval comes
from a cluster bootstrap over items (the pairs of one instance share nodes with its neighbours).

    python3 -m analysis_sid_recall.graphlet_verdict231 --arm <dir> --control <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_sid_recall import graphlet_contract as contract

REPO = Path(__file__).resolve().parents[1]
BANKS = REPO / "results" / "research" / "issue231" / "banks"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
ROUNDS = 4000
SEED = 42


def codes_of(arm: Path) -> np.ndarray:
    sids = json.loads((Path(arm) / "item_sids.json").read_text())
    return np.asarray([sids[str(i)][:3] for i in range(len(sids))], dtype=np.int64)


def per_instance(codes: np.ndarray, kind: str, instances: np.ndarray) -> np.ndarray:
    a, b, c = instances[:, 0], instances[:, 1], instances[:, 2]
    return contract.delta_of(kind,
                             contract.shared_prefix(codes, a, b),
                             contract.shared_prefix(codes, b, c),
                             contract.shared_prefix(codes, a, c))


def level_gains(codes: np.ndarray, kind: str, instances: np.ndarray) -> dict:
    """C40: the per-level increment, each an indicator, never multiplied by the level index."""
    a, b, c = instances[:, 0], instances[:, 1], instances[:, 2]
    width = min(codes.shape[1], 3)
    out = {}
    previous = 0.0
    for depth in range(1, width + 1):
        sliced = codes[:, :depth]
        s_at = lambda u, v: (sliced[u] == sliced[v]).all(axis=1).astype(np.float64) * depth
        value = float(contract.delta_of(kind, s_at(a, b), s_at(b, c), s_at(a, c)).mean())
        out[str(depth)] = float(value - previous)
        previous = value
    return out


def cluster_bootstrap(arm_delta, ref_delta, items, rounds=ROUNDS, seed=SEED) -> dict:
    diff = arm_delta - ref_delta
    rng = np.random.default_rng(seed)
    universe = np.unique(items)
    position = {int(v): i for i, v in enumerate(universe)}
    anchors = np.array([position[int(items[i * 3])] for i in range(len(diff))], dtype=np.int64)
    draws = np.empty(rounds)
    for r in range(rounds):
        chosen = rng.choice(len(universe), size=len(universe), replace=True)
        weight = np.zeros(len(universe))
        np.add.at(weight, chosen, 1.0)
        w = weight[anchors]
        draws[r] = float((diff * w).sum() / max(w.sum(), 1e-12))
    return {"difference": float(diff.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "rounds": int(rounds), "seed": int(seed)}


def verdict(arm: Path, control: Path) -> dict:
    arm_codes, ctrl_codes, base_codes = codes_of(arm), codes_of(control), codes_of(BASELINE)
    out = {"arm": str(arm), "control": str(control), "baseline": str(BASELINE), "kinds": {}}
    for kind in ("chain", "fork", "merge"):
        instances = np.load(BANKS / f"{kind}_validation.npy")
        test = np.load(BANKS / f"{kind}_test.npy")
        # §5 says the reported set is the frozen TEST bank; validation is recorded alongside
        for name, bank in (("test", test), ("validation", instances)):
            items = bank.reshape(-1)
            d_arm = per_instance(arm_codes, kind, bank)
            d_base = per_instance(base_codes, kind, bank)
            d_ctrl = per_instance(ctrl_codes, kind, bank)
            entry = {
                "instances": int(len(bank)),
                "baseline": contract.measure(base_codes, kind, bank),
                "arm": contract.measure(arm_codes, kind, bank),
                "control": contract.measure(ctrl_codes, kind, bank),
                "level_gains_arm": level_gains(arm_codes, kind, bank),
                "level_gains_baseline": level_gains(base_codes, kind, bank),
                "arm_minus_baseline": cluster_bootstrap(d_arm, d_base, items),
                "arm_minus_control": cluster_bootstrap(d_arm, d_ctrl, items),
            }
            entry["delta_above_baseline"] = bool(entry["arm"]["delta"] > entry["baseline"]["delta"])
            entry["satisfy_above_baseline"] = bool(
                entry["arm"]["satisfy_rate"] > entry["baseline"]["satisfy_rate"])
            entry["delta_above_control"] = bool(entry["arm"]["delta"] > entry["control"]["delta"])
            entry["near_not_below_baseline"] = bool(
                entry["arm"]["near_share"] >= entry["baseline"]["near_share"])
            entry["lower_bound_positive"] = bool(entry["arm_minus_baseline"]["ci_low"] > 0.0)
            out["kinds"].setdefault(kind, {})[name] = entry
    test = {k: v["test"] for k, v in out["kinds"].items()}
    out["test_summary"] = {
        "all_deltas_above_baseline": all(v["delta_above_baseline"] for v in test.values()),
        "all_satisfy_above_baseline": all(v["satisfy_above_baseline"] for v in test.values()),
        "all_deltas_above_control": all(v["delta_above_control"] for v in test.values()),
        "all_near_not_below_baseline": all(v["near_not_below_baseline"] for v in test.values()),
        "simultaneous_lower_bounds_positive": all(v["lower_bound_positive"] for v in test.values()),
        "fork_reversed": bool(out["kinds"]["fork"]["test"]["arm"]["delta"] > 0.0
                              and out["kinds"]["fork"]["test"]["baseline"]["delta"] < 0.0),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = verdict(args.arm, args.control)
    out = args.out or REPO / "results" / "research" / "issue231" / "verdict_result.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for kind, splits in report["kinds"].items():
        t = splits["test"]
        print(f"{kind:6s} arm {t['arm']['delta']:+.4f} base {t['baseline']['delta']:+.4f} "
              f"ctrl {t['control']['delta']:+.4f} "
              f"| vs base {t['arm_minus_baseline']['difference']:+.4f} "
              f"[{t['arm_minus_baseline']['ci_low']:+.4f},{t['arm_minus_baseline']['ci_high']:+.4f}] "
              f"| vs ctrl {t['arm_minus_control']['difference']:+.4f} "
              f"[{t['arm_minus_control']['ci_low']:+.4f},{t['arm_minus_control']['ci_high']:+.4f}] "
              f"| n={t['instances']}")
    print(json.dumps(report["test_summary"], indent=1))


if __name__ == "__main__":
    main()
