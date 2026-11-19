"""Issue #216 evaluation: heldout closure-vs-branch prefix gains, with bootstrap CIs.

    G_l = P[b, c share prefix l | y = 1] - P[b, c share prefix l | y = 0]

computed on the frozen heldout pool (never seen in training), per level prefix
(L1, L1+L2, full SID), plus the conditional "share L2 given share L1" rates and the
absolute rates themselves.  Clusters are the exact degree triples, so the bootstrap
resamples whole matched strata rather than individual pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
POOL = REPO / "results" / "research" / "g4_closure" / "pairs_heldout.npz"
OUT = REPO / "results" / "research" / "g4_closure" / "evaluation.json"
BOOTSTRAP = 2000


def load_sid(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    return {int(k): tuple(v[:3]) for k, v in payload.items()}


def rates(sid: dict, triples: np.ndarray, labels: np.ndarray, depth: int) -> dict:
    first = np.array([tuple(sid[int(r[1])][:depth]) for r in triples], dtype=object)
    second = np.array([tuple(sid[int(r[2])][:depth]) for r in triples], dtype=object)
    # tuple arrays compare elementwise; the pair shares a prefix only if the whole
    # prefix matches, so compare tuples one by one
    share = np.array([tuple(a) == tuple(b) for a, b in zip(first, second)], dtype=bool)
    positive = labels == 1
    return {"positive": float(share[positive].mean()), "negative": float(share[~positive].mean()),
            "gap": float(share[positive].mean() - share[~positive].mean()), "share": share}


def bootstrap_gap(share: np.ndarray, labels: np.ndarray, degrees: np.ndarray,
                  rounds: int = BOOTSTRAP, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    strata = [np.nonzero((degrees == triple).all(axis=1))[0]
              for triple in np.unique(degrees, axis=0)]
    gaps = []
    for _ in range(rounds):
        picks = np.concatenate([strata[i] for i in rng.integers(0, len(strata), size=len(strata))])
        positive = labels[picks] == 1
        if positive.sum() == 0 or (~positive).sum() == 0:
            continue
        gaps.append(share[picks][positive].mean() - share[picks][~positive].mean())
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main() -> None:
    payload = np.load(POOL, allow_pickle=False)
    triples = payload["triples"]
    labels = payload["labels"]
    degrees = payload["degrees"]
    results = {}
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        sid_path = Path(path)
        if not sid_path.is_file():
            continue
        sid = load_sid(sid_path)
        entry = {"sid": str(path)}
        for depth, label in ((1, "L1"), (2, "L1+L2"), (3, "full")):
            stats = rates(sid, triples, labels, depth)
            low, high = bootstrap_gap(stats["share"], labels, degrees)
            entry[label] = {"positive": stats["positive"], "negative": stats["negative"],
                            "gap": stats["gap"], "ci": [low, high]}
        # conditional: among pairs that share L1, how often do they also share L2
        l1 = rates(sid, triples, labels, 1)["share"]
        l2 = rates(sid, triples, labels, 2)["share"]
        for tag, mask in (("positive", labels == 1), ("negative", labels == 0)):
            base = l1[mask].sum()
            entry["conditional_L2_given_L1_" + tag] = float(l2[mask][l1[mask]].mean()) if base else 0.0
        codes = list(sid.values())
        entry["codes"] = {"L1": len({c[0] for c in codes}),
                          "L2_prefixes": len({tuple(c[:2]) for c in codes}),
                          "triples": len(set(codes)),
                          "collision": 1.0 - len(set(codes)) / len(codes)}
        results[name] = entry
    OUT.write_text(json.dumps(results, indent=1))
    print(f"{'arm':14s}{'L1 gap':>9s}{'95% CI':>18s}{'L1+L2 gap':>11s}{'95% CI':>18s}{'full gap':>10s}{'coll':>7s}")
    for name, entry in results.items():
        l1, l2, l3 = entry["L1"], entry["L1+L2"], entry["full"]
        print(f"{name:14s}{l1['gap']:9.4f}{'[%.3f,%.3f]' % tuple(l1['ci']):>18s}"
              f"{l2['gap']:11.4f}{'[%.3f,%.3f]' % tuple(l2['ci']):>18s}"
              f"{l3['gap']:10.4f}{entry['codes']['collision']:7.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
