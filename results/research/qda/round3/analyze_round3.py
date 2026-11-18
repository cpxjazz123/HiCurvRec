"""Round 3: where the structure is preserved, repaired and destroyed across levels.

Two questions, both answered from observation snapshots that already exist:

1. **Residual transfer** -- what does each level hand to the next?  The AUC of the
   input residual, of the readout, and of the residual left behind, all in the same
   mixed metric, so "L2 did nothing with the behaviour signal" is measurable rather
   than inferred from a flat final AUC.
2. **Per-edge flips** -- for each fixed (edge, matched non-edge) pair, was the
   distinction made at level l?  Then, per adjacent pair of levels:

       destroy  D(l->l+1) = P(C_{l+1}=0 | C_l=1)
       repair   R(l->l+1) = P(C_{l+1}=1 | C_l=0)

   with exact ties counted as their own class and never folded into either side.

Both are reported overall and inside the five frozen edge strata.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "round2"
QDA = HERE.parent
SAMPLE = QDA / "sample" / "sample.npz"

STRATA = {
    "acyclic": lambda tri, inside: tri == 0,
    "triangle": lambda tri, inside: tri == 1,
    "dense": lambda tri, inside: tri >= 2,
    "intra_community": lambda tri, inside: inside == 1,
    "bridge": lambda tri, inside: inside == 0,
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load("qda_analyze", QDA / "analyze.py")
#: `analyze.load_geometry()` registers the model package, so `logmap0` comes from the
#: same frozen tree the observations were produced with.
geometry = None


def _snapshot(arm: str, epoch: int) -> Path:
    return ROUND2 / "runs" / arm / "observations" / f"epoch_{epoch:04d}.npz"


def _distances(vectors: np.ndarray, sample, expmap0, squared):
    left = sample["edges"][:, 0].astype(np.int64)
    right = sample["edges"][:, 1].astype(np.int64)
    nleft = sample["negatives"][:, 0].astype(np.int64)
    nright = sample["negatives"][:, 1].astype(np.int64)
    positive = analyze.pair_distance(vectors, left, right, expmap0, squared)
    negative = analyze.pair_distance(vectors, nleft, nright, expmap0, squared)
    return positive, negative


def _residual_vector(record, level: int) -> np.ndarray:
    """Rebuild the 32-dim tangent code the next level actually consumes.

    The observer stores the residual in the split convention: `residual_e` in the
    tangent (Euclidean half) and `residual_h` as a point of the ball.  `pair_distance`
    applies `expmap0` to the last 16 columns, so the ball point has to be mapped back
    with `logmap0` before it is handed back.
    """
    import torch

    contract = sys.modules["qda_model.poincare_contract"]
    e = record[f"residual_e_{level}"].astype(np.float32)
    h = record[f"residual_h_{level}"].astype(np.float32)
    back = contract.logmap0(torch.from_numpy(h), 1.0).numpy()
    return np.concatenate([e, back], axis=1)


def outcomes(record, sample, expmap0, squared, mask) -> dict:
    """Per-level hit / tie masks, plus the residual AUCs, all on the held-out rows."""
    out = {"hit": {}, "tie": {}, "auc": {}}
    for level in (1, 2, 3):
        vectors = record[f"readout_{level}"].astype(np.float32)
        positive, negative = _distances(vectors, sample, expmap0, squared)
        out["hit"][level] = positive < negative
        out["tie"][level] = positive == negative
        out["auc"][f"readout_{level}"] = float(
            out["hit"][level][mask].mean() + 0.5 * out["tie"][level][mask].mean())
    for level in (2, 3):
        vectors = _residual_vector(record, level)
        positive, negative = _distances(vectors, sample, expmap0, squared)
        hit, tie = positive < negative, positive == negative
        out["auc"][f"residual_{level - 1}"] = float(hit[mask].mean() + 0.5 * tie[mask].mean())
        out["hit"][f"residual_{level - 1}"] = hit
        out["tie"][f"residual_{level - 1}"] = tie
    return out


def transitions(level_from, level_to, hit_a, hit_b, tie_a, tie_b, mask) -> dict:
    """Destroy / repair between two levels, ties excluded and reported."""
    valid = mask & ~tie_a & ~tie_b
    a = hit_a[valid]
    b = hit_b[valid]
    keep = int((a & b).sum())
    destroy = int((a & ~b).sum())
    repair = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    total = keep + destroy + repair + neither
    return {
        "rows": int(mask.sum()),
        "compared": total,
        "tie_rate": float(1 - total / max(int(mask.sum()), 1)),
        "correct_correct": keep / total if total else None,
        "correct_wrong": destroy / total if total else None,
        "wrong_correct": repair / total if total else None,
        "wrong_wrong": neither / total if total else None,
        "destroy_rate": destroy / (keep + destroy) if keep + destroy else None,
        "repair_rate": repair / (repair + neither) if repair + neither else None,
    }


def one_arm(arm: str, epoch: int, sample, expmap0, squared, draws: int, rng) -> dict:
    record = np.load(_snapshot(arm, epoch))
    split = sample["split"] == 1
    out = outcomes(record, sample, expmap0, squared, split)
    triangles, inside = sample["triangles"], sample["inside"]
    result = {"epoch": epoch, "auc": out["auc"], "overall": {}, "strata": {}}
    for a, b in ((1, 2), (2, 3), (1, 3)):
        result["overall"][f"{a}->{b}"] = transitions(
            a, b, out["hit"][a], out["hit"][b], out["tie"][a], out["tie"][b], split)
    for name, rule in STRATA.items():
        mask = split & rule(triangles, inside)
        result["strata"][name] = {
            f"{a}->{b}": transitions(a, b, out["hit"][a], out["hit"][b],
                                     out["tie"][a], out["tie"][b], mask)
            for a, b in ((1, 2), (2, 3), (1, 3))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=["A_fine:3000", "B3_long:3000",
                                                       "B0_replay:300", "B3_l1_balanced:300"])
    parser.add_argument("--out", default="report_round3.json")
    args = parser.parse_args()

    sample = np.load(SAMPLE)
    expmap0, squared = analyze.load_geometry()
    rng = np.random.default_rng(11)
    report = {"arms": {}}
    for spec in args.pairs:
        arm, epoch = spec.split(":")
        if not _snapshot(arm, int(epoch)).exists():
            print(f"{arm}@{epoch}: no snapshot")
            continue
        report["arms"][spec] = one_arm(arm, int(epoch), sample, expmap0, squared, 2000, rng)
        overview = report["arms"][spec]
        auc = overview["auc"]
        print(f"{spec:22s} AUC  L1 {auc['readout_1']:.4f}  L2 {auc['readout_2']:.4f}  "
              f"L3 {auc['readout_3']:.4f} | residual after L1 {auc['residual_1']:.4f}  "
              f"after L2 {auc['residual_2']:.4f}")
        for key, value in overview["overall"].items():
            print(f"    {key}  destroy {value['destroy_rate']:.3f}  "
                  f"repair {value['repair_rate']:.3f}  ties {value['tie_rate']:.3f}")
    (HERE / args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
