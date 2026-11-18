"""Run the CNCQ gate over every arm this investigation has produced.

No arm is expected to pass: that is the point. The gate exists so a collapsed model can
never be delivered, and the honest starting position is that we do not yet have a model
that clears all layers.

For arms trained before the module existed there is no prototype/offset decomposition, so
the geometry layer reports the *measured* minimum mixed distance between the codewords the
arm actually uses, and says so; the analytic bound applies only to a CNCQ-built codebook.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QDA = HERE.parent
sys.path.insert(0, str(HERE))
from cncq import Policy, certify, mixed_min_distance   # noqa: E402

SAMPLE = QDA / "sample" / "sample.npz"
ROUNDS = {"round2": QDA / "round2" / "runs", "round5": QDA / "round5" / "runs",
          "round6": QDA / "round6" / "runs"}
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


def budget(arm: str) -> int | None:
    """The training budget the arm ran, read from its own metrics stream.

    A gate that compares a 3000-epoch run against a 300-epoch reference is not a gate: the
    longer schedule changes the LR at the compared epoch. Budgets are therefore matched.
    """
    for root in ROUNDS.values():
        path = root / arm / "logs" / "training_metrics.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "train_start":
                return int(row.get("epochs", 0)) or None
    return None


def export_agreement(record, level: int) -> float | None:
    key = f"train_code_{level}"
    if key not in record:
        return None
    return float((record[key] == record["codes"][:, level - 1]).mean())


def find(arm: str, epoch: int) -> Path | None:
    for root in ROUNDS.values():
        candidate = root / arm / "observations" / f"epoch_{epoch:04d}.npz"
        if candidate.exists():
            return candidate
    return None


def codewords(record) -> np.ndarray:
    """The level-1 codewords this arm actually uses, taken from its own readouts."""
    codes = record["codes"][:, 0]
    readout = record["readout_1"].astype(np.float32)
    out = []
    for code in np.unique(codes):
        out.append(readout[codes == code][0])     # every item on a code shares its readout
    return np.stack(out)


def measure(arm: str, epoch: int, sample, expmap0, squared) -> dict | None:
    path = find(arm, epoch)
    if path is None:
        return None
    record = np.load(path)
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    split = sample["split"] == 1
    vectors = record["readout_3"].astype(np.float32)
    left = sample["edges"][:, 0].astype(np.int64)
    right = sample["edges"][:, 1].astype(np.int64)
    nleft = sample["negatives"][:, 0].astype(np.int64)
    nright = sample["negatives"][:, 1].astype(np.int64)
    positive = analyze.pair_distance(vectors, left, right, expmap0, squared)
    negative = analyze.pair_distance(vectors, nleft, nright, expmap0, squared)
    hit = (positive < negative) + 0.5 * (positive == negative)
    fidelity = float(hit[split].mean())
    triangles, inside = sample["triangles"], sample["inside"]
    strata = {}
    for name, rule in STRATA.items():
        mask = split & rule(triangles, inside)
        if mask.sum():
            strata[name] = float(hit[mask].mean())
    codes = codewords(record)
    separation = mixed_min_distance(codes)
    return {
        "counts": record["usage_1"].astype(int).tolist(),
        "geometry": {"ok": separation > 0, "delta": separation, "two_rho": 0.0,
                     "margin": separation},
        "effective_rank": float(summary["effective_rank"]["readout_1"]),
        "fidelity": fidelity,
        "strata": strata,
        "live": int(len(codes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--reference", default="B0_replay")
    parser.add_argument("--out", default="certification.json")
    parser.add_argument("--export-gate", action="store_true",
                        help="also require L1 training and export assignments to agree")
    args = parser.parse_args()

    sample = np.load(SAMPLE)
    expmap0, squared = analyze.load_geometry()
    policy = Policy()

    reference = measure(args.reference, args.epoch, sample, expmap0, squared)
    if reference is None:
        raise SystemExit(f"reference arm {args.reference} has no snapshot at {args.epoch}")
    reference_block = {"fidelity": reference["fidelity"], "strata": reference["strata"]}
    print(f"reference {args.reference}@{args.epoch}: L3 AUC {reference['fidelity']:.4f}, "
          f"L1 live {reference['live']}, min occupancy {min(reference['counts'])}")

    arms = sorted({path.parents[1].name
                   for root in ROUNDS.values() for path in root.glob(f"*/observations/epoch_{args.epoch:04d}.npz")})
    reference_budget = budget(args.reference)
    report = {"reference": {"arm": args.reference, **reference_block},
              "epoch": args.epoch, "arms": {}}
    print(f"\n{'arm':18s}{'live':>5s}{'minN':>6s}{'rank1':>7s}{'L3':>8s}  verdict")
    for arm in arms:
        metrics = measure(arm, args.epoch, sample, expmap0, squared)
        if metrics is None:
            continue
        verdict = certify(metrics, reference_block, policy)
        arm_budget = budget(arm)
        if reference_budget is not None and arm_budget not in (None, reference_budget):
            verdict = {"certified": False, "failed": [f"budget {arm_budget} != {reference_budget}"],
                       "gates": verdict["gates"]}
        else:
            agreement = export_agreement(np.load(find(arm, args.epoch)), 1)
            metrics["export_agreement_l1"] = agreement
            if args.export_gate:
                # Fail closed: an unmeasured export agreement is not a passing one.
                if agreement is None:
                    verdict["certified"] = False
                    verdict["failed"] = verdict["failed"] + ["export_l1:NOT_RUN"]
                elif agreement < 1.0:
                    verdict["certified"] = False
                    verdict["failed"] = verdict["failed"] + ["export_l1"]
        report["arms"][arm] = {"metrics": metrics, "verdict": verdict}
        state = "CERTIFIED" if verdict["certified"] else "rejected: " + ",".join(verdict["failed"])
        print(f"{arm:18s}{metrics['live']:>5d}{min(metrics['counts']):>6d}"
              f"{metrics['effective_rank']:>7.1f}{metrics['fidelity']:>8.4f}  {state}")

    (HERE / args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    certified = [a for a, v in report["arms"].items() if v["verdict"]["certified"]]
    print(f"\ncertified: {certified or 'none'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
