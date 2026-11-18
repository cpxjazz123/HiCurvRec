"""Round 2 analysis: who moved, what collapsed, and whether recovering it mattered.

Reads the per-epoch telemetry every round-2 arm writes and the observation snapshots
the arms keep at their full points, and answers the three pre-declared layers:

  1. mechanism  -- timing of the L1 collapse, encoder vs codebook velocity, the
                   gradient a dead code does or does not receive, and the exact
                   never-used / died / revived split;
  2. structure  -- behaviour-edge AUC and prefix lift per level at the same epochs,
                   so "more codes" is never mistaken for "better structure";
  3. strata     -- the same comparison inside five edge classes (acyclic, triangle,
                   dense, intra-community, bridge), paired on the same negatives.

Nothing here trains or writes into an arm; it only reads `round2/runs/<arm>/`.
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
RUNS = HERE / "runs"
SAMPLE = QDA / "sample" / "sample.npz"
LEVELS = ("l1", "l2", "l3")

#: Edge strata, all defined on the pre-frozen sample so no result can move them.
STRATA = {
    "acyclic": lambda tri, inside: tri == 0,
    "triangle": lambda tri, inside: tri == 1,
    "dense": lambda tri, inside: tri >= 2,
    "intra_community": lambda tri, inside: inside == 1,
    "bridge": lambda tri, inside: inside == 0,
}


def _load_analyze():
    spec = importlib.util.spec_from_file_location("qda_analyze", QDA / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qda_analyze"] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze()


def telemetry(arm: str) -> dict:
    path = RUNS / arm / "logs" / "training_metrics.jsonl"
    if not path.exists():
        return {"epochs": [], "lifecycle": None}
    epochs, lifecycle, events = [], None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        events[row["event"]] = events.get(row["event"], 0) + 1
        if row["event"] == "qda_epoch":
            epochs.append(row)
        elif row["event"] == "qda_lifecycle":
            lifecycle = row
        elif row["event"] == "qda_revive":
            events.setdefault("revived_events", 0)
    epochs.sort(key=lambda r: r["epoch"])
    return {"epochs": epochs, "lifecycle": lifecycle, "events": events}


def summaries(arm: str) -> dict:
    """The observation summaries (light or full) keyed by epoch."""
    out = {}
    obs = RUNS / arm / "observations"
    if not obs.is_dir():
        return out
    for path in sorted(obs.glob("epoch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[int(payload["epoch"])] = payload
    return out


def snapshots(arm: str) -> list[int]:
    obs = RUNS / arm / "observations"
    if not obs.is_dir():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in obs.glob("epoch_*.npz"))


def structural(arm: str, epochs=None) -> dict:
    """Behaviour AUC and prefix lift per level, from the observation snapshots."""
    sample = np.load(SAMPLE)
    split = sample["split"] == 1
    expmap0, squared = analyze.load_geometry()
    out = {}
    for epoch in snapshots(arm):
        if epochs is not None and epoch not in epochs:
            continue
        record = np.load(RUNS / arm / "observations" / f"epoch_{epoch:04d}.npz")
        codes = record["codes"]
        entry = {"levels": {}}
        for level, name in enumerate(LEVELS, start=1):
            vectors = record[f"readout_{level}"].astype(np.float32)
            entry["levels"][name] = {
                "auc": analyze.edge_auc(vectors, sample, split, expmap0, squared),
                "lift": analyze.prefix_lift(codes, sample, split, level)["lift"],
            }
        vectors = record["z"].astype(np.float32)
        entry["encoder_auc"] = analyze.edge_auc(vectors, sample, split, expmap0, squared)
        out[epoch] = entry
    return out


def strata(arm_a: str, arm_b: str, epoch: int, draws: int = 2000, seed: int = 17) -> dict:
    """Per-stratum paired comparison of two arms' level-1 (and level-3) readouts.

    Paired on the row: the same edge and its same matched non-edge are scored under
    both arms, so a stratum difference cannot come from a different item population.
    """
    sample = np.load(SAMPLE)
    split = sample["split"] == 1
    triangles, inside = sample["triangles"], sample["inside"]
    expmap0, squared = analyze.load_geometry()
    rng = np.random.default_rng(seed)
    left = sample["edges"][:, 0].astype(np.int64)
    right = sample["edges"][:, 1].astype(np.int64)
    nleft = sample["negatives"][:, 0].astype(np.int64)
    nright = sample["negatives"][:, 1].astype(np.int64)
    hits = {}
    for arm in (arm_a, arm_b):
        record = np.load(RUNS / arm / "observations" / f"epoch_{epoch:04d}.npz")
        for level in (1, 3):
            vectors = record[f"readout_{level}"].astype(np.float32)
            positive = analyze.pair_distance(vectors, left, right, expmap0, squared)
            negative = analyze.pair_distance(vectors, nleft, nright, expmap0, squared)
            # Same convention as `analyze.edge_auc`: a tie scores one half. Without
            # it the strata would quietly answer a different question -- with 26 live
            # codes many pairs are exactly tied, so raw hit rates understate an arm
            # whose readouts are coarse.
            hits[(arm, level)] = ((positive < negative).astype(np.float64)
                                  + 0.5 * (positive == negative).astype(np.float64))
    out = {}
    for name, rule in STRATA.items():
        mask = split & rule(triangles, inside)
        if mask.sum() < 200:
            out[name] = {"rows": int(mask.sum()), "verdict": "TOO_FEW_ROWS"}
            continue
        entry = {"rows": int(mask.sum())}
        for level in (1, 3):
            a = hits[(arm_a, level)][mask]
            b = hits[(arm_b, level)][mask]
            observed = float(a.mean() - b.mean())
            # Paired scores live on {0, 0.5, 1} (a tie scores one half), so the
            # bootstrap of the mean difference depends only on the joint cell counts.
            # Sampling those counts is exact and avoids 315k-index resamples; a
            # four-cell version would drop every tie and shift the interval.
            cells, counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
            probabilities = counts / counts.sum()
            n = int(counts.sum())
            sampled = rng.multinomial(n, probabilities, size=draws).astype(np.float64)
            boot = (sampled @ cells[:, 0] - sampled @ cells[:, 1]) / n
            # Monte-Carlo tolerance: the mean of `draws` resamples wobbles by
            # std/sqrt(draws), not by std/sqrt(n).
            if abs(boot.mean() - observed) > 4 * boot.std() / np.sqrt(draws) + 1e-12:
                raise RuntimeError(f"bootstrap is not centred on the estimate: "
                                   f"{boot.mean():.6f} vs {observed:.6f}")
            entry[f"l{level}"] = {
                "delta": observed,
                "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
            }
        out[name] = entry
    return out


def compact(epochs: list[dict], points) -> list[dict]:
    by_epoch = {row["epoch"]: row for row in epochs}
    return [{"epoch": point, **{key: by_epoch[point][key] for key in
                                ("used", "z_move_rel", "codebook_move_rel",
                                 "grad_norm_used", "grad_norm_unused")}}
            for point in points if point in by_epoch]


def velocity_ratio(epochs: list[dict]) -> list[dict]:
    """Encoder displacement per unit of codebook displacement, level 1."""
    out = []
    for row in epochs:
        code = row["codebook_move_rel"][0]
        out.append({"epoch": row["epoch"], "ratio": row["z_move_rel"] / max(code, 1e-12),
                    "z": row["z_move_rel"], "code": code})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--reference", default="B0_replay")
    parser.add_argument("--no-strata", action="store_true")
    parser.add_argument("--out", default="report_round2.json")
    parser.add_argument("--runs", type=Path, default=None,
                        help="run root to read (default: round2/runs)")
    args = parser.parse_args()
    global RUNS
    if args.runs is not None:
        RUNS = args.runs
    arms = args.arms or sorted(p.name for p in RUNS.iterdir() if p.is_dir())

    report = {"arms": {}, "strata": {}, "velocity": {}}
    for arm in arms:
        tel = telemetry(arm)
        epochs = tel["epochs"]
        if not epochs:
            print(f"{arm}: no telemetry yet")
            continue
        used1 = [row["used"][0] for row in epochs]
        trough = int(np.argmin(used1))
        report["arms"][arm] = {
            "epochs": len(epochs),
            "events": tel.get("events", {}),
            "l1_used_initial": used1[0],
            "l1_used_min": int(used1[trough]),
            "l1_used_min_epoch": epochs[trough]["epoch"],
            "l1_used_final": used1[-1],
            "l2_used_final": epochs[-1]["used"][1],
            "l3_used_final": epochs[-1]["used"][2],
            "points": compact(epochs, (0, 1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 60, 75,
                                       100, 150, 200, 250, 300)),
            "lifecycle": tel.get("lifecycle"),
            "structural": structural(arm),
        }
        report["velocity"][arm] = velocity_ratio(epochs)
        print(f"{arm}: {len(epochs)} epochs, L1 {used1[0]} -> {used1[trough]} (epoch "
              f"{epochs[trough]['epoch']}) -> {used1[-1]}; L2/L3 final "
              f"{epochs[-1]['used'][1]}/{epochs[-1]['used'][2]}")

    reference = args.reference
    for arm in ([] if args.no_strata else arms):
        if arm == reference or arm not in report["arms"]:
            continue
        for epoch in (50, 100, 200, 300, 1500, 3000):
            if (RUNS / arm / "observations" / f"epoch_{epoch:04d}.npz").exists() and \
               (RUNS / reference / "observations" / f"epoch_{epoch:04d}.npz").exists():
                report["strata"][f"{arm}_vs_{reference}@{epoch}"] = strata(arm, reference, epoch)

    (HERE / args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
