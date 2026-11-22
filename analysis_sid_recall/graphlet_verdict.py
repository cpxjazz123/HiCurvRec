#!/usr/bin/env python3
"""Issue #230 §5: the frozen-bank verdict, with a paired bootstrap on the same instances.

C36–C44 need more than three point estimates. The ties dominate (84–97% of instances), so the
sign of Δ is carried by a minority and a bare `arm > baseline` comparison would over-read it.
This script therefore reports, per structure, on the **same** frozen test instances:

* the point Δ, the strict-satisfy / tie / violate split and the absolute near/far shares;
* Δ_arm − Δ_baseline with a paired bootstrap confidence interval (C39/C40);
* Δ_arm − Δ_shuffled for the equal-budget control (C41);
* the per-structure verdict, never aggregated (C44).

Instances are resampled in dependence groups: an instance's three pairs share nodes with others,
so the bootstrap resamples **items** and keeps every instance all of whose items were drawn
(C42's "when dependence cannot be estimated, do not claim significance" — a node-cluster
bootstrap is the defensible middle, and it is wider than the i.i.d. one by construction).

    python3 -m analysis_sid_recall.graphlet_verdict --arm <dir> --control <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_sid_recall import graphlet_contract as contract

REPO = Path(__file__).resolve().parents[1]
BANKS = REPO / "results" / "research" / "issue230" / "banks"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
#: The baseline trainer still lives in the frozen tree, and its metrics stream is written
#: beside that source, not beside the exported SID.
BASELINE_SOURCE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
ROUNDS = 4000
SEED = 42


def codes_of(arm: Path) -> np.ndarray:
    sids = json.loads((Path(arm) / "item_sids.json").read_text())
    return np.asarray([sids[str(i)][:3] for i in range(len(sids))], dtype=np.int64)


def per_instance(codes: np.ndarray, kind: str, instances: np.ndarray) -> np.ndarray:
    a, b, c = instances[:, 0], instances[:, 1], instances[:, 2]
    s_ab = contract.shared_prefix(codes, a, b)
    s_bc = contract.shared_prefix(codes, b, c)
    s_ac = contract.shared_prefix(codes, a, c)
    return contract.delta_of(kind, s_ab, s_bc, s_ac)


def cluster_bootstrap(delta_arm: np.ndarray, delta_ref: np.ndarray, items: np.ndarray,
                      rounds: int = ROUNDS, seed: int = SEED) -> dict:
    """Paired difference Δ_arm − Δ_ref, resampling whole items instead of instances."""
    diff = delta_arm - delta_ref
    rng = np.random.default_rng(seed)
    universe = np.unique(items)
    position = {int(value): index for index, value in enumerate(universe)}
    # one cluster per instance: its three items, so an instance is in or out as a whole
    members = items.reshape(-1, 3)
    slot = np.array([position[int(members[index, 0])] for index in range(len(delta_arm))],
                    dtype=np.int64)
    draws = np.empty(rounds)
    for round_index in range(rounds):
        chosen = rng.choice(len(universe), size=len(universe), replace=True)
        weight = np.zeros(len(universe))
        np.add.at(weight, chosen, 1.0)
        # a cluster is drawn when its anchor item is drawn; the count of that item is its weight
        weights = np.array([weight[slot[i]] for i in range(len(slot))])
        draws[round_index] = float((diff * weights).sum() / max(weights.sum(), 1e-12))
    return {"difference": float(diff.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "rounds": int(rounds), "seed": int(seed)}


def sid_health(codes: np.ndarray, depth: int = 3) -> dict:
    """Collision rate and the bucket statistics §5 asks for, all from the three codes.

    collision = (N - unique SID) / N exactly as the issue writes it; the duplicate-bucket
    share, the largest bucket, the per-level effective codeword count and the conditional
    entropy of each level given the one above are reported beside it so a gain bought by
    collapsing the codebook cannot hide behind the structure deltas.
    """
    short = codes[:, :depth]
    total = len(short)
    buckets: dict = {}
    for row in short.tolist():
        buckets.setdefault(tuple(row), 0)
        buckets[tuple(row)] += 1
    unique = len(buckets)
    sizes = np.array(sorted(buckets.values(), reverse=True))
    per_level, conditional = {}, {}
    for level in range(depth):
        prefix = {}
        for row in short.tolist():
            prefix.setdefault(tuple(row[:level]), set()).add(int(row[level]))
        per_level[str(level + 1)] = len(prefix)
        entropy = {}
        for key, seen in prefix.items():
            counts = np.bincount([int(r[level]) for r in short.tolist() if tuple(r[:level]) == key])
            counts = counts[counts > 0]
            probability = counts / counts.sum()
            entropy[key] = float(-(probability * np.log2(probability)).sum())
        conditional[str(level + 1)] = float(np.mean(list(entropy.values())))
    return {
        "items": int(total),
        "unique_sid": int(unique),
        "collision": float((total - unique) / total),
        "duplicate_item_share": float(1.0 - unique / total),
        "largest_bucket": int(sizes[0]) if len(sizes) else 0,
        "effective_codes_per_level": per_level,
        "conditional_entropy_bits": conditional,
    }


def reconstruction_error(metrics: Path) -> dict:
    """The final-epoch reconstruction loss from the run's own metrics stream."""
    rows = []
    if Path(metrics).is_file():
        for line in Path(metrics).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "train" and "recon" in record:
                rows.append(record)
    if not rows:
        return {"recon_final": None, "epochs": 0}
    last = max(rows, key=lambda record: int(record.get("epoch", 0)))
    return {"recon_final": float(last["recon"]), "epoch": int(last.get("epoch", 0))}


def verdict(arm: Path, control: Path) -> dict:
    arm_codes, control_codes, base_codes = codes_of(arm), codes_of(control), codes_of(BASELINE)
    out = {"arm": str(arm), "control": str(control), "baseline": str(BASELINE), "kinds": {}}
    for kind in ("chain", "fork", "merge"):
        instances = np.load(BANKS / f"{kind}_test.npy")
        items = instances.reshape(-1)
        d_arm = per_instance(arm_codes, kind, instances)
        d_base = per_instance(base_codes, kind, instances)
        d_ctrl = per_instance(control_codes, kind, instances)
        base = contract.measure(base_codes, kind, instances)
        mine = contract.measure(arm_codes, kind, instances)
        shuffled = contract.measure(control_codes, kind, instances)
        vs_base = cluster_bootstrap(d_arm, d_base, items)
        vs_ctrl = cluster_bootstrap(d_arm, d_ctrl, items)
        out["kinds"][kind] = {
            "instances": int(len(instances)),
            "baseline": base, "arm": mine, "control": shuffled,
            "arm_minus_baseline": vs_base,
            "arm_minus_control": vs_ctrl,
            # C36/C38: strict improvement on this structure, per structure and not aggregated
            "delta_above_baseline": bool(mine["delta"] > base["delta"]),
            "satisfy_above_baseline": bool(mine["satisfy_rate"] > base["satisfy_rate"]),
            "delta_above_control": bool(mine["delta"] > shuffled["delta"]),
            "near_not_below_baseline": bool(mine["near_share"] >= base["near_share"]),
            # C20/C44: the per-level split for every arm, so a gain concentrated in one level
            # cannot be read as a gain in the structure's ordering
            "level_delta": {
                "arm": contract.level_measure(arm_codes, kind, instances),
                "baseline": contract.level_measure(base_codes, kind, instances),
                "control": contract.level_measure(control_codes, kind, instances),
            },
            "lower_bound_positive": bool(vs_base["ci_low"] > 0.0),
        }
    out["simultaneous_lower_bounds_positive"] = all(
        record["lower_bound_positive"] for record in out["kinds"].values())
    out["all_deltas_above_baseline"] = all(
        record["delta_above_baseline"] for record in out["kinds"].values())
    out["all_deltas_above_control"] = all(
        record["delta_above_control"] for record in out["kinds"].values())
    out["all_satisfy_rates_above_baseline"] = all(
        record["satisfy_above_baseline"] for record in out["kinds"].values())
    out["all_near_not_below_baseline"] = all(
        record["near_not_below_baseline"] for record in out["kinds"].values())
    arm_health = sid_health(arm_codes)
    base_health = sid_health(base_codes)
    control_health = sid_health(control_codes)
    arm_recon = reconstruction_error(REPO / "stage2_RQ-VAE" / Path(arm).name
                                     / "logs" / "training_metrics.jsonl")
    base_recon = reconstruction_error(BASELINE_SOURCE / "logs" / "training_metrics.jsonl")
    control_recon = reconstruction_error(REPO / "stage2_RQ-VAE" / Path(control).name
                                         / "logs" / "training_metrics.jsonl")
    out["sid_health"] = {"baseline": base_health, "arm": arm_health, "control": control_health}
    out["reconstruction"] = {"baseline": base_recon, "arm": arm_recon, "control": control_recon}
    ratio = None
    if arm_recon["recon_final"] is not None and base_recon["recon_final"]:
        ratio = arm_recon["recon_final"] / base_recon["recon_final"]
    out["recon_ratio_vs_baseline"] = ratio
    out["collision_not_above_baseline"] = bool(
        arm_health["collision"] <= base_health["collision"])
    out["recon_within_5pct"] = None if ratio is None else bool(ratio <= 1.05)
    out["fork_reversed"] = bool(out["kinds"]["fork"]["arm"]["delta"] > 0.0
                                and out["kinds"]["fork"]["baseline"]["delta"] < 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = verdict(args.arm, args.control)
    out = args.out or REPO / "results" / "research" / "issue230" / "verdict_result.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for kind, record in report["kinds"].items():
        mine, base = record["arm"], record["baseline"]
        print(f"{kind:6s} arm {mine['delta']:+.4f} base {base['delta']:+.4f} "
              f"diff {record['arm_minus_baseline']['difference']:+.4f} "
              f"[{record['arm_minus_baseline']['ci_low']:+.4f},"
              f"{record['arm_minus_baseline']['ci_high']:+.4f}] "
              f"satisfy {mine['satisfy_rate']:.3f} vs {base['satisfy_rate']:.3f}")
    health = report["sid_health"]
    print("collision: arm %.4f base %.4f control %.4f"
          % (health["arm"]["collision"], health["baseline"]["collision"],
             health["control"]["collision"]))
    print("recon ratio vs baseline:", report["recon_ratio_vs_baseline"])
    print(json.dumps({key: value for key, value in report.items()
                      if key.startswith("all_") or key in (
                          "fork_reversed", "simultaneous_lower_bounds_positive",
                          "collision_not_above_baseline", "recon_within_5pct")},
                     indent=1))


if __name__ == "__main__":
    main()
