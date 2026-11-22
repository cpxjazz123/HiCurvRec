"""Optional descriptive diagnostic: curvature groups versus the frozen TIGER SID.

Issue #235 C21 asks, *if* the frozen TIGER SID is available, for the L1 / L1-L2 /
L1-L2-L3 prefix sharing and collision inside curvature groups.  It is explicitly
descriptive -- no retraining, no claim that curvature explains the SID.

Two questions are answered separately because they are easy to confuse:

* **prefix sharing** -- within a curvature group, how often do two random items share
  the first k codes?  Compared against the whole catalogue, which is the honest
  baseline for "sharing is higher because the group is small".
* **collision** -- how many items in the group fall into a shared three-code bucket.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue235/run_sid_diagnostic.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue235"
SID = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
SEED = 42
#: Pairs drawn per group for the sharing estimate; a group much smaller than this uses
#: all of its pairs.
PAIRS_PER_GROUP = 20000


def main() -> int:
    sids = {int(key): value for key, value in json.loads(SID.read_text(encoding="utf-8")).items()}
    curvature = {}
    for line in (OUT / "node_curvature.csv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if parts[3]:
            curvature[int(parts[0])] = float(parts[3])
    print(f"{len(sids)} SIDs, {len(curvature)} items with a curvature mean", flush=True)

    groups = _groups(curvature)
    whole = _sharing(sids, list(sids), PAIRS_PER_GROUP)
    report = {
        "sid_path": str(SID.relative_to(REPO)),
        "items_with_curvature": len(curvature),
        "groups": {},
        "whole_catalogue": whole,
        "seed": SEED,
        "pairs_per_group": PAIRS_PER_GROUP,
        "notes": [
            "descriptive only: curvature is not used to fit anything and no model is trained",
            "sharing is per group and must be read against the whole-catalogue column, "
            "because a small group shares more by construction",
            "the SID is the frozen TIGER baseline (CLAUDE.md 25); it is not retrained here",
        ],
    }
    for name, members in groups.items():
        pool = [item for item in members if item in sids]
        if len(pool) < 2:
            continue
        report["groups"][name] = {
            "items": len(pool),
            "sharing": _sharing(sids, pool, PAIRS_PER_GROUP),
            "collision": _collision(sids, pool),
        }
    path = OUT / "sid_diagnostic.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"-> {path}")
    return 0


def _groups(curvature: dict[int, float]) -> dict[str, list[int]]:
    values = sorted(curvature.values())
    n = len(values)
    return {
        "negative": [item for item, v in curvature.items() if v < -1e-9],
        "near_zero": [item for item, v in curvature.items() if abs(v) <= 1e-9],
        "positive": [item for item, v in curvature.items() if v > 1e-9],
        "bottom_decile": [item for item, v in curvature.items() if v <= values[n // 10]],
        "top_decile": [item for item, v in curvature.items() if v >= values[-n // 10 - 1]],
    }


def _sharing(sids: dict[int, list], pool: list[int], draws: int) -> dict:
    """Share of random pairs agreeing on the first 1 / 2 / 3 codes."""
    rng = random.Random(SEED)
    total = len(pool) * (len(pool) - 1) // 2
    pairs = _all_pairs(pool) if total <= draws else [
        tuple(rng.sample(pool, 2)) for _ in range(draws)]
    counts = Counter()
    for left, right in pairs:
        a, b = sids[left], sids[right]
        if a[0] == b[0]:
            counts["l1"] += 1
            if a[1] == b[1]:
                counts["l2"] += 1
                if a[2] == b[2]:
                    counts["l3"] += 1
    denominator = max(len(pairs), 1)
    return {
        "pairs": len(pairs),
        "exact_pairs": total,
        "sampled": total > draws,
        "l1_share": counts["l1"] / denominator,
        "l2_share": counts["l2"] / denominator,
        "l3_share": counts["l3"] / denominator,
    }


def _all_pairs(pool: list[int]) -> list[tuple[int, int]]:
    return [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]


def _collision(sids: dict[int, list], pool: list[int]) -> dict:
    buckets = Counter(tuple(sids[item][:3]) for item in pool)
    duplicated = sum(1 for count in buckets.values() if count > 1)
    in_collision = sum(count for count in buckets.values() if count > 1)
    return {
        "distinct_buckets": len(buckets),
        "duplicated_buckets": duplicated,
        "items_in_collision": in_collision,
        "collision_fraction": 1.0 - len(buckets) / max(len(pool), 1),
        "largest_bucket": max(buckets.values()) if buckets else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
