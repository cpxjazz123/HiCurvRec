"""Compare a mechanism against the bar seed by seed, rather than family to family.

Two arms trained at different seeds differ in initialization, in the PCA basis
applied to the Stage1 embeddings and in the data order, and the spread that
produces is larger than every effect this program has measured. Comparing a
mechanism's family against the bar's family therefore buries the mechanism's
effect inside that spread. Comparing at matched seeds removes it: whatever the
seed did to the bar, it did to the mechanism too, and the difference of the two
is the mechanism's effect plus whatever noise is left.

Three matched pairs is not a significance test, and this script does not pretend
to run one. It reports the three differences, their mean and their spread, which
is enough to say whether a mechanism's gain is consistent in sign or is one seed
away from nothing.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train")

# mechanism arm and the bar arm it should be paired with, per seed
# The claim under test is that the hyperbolic model beats the euclidean bar, so
# every pair is the hyperbolic mechanism against the euclidean bar at the same
# seed. Pairing the euclidean mechanism at seed 42 would have made the first
# difference a different quantity from the other two.
PAIRS = {
    42: ("sidarm_SH_poincare_smooth", "sidarm_E72_euclid_l2_256_72k"),
    43: ("sidarm_M43_smooth_poincare_s43", "sidarm_E43_euclid_l2_256_72k_s43"),
    44: ("sidarm_M44_smooth_poincare_s44", "sidarm_E44_euclid_l2_256_72k_s44"),
}
# The mechanism's effect inside the bar's own geometry, over the same three
# seeds. The hyperbolic effect is smaller than the seed spread, so no number of
# seeds resolves it on its own; the euclidean effect is nearly four times as
# large and the same size as the spread, so this is the side a paired test has a
# chance of settling.
EUCLIDEAN_PAIRS = {
    42: ("sidarm_SE_euclid_smooth", "sidarm_E72_euclid_l2_256_72k"),
    43: ("sidarm_M43E_smooth_euclid_s43", "sidarm_E43_euclid_l2_256_72k_s43"),
    44: ("sidarm_M44E_smooth_euclid_s44", "sidarm_E44_euclid_l2_256_72k_s44"),
}


def _test_recall(variant: str) -> float | None:
    for metrics in sorted(glob.glob(str(RESULTS / variant / "logs" / "*" / "*" / "training_metrics.jsonl"))):
        last = None
        for line in Path(metrics).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("event") == "test":
                    last = row
        if last is not None:
            return float(last["test_recall_at_10"])
    return None


def _report(label: str, pairs: dict[int, tuple[str, str]]) -> list[float]:
    differences = []
    print(f"=== {label} ===")
    print(f"{'seed':>5} {'mechanism':>12} {'bar':>10} {'difference':>12}")
    for seed, (mechanism, bar) in sorted(pairs.items()):
        m = _test_recall(mechanism)
        b = _test_recall(bar)
        if m is None or b is None:
            print(f"{seed:>5} {'pending' if m is None else f'{m:.6f}':>12} "
                  f"{'pending' if b is None else f'{b:.6f}':>10} {'-':>12}")
            continue
        differences.append(m - b)
        print(f"{seed:>5} {m:>12.6f} {b:>10.6f} {m - b:>+12.6f}")
    if differences:
        mean = sum(differences) / len(differences)
        print(f"paired differences: n={len(differences)} mean={mean:+.6f} "
              f"min={min(differences):+.6f} max={max(differences):+.6f} "
              f"all positive={all(d > 0 for d in differences)}")
        if len(differences) > 1:
            print(f"spread across seeds: {max(differences) - min(differences):.6f} "
                  f"({100 * (max(differences) - min(differences)) / abs(mean):.1f}% of the mean)")
    print()
    return differences


def main() -> None:
    _report("hyperbolic mechanism minus euclidean bar", PAIRS)
    _report("euclidean mechanism minus euclidean bar", EUCLIDEAN_PAIRS)


if __name__ == "__main__":
    main()