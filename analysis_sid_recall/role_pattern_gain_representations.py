"""The same per-role pattern gain in z, q and the SID, so the quantization change is visible per role.

Uses exactly the instances and matched null groups of `role_pattern_gain.py` (through its `prepare()`),
so the delta between two representations is a paired difference on one instance set.

    delta_quant = G_q - G_z        what vector quantization did to the role's structure gain
    delta_SID   = G_SID - G_z      what the discrete identifier did to it

TIGER is the one arm in the repository with a pre-quantization vector, so this table is single-arm; the
correlation column comes from `role_pattern_gain.json`, which spans the 67 archived arms.

    python3 analysis_sid_recall/role_pattern_gain_representations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import role_pattern_gain as rpg  # noqa: E402

VECTORS = rpg.m.VECTORS
OUT = HERE / "role_pattern_gain_representations.json"
ARMS = ("TIGER",)
WARNING = "single arm: only TIGER keeps a pre-quantization vector"


def main() -> int:
    prepared = rpg.prepare()
    patterns = prepared["patterns"]
    subsets = prepared["subsets"]
    incidences = prepared["incidences"]
    role_of = prepared["role_of"]
    patterns_by_role = prepared["patterns_by_role"]

    result = {"arms": list(ARMS), "note": WARNING, "roles": {}}
    per_arm = {}
    for arm in ARMS:
        z = np.load(VECTORS / f"{arm}_z.npy").astype(np.float64)
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)

        def euclidean(left: int, right: int, matrix: np.ndarray) -> float:
            return float(np.linalg.norm(matrix[left] - matrix[right]))

        def lcp_distance(left: int, right: int) -> float:
            matches = 0
            for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
                if x != y:
                    break
                matches += 1
            return 1.0 - matches / 3.0

        distances = {"z": lambda left, right: euclidean(left, right, z),
                     "q": lambda left, right: euclidean(left, right, q),
                     "SID": lcp_distance}

        per_arm[arm] = {}
        for representation, distance in distances.items():
            per_pattern = {}
            for name in patterns:
                chosen, blocks = subsets[name]
                values_real, values_null = [], []
                for instance, draws in zip(chosen, blocks):
                    if not draws:
                        continue
                    computed = rpg.per_instance(name, instance, draws, distance)
                    if computed is None:
                        continue
                    values_real.append(computed[0])
                    values_null.append(computed[1])
                per_pattern[name] = (np.array(values_real), np.array(values_null),
                                     incidences[name])
            per_arm[arm][representation] = {
                str(role): rpg.role_gain(per_pattern, patterns_by_role[role], role, role_of)
                for role in range(6)}

    sid_correlations = json.loads((HERE / "role_pattern_gain.json").read_text())["roles"]
    print(f"\n=== per-role structure gain in z / q / SID ({ARMS[0]}); "
          f"correlation from the 67-arm SID series ===")
    print(f"  {'role':<8}{'patterns':<6}{'G_z':>8}{'G_q':>8}{'G_SID':>8}"
          f"{'d_quant':>9}{'d_SID':>8}{'rho(SID,recall)':>17}{'r':>8}")
    for role in range(6):
        values = {representation: per_arm[ARMS[0]][representation][str(role)]
                  for representation in ("z", "q", "SID")}
        correlation = sid_correlations.get(str(role), {})
        if any(value is None for value in values.values()):
            print(f"  Role {role + 1:<3}{'n/a':<6}")
            continue
        delta_quant = values["q"] - values["z"]
        delta_sid = values["SID"] - values["z"]
        result["roles"][str(role)] = {
            "label": f"Role {role + 1}", "patterns": patterns_by_role[role],
            **{f"G_{key}": value for key, value in values.items()},
            "delta_quant": delta_quant, "delta_SID": delta_sid,
            "correlation_with_recall": correlation,
        }
        print(f"  Role {role + 1:<3}{len(patterns_by_role[role]):<6}"
              f"{values['z']:>+8.3f}{values['q']:>+8.3f}{values['SID']:>+8.3f}"
              f"{delta_quant:>+9.3f}{delta_sid:>+8.3f}"
              f"{correlation.get('spearman', float('nan')):>+17.3f}"
              f"{correlation.get('pearson', float('nan')):>+8.3f}")
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(rpg.m.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
