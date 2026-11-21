#!/usr/bin/env python3
"""Apply the pre-registered success criteria to the 4000-step control/supervised pair."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE / "prototype_chain_long_control.json"
WIRED = HERE / "prototype_chain_long_wired.json"
OUT = HERE / "prototype_chain_long_verdict.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    control, wired = load(CONTROL), load(WIRED)
    steps = sorted({int(key) for key in control["curve"]}, key=int)
    rows = []
    for step in steps:
        c = control["curve"][str(step)]
        w = wired["curve"].get(str(step), {})
        rows.append({
            "step": step,
            "control": {key: c[key] for key in
                        ("chain_delta", "collision", "semantic_r2",
                         "L1_adjacent", "L1_far", "L2_adjacent", "L2_far")},
            "wired": {key: w[key] for key in
                      ("chain_delta", "collision", "semantic_r2",
                       "L1_adjacent", "L1_far", "L2_adjacent", "L2_far")} if w else None})
    final_control = rows[-1]["control"]
    final_wired = rows[-1]["wired"] or {}
    criteria = {
        "chain_delta_above_previous_best": final_wired.get("chain_delta", 0) > 0.0206,
        "chain_delta_beats_control": final_wired.get("chain_delta", -1)
        > final_control["chain_delta"],
        "collision_at_most_previous": final_wired.get("collision", 1.0) <= 0.1192,
        "collision_close_to_control": final_wired.get("collision", 1.0)
        <= 1.5 * final_control["collision"],
        "semantic_r2_at_least_95pct_of_control": final_wired.get("semantic_r2", 0)
        >= 0.95 * final_control["semantic_r2"],
    }
    verdict = {"rows": rows, "final": {"control": final_control, "wired": final_wired},
               "criteria": criteria, "passed": all(criteria.values())}
    OUT.write_text(json.dumps(verdict, indent=1) + "\n")
    header = ("step | ctrl Δ | wired Δ | ctrl coll | wired coll | "
              "ctrl R2 | wired R2 | wired L1 adj/far | wired L2 adj/far")
    print(header)
    for row in rows:
        c, w = row["control"], row["wired"] or {}
        print(f"{row['step']:>5} | {c['chain_delta']:+.4f} | "
              f"{w.get('chain_delta', float('nan')):+.4f} | {c['collision']:.4f} | "
              f"{w.get('collision', float('nan')):.4f} | {c['semantic_r2']:.3f} | "
              f"{w.get('semantic_r2', float('nan')):.3f} | "
              f"{w.get('L1_adjacent', float('nan')):.4f}/{w.get('L1_far', float('nan')):.4f} | "
              f"{w.get('L2_adjacent', float('nan')):.4f}/{w.get('L2_far', float('nan')):.4f}")
    print("criteria:", criteria)
    print("passed:", verdict["passed"], "->", OUT)


if __name__ == "__main__":
    main()
