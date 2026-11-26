"""Recompute the ten historical scores from #262 MATRIX and #266 endpoints."""
from __future__ import annotations

import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/research/issue267/analysis/FROZEN_SCORES_10.json"
R_RANGE = (0.00008205277845263481, 0.00010617653606459498)
G_RANGE = (0.0003398736207435491, 0.05941492674715333)
L_RANGE = (0.00004965698388494281, 0.00015923990773577049)
EXPECTED = {101: .2281179533, 104: .4475563464, 100: .5689364228, 107: .5782102943,
            106: .5789591370, 105: .5797449191, 108: .6454995374, 109: .6535990918,
            103: .6698993639, 102: .7411182932}


def n(value, rng):
    return (value - rng[0]) / (rng[1] - rng[0])


def score(r, g, l):
    return (0.52137 * (1 - n(r, R_RANGE))
            + 0.19389 * n(math.log(g), (math.log(G_RANGE[0]), math.log(G_RANGE[1])))
            + 0.28474 * n(math.log(l), (math.log(L_RANGE[0]), math.log(L_RANGE[1]))))


def main():
    rows = json.load(open(ROOT / "results/research/issue262/MATRIX.json"))["rows"]
    vals = {int(row["index"]): (row["grad_loss_reconstruction"],
                                row["grad_curvature_gate"],
                                row["grad_original_codebook2"]) for row in rows}
    for index in (108, 109):
        endpoint = json.load(open(ROOT / f"results/research/issue266/ENDPOINT_{index}.json"))
        vals[index] = (endpoint["losses"]["reconstruction"],
                       endpoint["groups"]["gate"]["curvature_norm"],
                       endpoint["groups"]["codebook2"]["original_norm"])
    result = []
    for index in sorted(vals):
        r, g, l = vals[index]
        s = score(r, g, l)
        result.append({"index": index, "R": r, "G_gate": g, "G_L3": l, "S": s,
                       "expected_issue_score": EXPECTED[index],
                       "abs_error": abs(s - EXPECTED[index])})
    if any(row["abs_error"] > 1e-10 for row in result):
        raise RuntimeError("frozen historical score mismatch: " + json.dumps(result))
    OUT.write_text(json.dumps({"schema": "genrec.issue267.frozen_scores_10/1",
                               "rows": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
