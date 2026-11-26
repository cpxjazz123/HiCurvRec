"""Run the five Issue #267 relative-term mutations against named assertions."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq/model/relative.py"
OUT = ROOT / "results/research/issue267/analysis/MUTATION_RESULTS.json"


def load(source: str):
    module = types.ModuleType("mutant_relative")
    module.__dict__.update({"__name__": "mutant_relative", "math": math,
                            "torch": torch, "F": F})
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


def check(mutation: str, source: str, assertion: str):
    module = load(source)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)
    if "sign" in mutation:
        correct = module.relative_rank_loss(torch.tensor([0.0, 2.0]), target, [0, 0], tau=1.0)
        wrong = module.relative_rank_loss(torch.tensor([0.0, -2.0]), target, [0, 0], tau=1.0)
        caught = not correct["loss"] < wrong["loss"]
    elif "tie" in mutation:
        result = module.relative_rank_loss(torch.tensor([0.0, 100.0, 0.0]),
                                           torch.tensor([0.0, 0.0, 1.0]),
                                           [0, 0, 0], tau=1.0)
        caught = not (result["valid_pairs"] == 2 and result["tied_pairs"] == 1)
    elif "centre" in mutation:
        result = module.relative_rank_loss(torch.tensor([0.0, 1.0, 0.0]),
                                           torch.tensor([0.0, 2.0, 3.0]),
                                           [0, 1, 2], tau=1.0)
        caught = not (result["valid_pairs"] == 0 and float(result["loss"]) == 0.0)
    elif "diagonal" in mutation:
        result = module.relative_rank_loss(torch.arange(28, dtype=torch.float64),
                                           torch.arange(28, dtype=torch.float64), [0] * 28)
        caught = not (result["all_pairs"] == 378 and result["valid_pairs"] == 378)
    elif "sum" in mutation:
        result = module.relative_rank_loss(torch.tensor([0.0, 100.0, 0.0]),
                                           torch.tensor([0.0, 0.0, 1.0]),
                                           [0, 0, 0], tau=1.0)
        caught = not abs(float(result["loss"]) - ((math.log(2.0) + 100.0) / 2.0)) < 1e-6
    else:
        raise ValueError(mutation)
    return {"mutation": mutation, "assertion": assertion, "caught": bool(caught)}


def main():
    source = SOURCE.read_text()
    mutations = [
        ("sign", source.replace("F.softplus(-sign * delta_predicted[valid] / float(tau))",
                                "F.softplus(sign * delta_predicted[valid] / float(tau))"),
         "test_known_sign_and_hand_value: correct loss < wrong loss"),
        ("tie", source.replace("valid = same & finite & (~tied)", "valid = same & finite"),
         "test_ties_are_excluded_not_averaged_in: valid=2, tied=1"),
        ("centre", source.replace("same = centre[left] == centre[right]",
                                  "same = torch.ones_like(centre[left], dtype=torch.bool)"),
         "test_cross_centre_pairs_are_never_used: valid=0, loss=0"),
        ("diagonal", source.replace("offset=1,", "offset=0,"),
         "test_upper_triangle_pair_count_28_rows: 378 not 406"),
        ("sum", source.replace("F.softplus(-sign * delta_predicted[valid] / float(tau)).mean()",
                               "F.softplus(-sign * delta_predicted[valid] / float(tau)).sum()"),
         "test_known_sign_and_hand_value: exact softplus(-2)"),
    ]
    results = [check(name, mutated, assertion) for name, mutated, assertion in mutations]
    if any(not result["caught"] for result in results):
        raise RuntimeError("one or more mutations escaped: " + json.dumps(results))
    OUT.write_text(json.dumps({"schema": "genrec.issue267.mutations/1", "results": results}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
