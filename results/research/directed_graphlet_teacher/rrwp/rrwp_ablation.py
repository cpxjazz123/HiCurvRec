#!/usr/bin/env python3
"""What in the RRWP block actually carries direction: degrees, two hops, or the direction?

Four feature sets, one probe, one split, one seed:

  E            embeddings only
  E+deg        + log out-degree of both endpoints          (the popularity control)
  E+deg+P2     + two-hop walk probabilities both ways      (the multi-hop claim)
  E+deg+P1P2   + one- and two-hop
  E+deg+P2rand + two-hop probabilities from a graph whose edge orientations were randomly
                 permuted while the degree sequence was kept -- if the real P2 does no
                 better than P2rand, the signal is degree structure rather than direction
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import augment_gnn as base  # noqa: E402
from rrwp_direction import gather, probe, rrwp, sha256  # noqa: E402

OUT = HERE / "rrwp_ablation.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    transition, two_step, out_degree = rrwp(msg_rows, msg_cols, n_items)

    # direction-randomised control: same endpoints, orientations permuted independently
    rng = np.random.default_rng(base.SEED + 11)
    flip = rng.random(len(msg_rows)) < 0.5
    rand_rows = np.where(flip, msg_cols, msg_rows)
    rand_cols = np.where(flip, msg_rows, msg_cols)
    _, two_step_rand, _ = rrwp(rand_rows, rand_cols, n_items)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]
    both = np.concatenate([one_way, one_way[:, ::-1]], axis=0)
    degree_feature = np.stack([np.log1p(out_degree[both[:, 0]]),
                               np.log1p(out_degree[both[:, 1]])], axis=1)
    p1 = np.stack([gather(transition, both[:, 0], both[:, 1]),
                   gather(transition, both[:, 1], both[:, 0])], axis=1)
    p2 = np.stack([gather(two_step, both[:, 0], both[:, 1]),
                   gather(two_step, both[:, 1], both[:, 0])], axis=1)
    p2_rand = np.stack([gather(two_step_rand, both[:, 0], both[:, 1]),
                        gather(two_step_rand, both[:, 1], both[:, 0])], axis=1)

    states = {"z": np.load(REPO / "results" / "research" / "directed_graphlet_teacher" /
                           "fidelity" / "fidelity_states.npz")["z"],
              "semantics": np.load(base.SEMANTICS)}
    report = {"split": {"message_edges": int(len(msg_rows)), "held_edges": int(len(held_rows)),
                        "held_one_way": int(len(one_way))},
              "protocol": {"probe": "seeded, item-disjoint, 5,514 eval pairs",
                           "rrwp": "from training edges only",
                           "random_control": "same endpoints, orientations permuted"},
              "arms": {}}
    feature_sets = {
        "E": None,
        "E+deg": degree_feature,
        "E+deg+P2": np.concatenate([degree_feature, p2], axis=1),
        "E+deg+P1+P2": np.concatenate([degree_feature, p1, p2], axis=1),
        "E+deg+P2random": np.concatenate([degree_feature, p2_rand], axis=1),
    }
    for state_name, x in states.items():
        for label, extra in feature_sets.items():
            key = f"{state_name}:{label}"
            report["arms"][key] = probe(x, extra, one_way, device)
            print(f"  {key}: {report['arms'][key]['direction_accuracy']:.4f}", flush=True)
    report["arms"]["rrwp_only_no_degree"] = probe(
        np.zeros((n_items, 1), dtype=np.float32), p2, one_way, device)

    se = 0.5 / math.sqrt(report["arms"]["z:E"]["pairs_eval"])
    readings = {"se": round(se, 4)}
    for label in ("E+deg", "E+deg+P2", "E+deg+P2random"):
        readings[f"z:{label}_minus_E_sigma"] = round(
            (report["arms"][f"z:{label}"]["direction_accuracy"]
             - report["arms"]["z:E"]["direction_accuracy"]) / se, 2)
    readings["z_P2_minus_P2random_sigma"] = round(
        (report["arms"]["z:E+deg+P2"]["direction_accuracy"]
         - report["arms"]["z:E+deg+P2random"]["direction_accuracy"]) / se, 2)
    report["readings"] = readings
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(readings, indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
