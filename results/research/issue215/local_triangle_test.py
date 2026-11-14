"""The local test the owner asked for: among a handful of items, which pairs does the model call edges?

The thresholded graph was the wrong instrument: taking each item's thirteen highest-scoring partners
produces a popularity-shaped graph that contains almost none of the real behaviour edges (0.001 of a
triangle's three edges), so it cannot say whether a *local* structure survives.  What the question
really asks is a **ranking** question: inside one neighbourhood, are the real edges ranked above the
non-edges?

So the score of every pair is kept, and the test is run inside behaviour triangles:

  * for every triangle (a, b, c) of the real behaviour graph, the three real pairs are the positive
    set, and the hard negatives are the pairs (a, w), (b, w), (c, w) where ``w`` is a neighbour of
    that item which is not in the triangle -- items one step away, the ones a local model confuses
    with the real ones;
  * the model is asked to rank the six pairs; we count how many of the three real edges it puts
    above the best negative, and above all of them;
  * a triangle counts as recovered when all three of its edges beat all of its negatives.

The real behaviour graph is the answer key only: it is never part of what the model sees.

    python3 results/research/issue215/local_triangle_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GRAPHS = OUT / "graphs"
SCORES = OUT / "scores"
JOURNAL = OUT / "local_triangle_test.json"
ROLE_NAMES = {0: "dense hub", 1: "tight community", 2: "two-hop / fan", 3: "degree middle",
              4: "cycle / long path", 5: "star / chain"}
MAX_TRIANGLES = 100_000
SEED = 42


def load_behaviour() -> tuple[int, np.ndarray, list[set[int]]]:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return n, edges, adjacency


def triangles_of(edges: np.ndarray, adjacency: list[set[int]]) -> np.ndarray:
    found = []
    for a, b in edges.tolist():
        a, b = int(a), int(b)
        for c in adjacency[a] & adjacency[b]:
            if c > b:
                found.append((a, b, c))
    return np.asarray(found, dtype=np.int64)


def load_roles() -> np.ndarray:
    roles = np.full(24_587, -1, dtype=np.int64)
    for line in (OUT / "orca_roles_frozen.csv").read_text().splitlines()[1:]:
        item, role = line.split(",")
        roles[int(item)] = int(role)
    return roles


def main() -> int:
    n, edges, adjacency = load_behaviour()
    triangles = triangles_of(edges, adjacency)
    print(f"behaviour: {len(edges)} edges, {len(triangles)} triangles", flush=True)
    if len(triangles) > MAX_TRIANGLES:
        rng = np.random.default_rng(SEED)
        picked = np.sort(rng.choice(len(triangles), size=MAX_TRIANGLES, replace=False))
        triangles = triangles[picked]
    roles = load_roles()

    report = {"triangles_evaluated": int(len(triangles)), "predictors": {}}
    print(f"\n{'predictor':<14} {'edges above best negative':>26} {'edges above all negs':>20} "
          f"{'triangles recovered':>20}")
    for path in sorted(SCORES.glob("*_scores.npz")):
        name = path.name[:-len("_scores.npz")]
        block = np.load(path)
        score = block["score"]
        pairs = block["pairs"]
        key = pairs[:, 0].astype(np.int64) * n + pairs[:, 1].astype(np.int64)
        lookup = dict(zip(key.tolist(), score.tolist()))

        def value(a: int, b: int) -> float:
            if a > b:
                a, b = b, a
            return lookup.get(a * n + b, float("-inf"))

        above_best, above_all, recovered, chance = [], [], [], []
        per_role = {role: [] for role in sorted(ROLE_NAMES)}
        for a, b, c in triangles.tolist():
            positives = [(a, b), (b, c), (a, c)]
            negatives = []
            for item in (a, b, c):
                candidates = [w for w in adjacency[item] if w not in (a, b, c)][:4]
                for w in candidates:
                    negatives.append((item, w))
            if not negatives:
                continue
            positive_scores = [value(*pair) for pair in positives]
            negative_scores = [value(*pair) for pair in negatives]
            best_negative = max(negative_scores)
            # what a scorer with no information would get here: each of the three real edges beats
            # the best of k local negatives with probability 1/(k+1)
            chance.append(3.0 / (len(negative_scores) + 1.0))
            count_above_best = sum(1 for score in positive_scores if score > best_negative)
            count_above_all = sum(1 for score in positive_scores if score > max(negative_scores))
            above_best.append(count_above_best)
            above_all.append(count_above_all)
            recovered.append(count_above_best == 3)
            for role in sorted(ROLE_NAMES):
                if any(roles[item] == role for item in (a, b, c)):
                    per_role[role].append(count_above_best)

        entry = {
            "edges_above_best_negative_mean": float(np.mean(above_best)),
            "edges_above_all_negatives_mean": float(np.mean(above_all)),
            "triangles_fully_recovered": float(np.mean(recovered)),
            "chance_edges_above_best_negative": float(np.mean(chance)),
            "per_role": {ROLE_NAMES[role]: {"triangles": len(values),
                                           "edges_above_best_negative_mean": float(np.mean(values))}
                         for role, values in per_role.items() if values},
        }
        report["predictors"][name] = entry
        print(f"{name:<14} {entry['edges_above_best_negative_mean']:>10.3f} "
              f"(chance {entry['chance_edges_above_best_negative']:.3f}) "
              f"{entry['edges_above_all_negatives_mean']:>10.3f} "
              f"{entry['triangles_fully_recovered']:>10.4f}", flush=True)
        for role, values in entry["per_role"].items():
            print(f"    {role:<20} triangles {values['triangles']:>7} | edges above the best negative "
                  f"{values['edges_above_best_negative_mean']:.3f}", flush=True)

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
