"""The one analysis script for the SID-versus-structure study: every method behind one entry point.

The six ORCA roles come from the repository's own definition: ORCA's 73 node orbits of graphlets up
to five nodes on the train behaviour graph, grouped into six NMF roles (`structure_roles5.npz`).  The
nine five-node patterns and their hypotheses are fixed here, once, and every command reuses them.

Commands, each printing a table and optionally writing its own JSON:

    roles        level-1/2/3 prefix retention per role across the archived arms, and its correlation
                 with test_recall@10
    distance     where a role's behaviour neighbours sit in the SID ordering: win rate against one
                 random item and a percentile over 200 of them
    patterns     per-pattern SID structure gain per arm, and per-role aggregated rate and gain, in
                 the SID and (single arm) in the quantized and pre-quantization vectors
    independence out-of-sample ridge on the six role rates: CV R^2, marginal contribution of each
                 role, permutation p-values
    k            the same rates under other NMF granularities, plus how the cached Role 3 is cut
    profile      what one role's items look like: graph invariants, ORCA orbits that separate them,
                 and which patterns they take part in

Everything is deterministic given --seed.  The heavy knobs have limits so a check run is cheap:
--arms, --instances, --null-draws, --permutations, --repeats, --edges.

    python3 analysis_sid_recall/analysis.py --help
    python3 analysis_sid_recall/analysis.py roles
    python3 analysis_sid_recall/analysis.py patterns --instances 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"        # every table and artifact this script writes lands here
REPO = HERE.parent
MIRROR = HERE / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
VECTORS = REPO / "results" / "research" / "issue215" / "vectors"
BEHAVIOUR = REPO / "results" / "research" / "issue215" / "graphs" / "behaviour.in"
ROLE_CACHE = HERE / "structure_roles5.npz"
FEATURE_BLOCK = HERE / "feature_block.npz"
ORBITS = REPO / "results" / "research" / "issue213" / "orca" / "node5.out"
# The role system is K=6 and nothing else: a k sweep asked the arms to be explained under partitions
# the report never uses.  materialised from the single cache below.
ROLE_K = 6
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
RECALL = "test_recall@10"

SEED = 42
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")
MIN_ITEMS = 30
BUCKETS = 5
CANDS = 200


def output_path(args, name):
    """Where a command writes: --out when given, otherwise results/<command>.json."""
    path = Path(args.out) if args.out else RESULTS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- statistics

def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    average = (starts + ends) / 2.0
    ranks = np.empty(len(values))
    ranks[order] = np.repeat(average, ends - starts + 1)
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a, b = average_ranks(first), average_ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first - first.mean(), second - second.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def partial(first, second, control, rank=True):
    if rank:
        first, second, control = average_ranks(first), average_ranks(second), average_ranks(control)
    design = np.column_stack([np.ones_like(control), control])
    residual_a = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    residual_b = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return pearson(residual_a, residual_b)


def permutation_p(first, second, observed, permutations):
    rng = np.random.default_rng(SEED)
    beats_s = beats_r = 0
    for _ in range(permutations):
        shuffled = rng.permutation(second)
        if abs(spearman(first, shuffled)) >= abs(observed["spearman"]):
            beats_s += 1
        if abs(pearson(first, shuffled)) >= abs(observed["pearson"]):
            beats_r += 1
    return {"spearman": (beats_s + 1) / (permutations + 1),
            "pearson": (beats_r + 1) / (permutations + 1)}


def correlate(values, recalls, collisions, permutations):
    usable = np.isfinite(values)
    first, second, control = values[usable], recalls[usable], collisions[usable]
    if first.size < 5:
        return {"n": int(first.size), "spearman": None}
    observed = {"spearman": spearman(first, second), "pearson": pearson(first, second)}
    p_values = permutation_p(first, second, observed, permutations)
    return {"n": int(first.size), **observed, "p_spearman": p_values["spearman"],
            "p_pearson": p_values["pearson"],
            "partial_spearman": partial(first, second, control),
            "partial_pearson": partial(first, second, control, rank=False)}


def wilson(successes, total, z=1.96):
    if not total:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (centre - half, centre + half)


# ---------------------------------------------------------------- ridge and cross-validation

def standardise(matrix, mean=None, scale=None):
    if mean is None:
        mean = matrix.mean(axis=0)
    if scale is None:
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
    return (matrix - mean) / scale, mean, scale


def fit_ridge(matrix, target, penalty):
    intercept = float(target.mean())
    gram = matrix.T @ matrix + penalty * np.eye(matrix.shape[1])
    return intercept, np.linalg.solve(gram, matrix.T @ (target - intercept))


def predict_ridge(matrix, intercept, coefficients):
    return intercept + matrix @ coefficients


def fold_plan(size, repeats, count, seed):
    """One fixed split per repeat, reused by every model so the comparison is paired."""
    rng = np.random.default_rng(seed)
    plan = []
    for _ in range(repeats):
        order = rng.permutation(size)
        plan.append([order[index::count] for index in range(count)])
    return plan


def inner_penalty(matrix, target, rng, lambdas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    best, best_error = lambdas[0], float("inf")
    split = fold_plan(len(target), 1, 4, int(rng.integers(1 << 30)))[0]
    for penalty in lambdas:
        errors = []
        for test in split:
            train = np.setdiff1d(np.arange(len(target)), test)
            scaled, mean, scale = standardise(matrix[train])
            intercept, coefficients = fit_ridge(scaled, target[train], penalty)
            held = (matrix[test] - mean) / scale
            errors.append(np.mean((target[test] - predict_ridge(held, intercept, coefficients)) ** 2))
        error = float(np.mean(errors))
        if error < best_error:
            best, best_error = penalty, error
    return best


def cross_validate(matrix, target, plan, seed):
    """Out-of-fold R^2 with a fixed split plan and a deterministic penalty search inside each fold."""
    scores = []
    for repeat_index, splits in enumerate(plan):
        predicted = np.full(len(target), np.nan)
        for fold_index, test in enumerate(splits):
            train = np.setdiff1d(np.arange(len(target)), test)
            if matrix.shape[1] == 0:
                predicted[test] = float(target[train].mean())
                continue
            rng = np.random.default_rng(seed + 1_000_003 * repeat_index + 97 * fold_index)
            penalty = inner_penalty(matrix[train], target[train], rng)
            scaled, mean, scale = standardise(matrix[train])
            intercept, coefficients = fit_ridge(scaled, target[train], penalty)
            predicted[test] = predict_ridge((matrix[test] - mean) / scale, intercept, coefficients)
        residual = float(np.sum((target - predicted) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        scores.append(1.0 - residual / total)
    return {"r2_mean": float(np.mean(scores)), "r2_sd": float(np.std(scores))}


# ---------------------------------------------------------------- graph, roles, features

def load_graph():
    with BEHAVIOUR.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency = [set() for _ in range(n)]
    for left, right in edges.tolist():
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    return n, adjacency, edges


def load_roles(n, adjacency=None, overlap_mode="mean", cache=None,
               scaling=None, normalise=None, seed=None):
    """Read the six-role cache, refusing one that does not match the graph and settings it claims.

    A cache keyed only on the node count can survive a changed graph or a different overlap mode, and
    then every analysis silently reports roles from another configuration.  When the adjacency is
    available the fingerprint is recomputed and compared; a mismatch stops the run instead.
    """
    cache = Path(cache) if cache else ROLE_CACHE
    if not cache.exists():
        raise SystemExit(f"no role cache at {cache}; run: analysis.py cache")
    cached = np.load(cache)
    if adjacency is not None:
        if "fingerprint" not in cached:
            raise SystemExit(f"role cache {cache.name} predates the fingerprint; run: analysis.py cache")
        expected = feature_fingerprint(adjacency, ORBITS, overlap_mode)[:16]
        if str(cached["fingerprint"]) != expected:
            raise SystemExit(
                f"role cache {cache.name} is stale: fingerprint {cached['fingerprint']} != {expected}; "
                f"run: analysis.py cache (or pass the matching --overlap-mode)")
        for key, wanted in (("scaling", scaling), ("normalise", normalise), ("seed", seed)):
            if wanted is None or key not in cached:
                continue
            stored = str(cached[key]) if key != "normalise" else int(cached[key])
            have = str(wanted) if key != "normalise" else int(bool(wanted))
            if stored != have:
                raise SystemExit(f"role cache {cache.name} was fitted with {key}={stored}, "
                                 f"not {have}; refit with: analysis.py cache --{key}")
    role_of = np.full(n, -1, dtype=np.int64)
    for node, role in zip(cached["nodes"], cached["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)
    return role_of


def load_popularity(n):
    import pandas as pd

    counts = pd.read_parquet(TRAIN, columns=["target"])["target"].value_counts()
    popularity = np.zeros(n, dtype=np.float64)
    for item, count in counts.items():
        if 0 <= int(item) < n:
            popularity[int(item)] = float(count)
    return popularity


def bucketise(values, buckets):
    """Quantile buckets cut on the *values*, so two items with the same degree and popularity always
    land in the same cell.  The earlier version bucketed by sorted position, which could split a tie
    purely because of item id."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    sorted_values = np.sort(values)
    edges = []
    for quantile in range(1, buckets):
        position = int(round(quantile * values.size / buckets))
        position = min(max(position, 0), values.size - 1)
        edges.append(sorted_values[position])
    edges = np.unique(np.asarray(edges, dtype=np.float64))
    if edges.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def enumerate_cliques(adjacency, size):
    base = [(node,) for node in range(len(adjacency))]
    for depth in range(2, size + 1):
        grown = []
        for clique in base:
            last = clique[-1]
            for candidate in adjacency[last]:
                if candidate <= last:
                    continue
                if all(candidate in adjacency[member] for member in clique[:-1]):
                    grown.append(clique + (candidate,))
        base = grown
        if not base:
            break
    return base


def build_patterns(adjacency, rng, cliques4, cliques5, max_instances):
    n = len(adjacency)
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    pools = {name: {} for name in PATTERNS}

    def pick(values):
        values = list(values)
        return int(values[rng.integers(len(values))])

    def keep(name, nodes, block):
        if len(set(nodes)) != len(nodes):
            return
        key = frozenset(nodes)
        if key in pools[name] or len(pools[name]) >= max_instances:
            return
        pools[name][key] = {"nodes": list(nodes), **block}

    def adjacent(left, right):
        return right in adjacency[left]

    picks = rng.choice(len(cliques5), size=min(len(cliques5), max_instances), replace=False)
    for index in sorted(picks.tolist()):
        keep("K5", list(cliques5[index]), {"kind": "tight"})

    attempts = 0
    while attempts < 200_000 and len(pools["K4_tail"]) < max_instances:
        attempts += 1
        core = list(cliques4[int(rng.integers(len(cliques4)))])
        attachment = int(core[int(rng.integers(4))])
        others = [member for member in core if member != attachment]
        tails = [w for w in adjacency[attachment]
                 if w not in core and not any(adjacent(w, other) for other in others)]
        if not tails:
            continue
        tail = int(tails[int(rng.integers(len(tails)))])
        nodes = core + [tail]
        keep("K4_tail", nodes, {"core": list(combinations(range(4), 2)),
                                "tail": [(4, position) for position in range(4)
                                         if nodes[position] != attachment]})

    for _ in range(200_000):
        if len(pools["double_triangle"]) >= max_instances:
            break
        hub = pick(range(n))
        if len(neighbours[hub]) < 4:
            continue
        picks = rng.choice(len(neighbours[hub]), size=4, replace=False)
        leaves = [neighbours[hub][int(index)] for index in picks]
        first, second = leaves[:2], leaves[2:]
        if not adjacent(*first) or not adjacent(*second):
            continue
        if any(adjacent(x, y) for x in first for y in second):
            continue
        keep("double_triangle", [hub] + leaves, {
            "inside": [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)],
            "cross": [(1, 3), (1, 4), (2, 3), (2, 4)]})

    for _ in range(200_000):
        if len(pools["K2_3"]) >= max_instances:
            break
        a = pick(range(n))
        if len(neighbours[a]) < 3:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        shared = [neighbours[a][int(index)] for index in picks]
        if any(adjacent(x, y) for x, y in combinations(shared, 2)):
            continue                      # the same side must be an independent set
        candidates = [w for w in neighbours[shared[0]]
                      if w != a and all(adjacent(w, other) for other in shared[1:])
                      and not adjacent(w, a)]
        if not candidates:
            continue
        keep("K2_3", [a, pick(candidates)] + shared, {
            "cross": [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
            "same": [(0, 1), (2, 3), (2, 4), (3, 4)]})

    for _ in range(200_000):
        if len(pools["P5_centre"]) >= max_instances:
            break
        a = pick(range(n))
        if not neighbours[a]:
            continue
        b = pick(neighbours[a])
        rest_b = [w for w in neighbours[b] if w != a and not adjacent(w, a)]
        if not rest_b:
            continue
        c = pick(rest_b)
        rest_c = [w for w in neighbours[c] if w not in (a, b) and not adjacent(w, b)
                  and not adjacent(w, a)]
        if not rest_c:
            continue
        d = pick(rest_c)
        rest_d = [w for w in neighbours[d]
                  if w not in (a, b, c) and not adjacent(w, c) and not adjacent(w, b)
                  and not adjacent(w, a)]
        if not rest_d:
            continue
        e = pick(rest_d)
        nodes = [a, b, c, d, e]
        keep("P5_centre", nodes, {"hop1": [(2, 1), (2, 3)], "hop2": [(2, 0), (2, 4)]})
        keep("P5_hops", nodes, {"D1": [(0, 1), (1, 2), (2, 3), (3, 4)],
                                "D2": [(0, 2), (1, 3), (2, 4)],
                                "D3": [(0, 3), (1, 4)], "D4": [(0, 4)]})

    for _ in range(200_000):
        if len(pools["C5"]) >= max_instances:
            break
        a = pick(range(n))
        if len(neighbours[a]) < 2:
            continue
        b = pick(neighbours[a])
        rest_b = [w for w in neighbours[b] if w != a and not adjacent(w, a)]
        if not rest_b:
            continue
        c = pick(rest_b)
        rest_c = [w for w in neighbours[c] if w not in (a, b) and not adjacent(w, b)
                  and not adjacent(w, a)]
        if not rest_c:
            continue
        d = pick(rest_c)
        rest_d = [w for w in neighbours[d] if w not in (a, b, c) and adjacent(w, a)
                  and not adjacent(w, b) and not adjacent(w, c)]
        if not rest_d:
            continue
        keep("C5", [a, b, c, d, pick(rest_d)], {
            "adjacent": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
            "skip": [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]})

    for _ in range(200_000):
        if len(pools["K1_4"]) >= max_instances:
            break
        centre = pick(range(n))
        if len(neighbours[centre]) < 4:
            continue
        picks = rng.choice(len(neighbours[centre]), size=4, replace=False)
        leaves = [neighbours[centre][int(index)] for index in picks]
        if any(adjacent(x, y) for x, y in combinations(leaves, 2)):
            continue
        keep("K1_4", [centre] + leaves, {"spoke": [(0, 1), (0, 2), (0, 3), (0, 4)],
                                         "leaf": list(combinations(range(1, 5), 2))})

    for _ in range(200_000):
        if len(pools["T5"]) >= max_instances:
            break
        centre = pick(range(n))
        if len(neighbours[centre]) < 3:
            continue
        picks = rng.choice(len(neighbours[centre]), size=3, replace=False)
        arm = [neighbours[centre][int(index)] for index in picks]
        if any(adjacent(x, y) for x, y in combinations(arm, 2)):
            continue
        extension = [w for w in neighbours[arm[0]]
                     if w != centre and not adjacent(w, centre)
                     and not adjacent(w, arm[1]) and not adjacent(w, arm[2])]
        if not extension:
            continue                     # a tip touching a second arm is not a T5 tree
        keep("T5", [centre] + arm + [pick(extension)], {
            "hop1": [(0, 1), (0, 2), (0, 3)], "hop2": [(0, 4)]})

    return {name: list(blocks.values()) for name, blocks in pools.items()}


def build_nulls(patterns, cell_of, cell_pool, rng, draws):
    out = {}
    for name, instances in patterns.items():
        blocks = []
        for instance in instances:
            real = set(instance["nodes"])
            sets = []
            for _ in range(draws):
                drawn = []
                for node in instance["nodes"]:
                    pool = cell_pool[int(cell_of[node])]
                    if pool.size == 0:
                        continue
                    for _ in range(64):
                        candidate = int(pool[rng.integers(pool.size)])
                        if candidate in real or candidate in drawn:
                            continue
                        drawn.append(candidate)
                        break
                if len(drawn) == len(instance["nodes"]):
                    sets.append(tuple(drawn))
            blocks.append(sets)
        out[name] = blocks
    return out


def psi(left, right):
    if left < right:
        return 1.0
    if left == right:
        return 0.5
    return 0.0


def hypothesis(pattern, distances):
    mean = lambda values: float(np.mean(values))  # noqa: E731
    if pattern in ("K5", "K4_tail", "K1_4", "T5", "P5_centre", "C5", "double_triangle", "K2_3"):
        return psi(mean(distances["near"]), mean(distances["far"]))
    if pattern == "P5_hops":
        return float(np.mean([psi(mean(distances[key]), mean(distances[next_key]))
                              for key, next_key in (("D1", "D2"), ("D2", "D3"), ("D3", "D4"))]))
    raise KeyError(pattern)


#: which stored group plays "closer" and which "farther" for each pattern
NEAR_FAR = {"double_triangle": ("inside", "cross"), "K4_tail": ("core", "tail"),
            "K2_3": ("cross", "same"), "P5_centre": ("hop1", "hop2"),
            "C5": ("adjacent", "skip"), "K1_4": ("spoke", "leaf"), "T5": ("hop1", "hop2")}


def instance_distances(pattern, instance, nodes, distance):
    """Pair distances for one instance, taken from the groups its builder stored.

    The groups are stored as *positions inside this instance*, so a pattern whose shape varies --
    K4 with the tail on any of the four corners, for instance -- is scored on its own topology instead
    of on a fixed template.
    """
    if pattern == "K5":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in combinations(range(5), 2)]}
    groups = instance            # the builder stores each role group as a top-level key
    if pattern == "P5_hops":
        return {key: [distance(nodes[a], nodes[b]) for a, b in groups[key]]
                for key in ("D1", "D2", "D3", "D4")}
    near_key, far_key = NEAR_FAR[pattern]
    out = {"near": [distance(nodes[a], nodes[b]) for a, b in groups[near_key]],
           "far": [distance(nodes[a], nodes[b]) for a, b in groups[far_key]]}
    for key in ("D1", "D2", "D3", "D4"):
        if key in groups:
            out[key] = [distance(nodes[a], nodes[b]) for a, b in groups[key]]
    return out


def comparisons(pattern, groups):
    """The orderings the pattern asserts, as psi values on group means (1 = holds, 0.5 = tie, 0 = not).

    These are exactly the comparisons the loose score averages, so their conjunction is the strict
    reading of the same hypothesis: every ordering the structure asserts has to hold at once.
    """
    if pattern == "P5_hops":
        keys = ("D1", "D2", "D3", "D4")
        return [psi(float(np.mean(groups[keys[index]])), float(np.mean(groups[keys[index + 1]])))
                for index in range(3)]
    return [psi(float(np.mean(groups["near"])), float(np.mean(groups["far"])))]


def conj_indicator(pattern, groups):
    """1 when every asserted ordering holds (no 2/3 credit), 0 otherwise."""
    return float(all(value == 1.0 for value in comparisons(pattern, groups)))


def strict_indicator(pattern, groups):
    """1 when the *whole* ordering holds pair by pair, 0 otherwise.

    The loose score compares group means (near mean < far mean, or the three consecutive hops of a
    P5), so an instance where two of three orderings hold earns two thirds.  This asks the strict
    question the review raised: no pair at a farther level may be closer than any pair at a nearer one.
    """
    if pattern == "P5_hops":
        keys = ("D1", "D2", "D3", "D4")
        return float(all(max(groups[keys[index]]) < min(groups[keys[index + 1]])
                         for index in range(3)))
    return float(max(groups["near"]) < min(groups["far"]))


def per_instance(pattern, instance, draws, distance):
    """The pattern's psi on the real instance and on its matched groups, both per instance.

    Both sides use one comparison rule: a group is compared against every other group individually
    and the results are averaged.  For K5 that makes the null a symmetric comparison between random
    groups -- 0.5 by construction -- instead of comparing each random group with the mean of the rest,
    which drifts with the spread of the draws.
    """
    nodes = tuple(instance["nodes"])
    if pattern == "K5":
        inside_real = float(np.mean([distance(nodes[a], nodes[b])
                                     for a, b in combinations(range(5), 2)]))
        inside_null = [float(np.mean([distance(drawn[a], drawn[b])
                                      for a, b in combinations(range(5), 2)])) for drawn in draws]
        if not inside_null:
            return None
        real = float(np.mean([psi(inside_real, value) for value in inside_null]))
        null_scores = [psi(inside_null[i], inside_null[j])
                       for i in range(len(inside_null)) for j in range(len(inside_null)) if i != j]
        if not null_scores:
            return None
        return real, float(np.mean(null_scores))
    real = hypothesis(pattern, instance_distances(pattern, instance, nodes, distance))
    responses = [hypothesis(pattern, instance_distances(pattern, instance, drawn, distance))
                 for drawn in draws]
    if not responses:
        return None
    return real, float(np.mean(responses))


def lcp_distance(tokens):
    def distance(left, right):
        count = 0
        for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
            if x != y:
                break
            count += 1
        return 1.0 - count / 3.0
    return distance


def item_means(values, incidence, n):
    """Mean over the instances an item takes part in, ignoring instances whose null was unavailable."""
    items = sorted(incidence)
    if not items:
        return np.full(n, np.nan)
    item_index = np.concatenate([np.full(len(incidence[item]), item, dtype=np.int64) for item in items])
    inst_index = np.concatenate([np.asarray(incidence[item], dtype=np.int64) for item in items])
    sampled = values[inst_index]
    finite = np.isfinite(sampled)
    sums = np.bincount(item_index[finite], weights=sampled[finite], minlength=n)
    counts = np.bincount(item_index[finite], minlength=n)
    means = np.full(n, np.nan)
    present = counts > 0
    means[present] = sums[present] / counts[present]
    return means


def role_patterns(incidence, role_of, role, n):
    return [name for name in incidence
            if int(np.sum((np.bincount(np.concatenate([np.full(len(v), k, dtype=np.int64)
                                                       for k, v in incidence[name].items()])
                                       if incidence[name] else np.zeros(0, np.int64),
                                       minlength=n) > 0)[role_of == role])) >= MIN_ITEMS]


# ---------------------------------------------------------------- arms

def load_arms(limit=None):
    ledger = json.loads(LEDGER.read_text())
    arms = []
    for number in sorted(p.name for p in MIRROR.iterdir() if p.is_dir()):
        entry = ledger.get(number) or {}
        if entry.get("retired"):
            continue
        sid_path, recall_path = MIRROR / number / "item_sids.json", MIRROR / number / "recall.json"
        if not (sid_path.exists() and recall_path.exists()):
            continue
        recall = json.loads(recall_path.read_text()).get(RECALL)
        if recall is None:
            continue
        attested = (entry.get("files") or {}).get("item_sids.json")
        if attested is not None and hashlib.sha256(sid_path.read_bytes()).hexdigest() != attested:
            continue
        arms.append({"number": number, "recall": float(recall), "sid_path": sid_path})
        if limit and len(arms) >= limit:
            break
    return arms


def tokens_of(path, n):
    payload = json.loads(Path(path).read_text())
    tokens = np.full((n, 3), -1, dtype=np.int64)
    for key, row in payload.items():
        item = int(key)
        if 0 <= item < n:
            tokens[item] = [int(code) for code in row[:3]]
    return tokens


def collision_of(tokens):
    present = tokens[tokens[:, 0] >= 0]
    triples = {tuple(int(value) for value in row) for row in present}
    return 1.0 - len(triples) / max(len(present), 1)


# ---------------------------------------------------------------- commands

def lcp_arrays(tokens, left, right):
    shape = np.broadcast(left, right).shape
    total = np.zeros(shape, dtype=np.int64)
    alive = np.ones(shape, dtype=bool)
    for level in range(3):
        alive &= (tokens[left, level] == tokens[right, level]) & (tokens[left, level] >= 0)
        total += alive.astype(np.int64)
    return total


def behaviour_edges(adjacency, n):
    """Directed behaviour pairs (last history item, target), exactly as the 213 audit builds them.

    Retention is attributed to the role of the first element, so the direction matters: using sorted
    undirected pairs would hand half the edges to the wrong endpoint's role.
    """
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(),
                               frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    ordered = sorted(pairs)
    u = np.array([left for left, _ in ordered], dtype=np.int64)
    v = np.array([right for _, right in ordered], dtype=np.int64)
    return u, v


def cmd_roles(args):
    n, adjacency, _ = load_graph()
    role_of = load_roles(n, adjacency, args.overlap_mode)
    u, v = behaviour_edges(adjacency, n)
    role_of_u = role_of[u]
    arms = load_arms(args.arms)
    rows = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        present = (tokens[u, 0] >= 0) & (tokens[v, 0] >= 0)
        close = 1.0 - lcp_arrays(tokens, u, v) / 3.0
        entry = {"number": arm["number"], "recall": arm["recall"],
                 "collision": collision_of(tokens), "roles": {}}
        for role in range(6):
            mask = present & (role_of_u == role)
            if not mask.any():
                entry["roles"][str(role)] = None
                continue
            left, right = u[mask], v[mask]
            levels = {}
            for depth in (1, 2, 3):
                match = np.ones(left.shape[0], dtype=bool)
                for index in range(depth):
                    match &= (tokens[left, index] == tokens[right, index]) & (tokens[left, index] >= 0)
                levels[f"prefix@{depth}"] = float(match.mean())
            entry["roles"][str(role)] = {"edges": int(mask.sum()), **levels}
        rows.append(entry)
        print(f"  {arm['number']} recall {arm['recall']:.6f} collision {entry['collision']:.4f}",
              flush=True)

    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] if "collision" in row else row["l3_collision"]
                           for row in rows])
    result = {"n_arms": len(rows), "metric": RECALL, "roles": {}, "rows": rows}
    print("\n=== per-role prefix retention and its correlation with recall ===")
    print(f"  {'role':<7}{'prefix@1':>10}{'rho':>8}{'p':>7}{'r':>8}{'p':>7}"
          f"{'prefix@2 rho':>15}{'prefix@3 rho':>15}")
    for role in range(6):
        block = {}
        for depth in (1, 2, 3):
            values = np.array([row["roles"][str(role)][f"prefix@{depth}"]
                               if row["roles"][str(role)] else np.nan for row in rows])
            block[f"prefix@{depth}"] = correlate(values, recalls, collisions, args.permutations)
        block["prefix@1_mean"] = float(np.nanmean([row["roles"][str(role)]["prefix@1"]
                                                   for row in rows if row["roles"][str(role)]]))
        result["roles"][str(role)] = block
        first = block["prefix@1"]
        if first.get("spearman") is None:
            print(f"  Role {role + 1:<2}{block['prefix@1_mean']:>10.3f}   n={first['n']} < 5, "
                  f"no correlation")
            continue
        print(f"  Role {role + 1:<2}{block['prefix@1_mean']:>10.3f}"
              f"{first['spearman']:>+8.3f}{first['p_spearman']:>7.3f}"
              f"{first['pearson']:>+8.3f}{first['p_pearson']:>7.3f}"
              f"{block['prefix@2']['spearman']:>+15.3f}{block['prefix@3']['spearman']:>+15.3f}")
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def cmd_distance(args):
    n, adjacency, _ = load_graph()
    role_of = load_roles(n, adjacency, args.overlap_mode)
    u, v = behaviour_edges(adjacency, n)
    role_of_u = role_of[u]
    rng = np.random.default_rng(args.seed)

    popularity = load_popularity(n)
    degree_bucket = bucketise(np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64),
                              BUCKETS)
    popularity_bucket = bucketise(popularity, BUCKETS)
    cell_of = degree_bucket * BUCKETS + popularity_bucket
    cell_pool = [np.flatnonzero(cell_of == c) for c in range(BUCKETS * BUCKETS)]

    def matched(anchor, count, exclude_neighbours=True):
        """Random items from the anchor's degree x popularity cell, never a neighbour or the anchor."""
        pool = cell_pool[int(cell_of[anchor])]
        chosen = []
        if pool.size == 0:
            return chosen
        for _ in range(16):
            draw = pool[rng.integers(pool.size, size=count)]
            for cand in draw:
                candidate = int(cand)
                if candidate == anchor:
                    continue
                if exclude_neighbours and candidate in adjacency[anchor]:
                    continue
                chosen.append(candidate)
                if len(chosen) >= count:
                    return chosen
        return chosen

    partner = np.full(u.shape[0], -1, dtype=np.int64)
    for index in range(u.shape[0]):
        chosen = matched(int(u[index]), 1)
        if chosen:
            partner[index] = chosen[0]

    keep_roles = {}
    for role in range(6):
        indices = np.flatnonzero(role_of_u == role)
        if args.edges and indices.size > args.edges:
            picks = np.sort(rng.choice(indices.size, size=args.edges, replace=False))
            indices = indices[picks]
        keep_roles[role] = indices

    candidates = {}
    for role, indices in keep_roles.items():
        rows = np.full((indices.size, CANDS), -1, dtype=np.int64)
        for row, index in enumerate(indices):
            picked = matched(int(u[index]), CANDS)
            if picked:
                rows[row, :len(picked)] = picked[:CANDS]
        candidates[role] = rows

    arms = load_arms(args.arms)
    rows = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        present = tokens[u, 0] >= 0
        close = 1.0 - lcp_arrays(tokens, u, v) / 3.0
        far = 1.0 - lcp_arrays(tokens, u, partner) / 3.0
        win = np.where(close < far, 1.0, np.where(close == far, 0.5, 0.0))
        entry = {"number": arm["number"], "recall": arm["recall"],
                 "collision": collision_of(tokens), "roles": {}}
        for role, indices in keep_roles.items():
            mask = present[indices] & (partner[indices] >= 0)
            if not mask.any():
                entry["roles"][str(role)] = None
                continue
            draws = candidates[role][mask]
            anchors = u[indices][mask]
            truth = v[indices][mask]
            truth_distance = 1.0 - lcp_arrays(tokens, anchors, truth) / 3.0
            valid = draws >= 0
            safe = np.where(valid, draws, 0)
            candidate_distance = 1.0 - lcp_arrays(tokens, np.repeat(anchors[:, None], CANDS, axis=1),
                                                  safe) / 3.0
            closer = ((candidate_distance < truth_distance[:, None]) & valid).sum(axis=1)
            equal = ((candidate_distance == truth_distance[:, None]) & valid).sum(axis=1)
            counts = np.maximum(valid.sum(axis=1), 1)
            percentile = (closer + 0.5 * equal) / counts
            entry["roles"][str(role)] = {
                "edges": int(mask.sum()),
                "win_rate": float(win[indices][mask].mean()),
                "score": float(1.0 - np.mean(percentile)),
                "candidates_per_edge": float(valid.sum(axis=1).mean())}
        rows.append(entry)
        print(f"  {arm['number']} recall {arm['recall']:.6f}", flush=True)
    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] for row in rows])
    result = {"n_arms": len(rows), "candidates": CANDS, "roles": {}, "rows": rows}
    print("\n=== per-role retention score across arms, and its correlation with recall ===")
    print(f"  {'role':<7}{'score mean':>11}{'min':>7}{'max':>7}{'rho':>8}{'p':>7}{'r':>8}{'p':>7}")
    for role in range(6):
        values = np.array([row["roles"][str(role)]["score"] if row["roles"][str(role)] else np.nan
                           for row in rows])
        block = correlate(values, recalls, collisions, args.permutations)
        finite = values[np.isfinite(values)]
        block.update({"mean": float(finite.mean()) if finite.size else None,
                      "min": float(finite.min()) if finite.size else None,
                      "max": float(finite.max()) if finite.size else None})
        result["roles"][str(role)] = block
        if block.get("spearman") is None:
            continue
        print(f"  Role {role + 1:<2}{block['mean']:>11.3f}{block['min']:>7.3f}{block['max']:>7.3f}"
              f"{block['spearman']:>+8.3f}{block['p_spearman']:>7.3f}"
              f"{block['pearson']:>+8.3f}{block['p_pearson']:>7.3f}")
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def cmd_sid(args):
    n, _, _ = load_graph()
    arms = load_arms(args.arms)
    rows = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        present = tokens[tokens[:, 0] >= 0]
        codes = [len({int(row[level]) for row in present}) for level in range(3)]
        prefixes2 = len({(int(row[0]), int(row[1])) for row in present})
        triples = {(int(row[0]), int(row[1]), int(row[2])) for row in present}
        rows.append({"number": arm["number"], "recall": arm["recall"],
                     "l1_codes": codes[0], "l2_codes": codes[1], "l3_codes": codes[2],
                     "l2_prefixes": prefixes2, "l3_distinct": len(triples),
                     "collision": 1.0 - len(triples) / max(len(present), 1),
                     "gain_bits": math.log2(len(present) / max(len(triples), 1))})
        print(f"  {arm['number']} recall {arm['recall']:.6f}  l2 {prefixes2}  "
              f"l3 {len(triples)}  collision {rows[-1]['collision']:.3f}", flush=True)
    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] for row in rows])
    result = {"n_arms": len(rows), "metric": RECALL, "metrics": {}, "rows": rows}
    print("\n=== SID-level quantities against recall ===")
    for name in ("l1_codes", "l2_prefixes", "l3_distinct", "collision", "gain_bits"):
        values = np.array([row[name] for row in rows], dtype=float)
        block = correlate(values, recalls, collisions, args.permutations)
        result["metrics"][name] = block
        if block.get("spearman") is None:
            continue
        print(f"  {name:<12}{np.mean(values):>10.2f}{block['spearman']:>+9.3f}"
              f"{block['p_spearman']:>8.3f}{block['pearson']:>+9.3f}{block['p_pearson']:>8.3f}")
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def prepare_instances(args, n, adjacency):
    rng = np.random.default_rng(args.seed)
    cliques4 = enumerate_cliques(adjacency, 4)
    cliques5 = enumerate_cliques(adjacency, 5)
    print(f"cliques: {len(cliques4)} of size 4, {len(cliques5)} of size 5", flush=True)
    patterns = build_patterns(adjacency, rng, cliques4, cliques5, args.instances)
    popularity = load_popularity(n)
    degree_bucket = bucketise(np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64), BUCKETS)
    popularity_bucket = bucketise(popularity, BUCKETS)
    cell_of = degree_bucket * BUCKETS + popularity_bucket
    cell_pool = [np.flatnonzero(cell_of == cell) for cell in range(BUCKETS * BUCKETS)]
    nulls = build_nulls(patterns, cell_of, cell_pool, rng, args.null_draws)
    incidences = {}
    for name, instances in patterns.items():
        incidence = defaultdict(list)
        for index, instance in enumerate(instances):
            for node in instance["nodes"]:
                incidence[node].append(index)
        incidences[name] = dict(incidence)
        available = sum(1 for draws in nulls[name] if draws)
        print(f"  {name:<16} instances {len(instances):>5} with a null {available:>5} "
              f"distinct items {len(incidence):>5}", flush=True)
    return patterns, nulls, incidences


def instance_scores(pattern, instance, draws, distance):
    """Loose and strict scores for one instance, on the real group and on its matched draws.

    Both metrics come from the same distance evaluation, so the strict column costs no extra
    measuring:

    * loose -- ``psi`` on group means (the published metric; a partial ordering can score 2/3);
    * strict -- the full ordering must hold pair by pair (see ``strict_indicator``).  For K5, whose
      single group has no levels, the strict analogue is that its *widest* pair is still closer than
      the mean random pair distance, so nine tight pairs out of ten do not earn a pass.

    The null side is the average over the matched draws, so both gains are ``real - matched null``.
    """
    nodes = tuple(instance["nodes"])
    if pattern == "K5":
        def pairs(group):
            return [distance(group[a], group[b]) for a, b in combinations(range(5), 2)]

        inside_real = float(np.mean(pairs(nodes)))
        draw_means = [float(np.mean(pairs(draw))) for draw in draws]
        loose_real = float(np.mean([psi(inside_real, value) for value in draw_means]))
        loose_null = float(np.mean([psi(draw_means[i], draw_means[j])
                                    for i in range(len(draws)) for j in range(len(draws)) if i != j]))
        # strict K5: the *widest* pair of the clique must still be closer than a random group's mean
        # pair distance -- every one of the ten pairs has to be tight, not nine of ten.
        mu = float(np.mean(draw_means))
        strict_real = float(max(pairs(nodes)) < mu)
        strict_null = float(np.mean([
            float(max(pairs(draw)) < float(np.mean(np.delete(draw_means, index))))
            for index, draw in enumerate(draws)]))
        # K5 asserts one ordering (clique inside closer than random pairs), so its conjunction is the
        # loose comparison itself; the all-pairs form above stays as the stricter diagnostic.
        return loose_real, loose_null, loose_real, loose_null, strict_real, strict_null

    groups_real = instance_distances(pattern, instance, nodes, distance)
    loose_real = hypothesis(pattern, groups_real)
    conj_real = conj_indicator(pattern, groups_real)
    strict_real = strict_indicator(pattern, groups_real)
    loose, conj, strict = [], [], []
    for draw in draws:
        groups = instance_distances(pattern, instance, tuple(draw), distance)
        loose.append(hypothesis(pattern, groups))
        conj.append(conj_indicator(pattern, groups))
        strict.append(strict_indicator(pattern, groups))
    return (loose_real, float(np.mean(loose)), conj_real, float(np.mean(conj)),
            strict_real, float(np.mean(strict)))


def score_arm(patterns, nulls, incidences, distance, n):
    """Per-instance loose and strict scores, NaN where a null was unavailable.

    Each array keeps one slot per instance -- the same index the incidence map uses -- so an instance
    without a usable null cannot shift the scores of the instances after it.
    """
    loose_arrays, conj_arrays, strict_arrays = {}, {}, {}
    for name, instances in patterns.items():
        size = len(instances)
        real, null = np.full(size, np.nan), np.full(size, np.nan)
        conj_real, conj_null = np.full(size, np.nan), np.full(size, np.nan)
        strict_real, strict_null = np.full(size, np.nan), np.full(size, np.nan)
        for index, (instance, draws) in enumerate(zip(instances, nulls[name])):
            if not draws:
                continue
            scored = instance_scores(name, instance, draws, distance)
            (real[index], null[index]) = scored[0], scored[1]
            (conj_real[index], conj_null[index]) = scored[2], scored[3]
            (strict_real[index], strict_null[index]) = scored[4], scored[5]
        loose_arrays[name] = (real, null)
        conj_arrays[name] = (conj_real, conj_null)
        strict_arrays[name] = (strict_real, strict_null)
    return loose_arrays, conj_arrays, strict_arrays


def unusable(patterns, nulls):
    """How many instances of each pattern have no matched group at all."""
    return {name: sum(1 for draws in nulls[name] if not draws) for name in patterns}


def role_scores(arrays, incidences, role_of, role, n):
    """Role-level rate and gain: per item first, then over the role's items (NaN-aware)."""
    rates, gains = [], []
    for name, (real_values, null_values) in arrays.items():
        if real_values.size == 0:
            continue
        incidence = incidences[name]
        real_means = item_means(real_values, incidence, n)
        null_means = item_means(null_values, incidence, n)
        mask = (role_of == role) & np.isfinite(real_means)
        if int(mask.sum()) < MIN_ITEMS:
            continue
        rates.append(float(np.nanmean(real_means[mask])))
        gains.append(float(np.nanmean(real_means[mask]) - np.nanmean(null_means[mask])))
    if not rates:
        return None, None
    return float(np.mean(rates)), float(np.mean(gains))


def cmd_patterns(args):
    n, adjacency, _ = load_graph()
    role_of = load_roles(n, adjacency, args.overlap_mode)
    patterns, nulls, incidences = prepare_instances(args, n, adjacency)
    print("instances with no usable null:", unusable(patterns, nulls), flush=True)
    arms = load_arms(args.arms)
    rows = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        arrays, conj, strict = score_arm(patterns, nulls, incidences, lcp_distance(tokens), n)
        entry = {"number": arm["number"], "recall": arm["recall"],
                 "collision": collision_of(tokens), "gain": {}, "rate": {},
                 "strict_gain": {}, "strict_rate": {},
                 "conj_gain": {}, "conj_rate": {}}
        for role in range(6):
            rate, gain = role_scores(arrays, incidences, role_of, role, n)
            entry["gain"][str(role)] = gain
            entry["rate"][str(role)] = rate
            strict_rate, strict_gain = role_scores(strict, incidences, role_of, role, n)
            entry["strict_gain"][str(role)] = strict_gain
            entry["strict_rate"][str(role)] = strict_rate
            conj_rate, conj_gain = role_scores(conj, incidences, role_of, role, n)
            entry["conj_gain"][str(role)] = conj_gain
            entry["conj_rate"][str(role)] = conj_rate
        entry["patterns"] = {name: {"real": float(np.nanmean(values[0])) if values[0].size else None,
                                    "null": float(np.nanmean(values[1])) if values[1].size else None}
                             for name, values in arrays.items()}
        entry["patterns_conj"] = {
            name: {"real": float(np.nanmean(values[0])) if values[0].size else None,
                   "null": float(np.nanmean(values[1])) if values[1].size else None}
            for name, values in conj.items()}
        entry["patterns_strict"] = {
            name: {"real": float(np.nanmean(values[0])) if values[0].size else None,
                   "null": float(np.nanmean(values[1])) if values[1].size else None}
            for name, values in strict.items()}
        rows.append(entry)
        print(f"  {arm['number']} recall {arm['recall']:.6f}  " +
              " ".join(f"r{role}:{entry['gain'][str(role)]:+.3f}"
                       if entry["gain"][str(role)] is not None else f"r{role}:n/a"
                       for role in range(6)), flush=True)

    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] if "collision" in row else row["l3_collision"]
                           for row in rows])
    result = {"n_arms": len(rows), "metric": RECALL, "roles": {}, "per_pattern": {}, "rows": rows}
    print("\n=== per-role rate and gain across arms, and the correlation with recall ===")
    print(f"  {'role':<7}{'rate':>8}{'gain':>8}{'rho(g)':>9}{'r(g)':>8}"
          f" | {'conj':>7}{'cgain':>7}{'rho(cg)':>9}{'r(cg)':>8}"
          f" | {'pair':>6}{'pgain':>7}{'rho(pg)':>9}{'r(pg)':>8}")
    for role in range(6):
        rate = np.array([row["rate"][str(role)] if row["rate"][str(role)] is not None else np.nan
                         for row in rows])
        gain = np.array([row["gain"][str(role)] if row["gain"][str(role)] is not None else np.nan
                         for row in rows])
        strict_rate = np.array([row["strict_rate"][str(role)]
                                if row["strict_rate"][str(role)] is not None else np.nan
                                for row in rows])
        strict_gain = np.array([row["strict_gain"][str(role)]
                                if row["strict_gain"][str(role)] is not None else np.nan
                                for row in rows])
        conj_rate = np.array([row["conj_rate"][str(role)]
                              if row["conj_rate"][str(role)] is not None else np.nan for row in rows])
        conj_gain = np.array([row["conj_gain"][str(role)]
                              if row["conj_gain"][str(role)] is not None else np.nan for row in rows])
        result["roles"][str(role)] = {"rate": correlate(rate, recalls, collisions, args.permutations),
                                      "conj_rate": correlate(conj_rate, recalls, collisions,
                                                             args.permutations),
                                      "conj_gain": correlate(conj_gain, recalls, collisions,
                                                             args.permutations),
                                      "conj_rate_mean": (float(np.nanmean(conj_rate))
                                                          if np.isfinite(conj_rate).any() else None),
                                      "conj_gain_mean": (float(np.nanmean(conj_gain))
                                                          if np.isfinite(conj_gain).any() else None),
                                      "gain": correlate(gain, recalls, collisions, args.permutations),
                                      "strict_rate": correlate(strict_rate, recalls, collisions,
                                                               args.permutations),
                                      "strict_gain": correlate(strict_gain, recalls, collisions,
                                                               args.permutations),
                                      "rate_mean": float(np.nanmean(rate)) if np.isfinite(rate).any() else None,
                                      "gain_mean": float(np.nanmean(gain)) if np.isfinite(gain).any() else None,
                                      "strict_rate_mean": (float(np.nanmean(strict_rate))
                                                           if np.isfinite(strict_rate).any() else None),
                                      "strict_gain_mean": (float(np.nanmean(strict_gain))
                                                           if np.isfinite(strict_gain).any() else None)}
        block = result["roles"][str(role)]
        if block["gain"].get("spearman") is None or block["strict_gain"].get("spearman") is None:
            print(f"  Role {role + 1:<2}  n={block['gain']['n']} arms with a gain, too few to correlate")
            continue
        print(f"  Role {role + 1:<2}{block['rate_mean']:>8.3f}{block['gain_mean']:>+8.3f}"
              f"{block['gain']['spearman']:>+9.3f}{block['gain']['pearson']:>+8.3f}"
              f" | {block['conj_rate_mean']:>7.3f}{block['conj_gain_mean']:>+7.3f}"
              f"{block['conj_gain']['spearman']:>+9.3f}{block['conj_gain']['pearson']:>+8.3f}"
              f" | {block['strict_rate_mean']:>6.3f}{block['strict_gain_mean']:>+7.3f}"
              f"{block['strict_gain']['spearman']:>+9.3f}{block['strict_gain']['pearson']:>+8.3f}")
    if args.representations:
        import torch  # only needed to read the frozen codebooks

        state = torch.load(REPO / "analysis_sid_recall" / "r2" / "24" / "rqvae_best.pth",
                           map_location="cpu")["state_dict"]
        books = [state[f"rq.vq_layers.{level}.embed.weight"].numpy().astype(np.float64)
                 for level in range(3)]
        z = np.load(VECTORS / "TIGER_z.npy").astype(np.float64)
        q = np.load(VECTORS / "TIGER_q.npy").astype(np.float64)
        tokens_tiger = np.load(VECTORS / "TIGER_tokens.npy").astype(np.int64)
        distances = {
            "z": lambda left, right: float(np.linalg.norm(z[left] - z[right])),
            "q": lambda left, right: float(np.linalg.norm(q[left] - q[right])),
            "SID": lcp_distance(tokens_tiger)}
        result["representations"] = {}
        print("\n=== per-role rate and gain in z / q / SID (TIGER, the only arm with vectors) ===")
        print(f"  {'role':<7}{'G_z':>8}{'G_q':>8}{'G_SID':>8}{'d_quant':>9}{'d_SID':>8}")
        for role in range(6):
            cell = {}
            for label, distance in distances.items():
                arrays, _, _ = score_arm(patterns, nulls, incidences, distance, n)
                rate, gain = role_scores(arrays, incidences, role_of, role, n)
                cell[label] = {"rate": rate, "gain": gain}
            result["representations"][str(role)] = cell
            if cell["z"]["gain"] is not None and cell["SID"]["gain"] is not None:
                print(f"  Role {role + 1:<2}{cell['z']['gain']:>+8.3f}{cell['q']['gain']:>+8.3f}"
                      f"{cell['SID']['gain']:>+8.3f}{cell['q']['gain'] - cell['z']['gain']:>+9.3f}"
                      f"{cell['SID']['gain'] - cell['z']['gain']:>+8.3f}")

    for name in patterns:
        real = np.array([row["patterns"][name]["real"] if row["patterns"][name]["real"] is not None
                         else np.nan for row in rows])
        null = np.array([row["patterns"][name]["null"] if row["patterns"][name]["null"] is not None
                         else np.nan for row in rows])
        result["per_pattern"][name] = {
            "real_mean": float(np.nanmean(real)), "null_mean": float(np.nanmean(null)),
            "gain": correlate(real - null, recalls, collisions, args.permutations)}
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def cmd_independence(args):
    in_path = Path(args.in_path) if args.in_path else None
    if in_path is None:
        for candidate in (RESULTS / "patterns.json", RESULTS / "role_pattern_gain.json"):
            if candidate.exists():
                in_path = candidate
                break
    if in_path is None:
        raise SystemExit("independence needs per-arm role rates: run `patterns` first, or pass --in")
    payload = json.loads(in_path.read_text())
    rows = payload["rows"]
    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] if "collision" in row else row["l3_collision"]
                           for row in rows])
    roles = [str(role) for role in range(6)]
    key = args.measure
    if key not in rows[0]:
        raise SystemExit(f"--measure {key} is not in {args.in_path}; the file has "
                         f"{[name for name in rows[0] if name in ('rate', 'gain', 'strict_rate', 'strict_gain')]}")
    print(f"independence is measured on `{key}` (use --measure rate|gain to switch)", flush=True)
    rates = np.array([[row[key].get(role) if row[key].get(role) is not None else np.nan
                       for role in roles] for row in rows])
    if np.isnan(rates).any():
        means = np.nanmean(rates, axis=0)
        rates = np.where(np.isnan(rates), means, rates)
    corr = np.corrcoef(rates, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    off = [abs(corr[i, j]) for i in range(6) for j in range(i + 1, 6)]
    print(f"collinearity: mean |rho| {np.mean(off):.3f}, max {max(off):.3f}, "
          f"first component {eigenvalues[-1] / eigenvalues.sum():.1%}")
    features = rates
    plan = fold_plan(len(recalls), args.repeats, 5, args.seed)
    full = cross_validate(features, recalls, plan, args.seed)
    print(f"all six roles: CV R^2 {full['r2_mean']:+.4f} (sd {full['r2_sd']:.4f})")
    result = {"n_arms": len(rows), "collinearity": {"mean": float(np.mean(off)),
                                                    "max": float(max(off)),
                                                    "first_component": float(eigenvalues[-1] / eigenvalues.sum())},
              "full": full, "marginal": {}}
    print(f"  {'removed':<9}{'without':>10}{'drop':>9}{'p':>8}")
    for position, role in enumerate(roles):
        keep = [index for index in range(6) if index != position]
        without = cross_validate(features[:, keep], recalls, plan, args.seed)
        drop = full["r2_mean"] - without["r2_mean"]
        shuffle_rng = np.random.default_rng(args.seed + 101 * (position + 1))
        null_drops = []
        for _ in range(args.permutations):
            shuffled = features.copy()
            shuffled[:, position] = shuffle_rng.permutation(features[:, position])
            null_drops.append(cross_validate(shuffled, recalls, plan, args.seed)["r2_mean"]
                              - without["r2_mean"])
        p_value = float((np.sum(np.array(null_drops) >= drop) + 1) / (args.permutations + 1))
        result["marginal"][f"Role {int(role) + 1}"] = {"without": without["r2_mean"],
                                                       "drop": drop, "p": p_value}
        print(f"  Role {int(role) + 1:<4}{without['r2_mean']:>10.4f}{drop:>+9.4f}{p_value:>8.3f}")
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def build_features(n, adjacency, orbit_path=None, overlap_mode="mean"):
    """The ORCA feature block: four invariants plus log1p orbit counts, standardised.

    ``overlap`` is the mean Jaccard overlap between pairs of a node's neighbours.  It is undefined when
    a node has fewer than two neighbours -- there is no pair to compare -- so those nodes are imputed at
    the mean of the defined values (``overlap_mode="mean"``), which standardises to zero.  The old
    definition gave them 1.0, the maximum, which made a one-neighbour item look maximally closed; the
    ``"old"`` mode exists only to reproduce the published tables.
    """
    import itertools

    import networkx as nx

    orbit_path = Path(orbit_path) if orbit_path else ORBITS
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(n):
        for other in adjacency[node]:
            if node < other:
                graph.add_edge(node, other)
    degrees = {node: len(adjacency[node]) for node in range(n)}
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)
    mean_neighbour_degree = {node: float(np.mean([degrees[other] for other in adjacency[node]]))
                             if adjacency[node] else 0.0 for node in range(n)}
    defined, undefined = {}, []
    for node in range(n):
        near = list(adjacency[node])
        if len(near) < 2:
            undefined.append(node)
            continue
        total = 0.0
        for first, second in itertools.combinations(near, 2):
            union = len(adjacency[first] | adjacency[second])
            total += len(adjacency[first] & adjacency[second]) / union if union else 0.0
        defined[node] = total / (len(near) * (len(near) - 1) / 2)
    imputed = float(np.mean(list(defined.values()))) if defined else 0.0
    overlap = {node: defined.get(node, 1.0 if overlap_mode == "old" else imputed)
               for node in range(n)}
    orbits = np.loadtxt(orbit_path, dtype=np.float64)
    rows = []
    for node in range(n):
        rows.append([np.log1p(degrees[node]), clustering.get(node, 0.0), core.get(node, 0),
                     np.log1p(mean_neighbour_degree[node]), overlap[node],
                     *[np.log1p(value) for value in orbits[node]]])
    matrix = np.asarray(rows, dtype=np.float64)
    matrix = (matrix - matrix.mean(0)) / matrix.std(0).clip(min=1e-9)
    info = {"orbit_width": int(orbits.shape[1]), "overlap_mode": overlap_mode,
            "overlap_undefined": len(undefined), "overlap_imputed_value": imputed,
            "features": 5 + int(orbits.shape[1])}
    return matrix, info


def fit_roles(matrix, k=ROLE_K, seed=SEED, scaling="clip", normalise_components=False):
    """NMF over the non-negative part of the standardised features, then one role per item.

    ``scaling="clip"`` drops the below-average part (RolX's convention, what the published tables used);
    ``"shift"`` keeps the whole ordering by shifting the block to be non-negative.  Components differ in
    scale, so ``normalise_components`` divides each weight column by its norm before the argmax -- the
    assignment then cannot be driven by a component's magnitude alone.
    """
    from sklearn.decomposition import NMF

    block = matrix.clip(min=0.0) if scaling == "clip" else matrix - matrix.min()
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=800)
    weights = model.fit_transform(block)
    if normalise_components:
        weights = weights / np.linalg.norm(weights, axis=0, keepdims=True).clip(min=1e-12)
    return np.argmax(weights, axis=1)


def adjusted_rand(first, second):
    """Adjusted Rand index between two label vectors, the usual way to say how much a partition moved."""
    from math import comb

    first = np.asarray(first)
    second = np.asarray(second)
    table = {}
    for a, b in zip(first, second):
        table[(int(a), int(b))] = table.get((int(a), int(b)), 0) + 1
    index = sum(comb(value, 2) for value in table.values())
    rows = sum(comb(int(count), 2) for count in np.bincount(first - first.min()))
    columns = sum(comb(int(count), 2) for count in np.bincount(second - second.min()))
    total = comb(len(first), 2)
    expected = rows * columns / total if total else 0.0
    maximum = 0.5 * (rows + columns)
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.0


def feature_fingerprint(adjacency, orbit_path, overlap_mode):
    """A digest of everything that determines the roles, so a stale cache cannot be reused silently."""
    import hashlib

    digest = hashlib.sha256()
    for node in range(len(adjacency)):
        digest.update(b";".join(str(other).encode() for other in sorted(adjacency[node])))
        digest.update(b"|")
    digest.update(Path(orbit_path).read_bytes())
    digest.update(overlap_mode.encode())
    return digest.hexdigest()


def cmd_split(args):
    """Integrity check: a fresh K=6 fit must reproduce the cached partition exactly.

    With the role system fixed at K=6 there is no k to compare against, so this now verifies the one
    thing that matters -- that the cache on disk is a K=6 fit of this graph, and that `args.role` is
    one of its six roles.
    """
    n, adjacency, _ = load_graph()
    cached = load_roles(n, adjacency, args.overlap_mode)
    nodes = np.flatnonzero(cached >= 0)
    matrix, _ = cached_features(adjacency, overlap_mode=args.overlap_mode)
    fresh = fit_roles(matrix[nodes], ROLE_K, args.seed)
    identical = bool(np.array_equal(fresh, cached[nodes]))
    sizes = np.bincount(cached[nodes], minlength=ROLE_K).tolist()
    result = {"k": ROLE_K, "cached_role": args.role + 1,
              "size": int((cached == args.role).sum()), "role_sizes": sizes,
              "fresh_fit_matches_cache": identical}
    print(f"K={ROLE_K}: a fresh fit {'reproduces' if identical else 'DIFFERS FROM'} the cache "
          f"(roles {sizes})")
    path = output_path(args, args.command)
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def cached_features(adjacency, orbit_path=None, overlap_mode="mean"):
    """The ORCA feature block, rebuilt only when the graph or the settings change.

    The build costs about two and a half minutes and every role-bearing command needs it, so it is
    persisted next to the role cache under the same fingerprint discipline.
    """
    orbit_path = orbit_path or ORBITS
    fingerprint = feature_fingerprint(adjacency, orbit_path, overlap_mode)[:16]
    n = len(adjacency)
    if FEATURE_BLOCK.exists():
        cached = np.load(FEATURE_BLOCK)
        if "fingerprint" in cached and str(cached["fingerprint"]) == fingerprint:
            print(f"features: {FEATURE_BLOCK.name} reused (fingerprint {fingerprint})", flush=True)
            return cached["matrix"], json.loads(str(cached["info"]))
    matrix, info = build_features(n, adjacency, orbit_path, overlap_mode)
    np.savez(FEATURE_BLOCK, matrix=matrix, fingerprint=fingerprint, info=json.dumps(info))
    return matrix, info


def cmd_cache(args):
    """Rebuild the frozen role cache: ORCA features over the edge-bearing items, NMF, argmax."""
    n, adjacency, _ = load_graph()
    nodes = np.array([node for node in range(n) if adjacency[node]], dtype=np.int64)
    sub = [set() for _ in range(n)]
    for node in nodes:
        sub[int(node)] = adjacency[int(node)]
    matrix, info = cached_features(sub, overlap_mode=args.overlap_mode)
    matrix = matrix[nodes]
    print(f"features: {info}")
    if args.k_n != ROLE_K:
        raise SystemExit(f"the role system is K={ROLE_K} only; --k-n {args.k_n} is not supported")
    role = fit_roles(matrix, ROLE_K, args.seed, args.nmf_scaling, args.normalise)
    path = Path(args.out) if args.out else ROLE_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, nodes=nodes, role=role.astype(np.int64),
             fingerprint=feature_fingerprint([set(adjacency[i]) for i in range(n)], ORBITS,
                                             args.overlap_mode)[:16],
             overlap_mode=args.overlap_mode, scaling=args.nmf_scaling,
             normalise=int(args.normalise), seed=args.seed)
    print(f"wrote {len(nodes)} role assignments to {path}")
    print("role sizes:", dict(zip(*np.unique(role, return_counts=True))))
    return 0


def cmd_profile(args):
    import itertools

    import networkx as nx

    n, adjacency, edges = load_graph()
    role_of = load_roles(n, adjacency, args.overlap_mode)
    members = {role: int((role_of == role).sum()) for role in range(6)}
    print("role sizes:", {f"Role {role + 1}": size for role, size in members.items()})
    if args.skip_cliques:
        # walking every 4- and 5-clique of a 315k-edge graph costs minutes; the rest of the profile does
        # not need it, and the two counts are reported as unavailable rather than as zero.
        k4 = np.full(n, np.nan)
        k5 = np.full(n, np.nan)
        cliques4 = cliques5 = []
    else:
        cliques4 = enumerate_cliques(adjacency, 4)
        cliques5 = enumerate_cliques(adjacency, 5)
        k4 = np.zeros(n, dtype=np.int64)
        k5 = np.zeros(n, dtype=np.int64)
    for clique in cliques4:
        for node in clique:
            k4[node] += 1
    for clique in cliques5:
        for node in clique:
            k5[node] += 1
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    degree = np.array([len(w) for w in neighbours], dtype=np.float64)
    triangles = np.zeros(n)
    overlap = np.zeros(n)
    two_hop = np.zeros(n)
    neighbour_degree = np.zeros(n)
    for node in range(n):
        near = neighbours[node]
        if not near:
            continue
        common_total = 0
        reach = set()
        for other in near:
            common_total += len(adjacency[node] & adjacency[other])
            reach.update(neighbours[other])
        triangles[node] = common_total / 2.0
        overlap[node] = float(np.mean([len(adjacency[node] & adjacency[other]) for other in near]))
        reach.discard(node)
        two_hop[node] = float(len(reach))
        neighbour_degree[node] = float(np.mean([len(adjacency[other]) for other in near]))
    graph = nx.Graph()
    graph.add_edges_from(edges.tolist())
    core_number = nx.core_number(graph)
    k_core = np.array([core_number.get(node, 0) for node in range(n)], dtype=np.float64)
    clustering = nx.clustering(graph)
    local_clustering = np.array([clustering.get(node, 0.0) for node in range(n)])
    features = {"degree": degree, "triangles": triangles, "k4": k4.astype(np.float64),
                "k5": k5.astype(np.float64), "clustering": local_clustering, "k_core": k_core,
                "two_hop": two_hop, "neighbour_overlap": overlap, "neighbour_degree": neighbour_degree}

    def cohens_d(values, mask):
        inside, outside = values[mask], values[~mask]
        pooled = np.sqrt((inside.var(ddof=1) + outside.var(ddof=1)) / 2.0)
        return float((inside.mean() - outside.mean()) / pooled) if pooled else float("nan")

    target = role_of == args.role
    print(f"\n=== graph profile per role (mean); d = Role {args.role + 1} against the other five ===")
    print(f"  {'feature':<20}" + "".join(f"{f'R{role + 1}':>9}" for role in range(6)) + "   d")
    table = {}
    for name, values in features.items():
        row = [float(values[role_of == role].mean()) if (role_of == role).any() else float("nan")
               for role in range(6)]
        effect = cohens_d(values, target)
        table[name] = {"means": row, "cohens_d": effect}
        print(f"  {name:<20}" + "".join(f"{value:>9.2f}" for value in row) + f"{effect:>10.2f}")
    orbits = np.loadtxt(ORBITS, dtype=np.float64)
    orbit_z = (orbits - orbits.mean(0)) / (orbits.std(0) + 1e-12)
    ranked = sorted(((-abs(cohens_d(orbit_z[:, index], target)), index,
                     float(orbit_z[target, index].mean())) for index in range(orbit_z.shape[1])),
                    key=lambda row: row[0])
    print(f"\n=== ORCA orbits that separate Role {args.role + 1} most ===")
    print(f"  {'orbit':<10}{'mean z':>10}{'d':>8}")
    top = []
    for neg_abs, index, mean_z in ranked[:10]:
        top.append({"orbit": index, "mean_z": mean_z, "cohens_d": -neg_abs})
        print(f"  o5_{index:<7}{mean_z:>10.2f}{-neg_abs:>+8.2f}")
    patterns, nulls, incidences = prepare_instances(args, n, adjacency)
    print(f"\n=== pattern participation: share of Role {args.role + 1}'s items in the sampled instances ===")
    participation = {}
    for role in range(6):
        of_role = np.flatnonzero(role_of == role)
        participation[f"role{role + 1}"] = {
            name: (sum(1 for node in of_role if node in incidence) / max(of_role.size, 1))
            for name, incidence in incidences.items()}
    for name, incidence in incidences.items():
        share = participation[f"role{args.role + 1}"][name]
        print(f"  {name:<16}{share:.3f}")
    path = output_path(args, args.command)
    path.write_text(json.dumps({"profile": table, "orbits": top,
                                "participation": participation, "sizes": members},
                               indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("roles", "distance", "patterns", "independence", "profile", "split", "sid",
                 "cache"):
        block = sub.add_parser(name)
        block.add_argument("--arms", type=int, default=None)
        block.add_argument("--instances", type=int, default=1000)
        block.add_argument("--null-draws", type=int, default=5)
        block.add_argument("--permutations", type=int, default=2000)
        block.add_argument("--repeats", type=int, default=5)
        block.add_argument("--edges", type=int, default=None)
        block.add_argument("--role", type=int, default=2)
        block.add_argument("--seed", type=int, default=SEED)
        block.add_argument("--in", dest="in_path", type=str, default=None)
        block.add_argument("--representations", action="store_true")
        block.add_argument("--measure", default="gain",
                           choices=("gain", "rate", "conj_gain", "conj_rate",
                                    "strict_gain", "strict_rate"))
        block.add_argument("--skip-cliques", dest="skip_cliques", action="store_true")
        block.add_argument("--k-n", dest="k_n", type=int, default=6)
        block.add_argument("--overlap-mode", dest="overlap_mode", default="mean",
                           choices=("mean", "old"))
        block.add_argument("--nmf-scaling", dest="nmf_scaling", default="clip",
                           choices=("clip", "shift"))
        block.add_argument("--normalise", action="store_true")
        block.add_argument("--out", type=str, default=None)

    args = parser.parse_args()
    handlers = {"roles": cmd_roles, "distance": cmd_distance, "patterns": cmd_patterns,
                "independence": cmd_independence, "profile": cmd_profile,
                "split": cmd_split, "sid": cmd_sid, "cache": cmd_cache}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
