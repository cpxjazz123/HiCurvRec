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
REPO = HERE.parent
MIRROR = HERE / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
VECTORS = REPO / "results" / "research" / "issue215" / "vectors"
BEHAVIOUR = REPO / "results" / "research" / "issue215" / "graphs" / "behaviour.in"
ROLE_CACHE = HERE / "structure_roles5.npz"
ORBITS = REPO / "results" / "research" / "issue213" / "orca" / "node5.out"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
RECALL = "test_recall@10"

SEED = 42
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")
MIN_ITEMS = 30
BUCKETS = 5
CANDS = 200


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


def folds(size, count, rng):
    order = rng.permutation(size)
    return [order[index::count] for index in range(count)]


def inner_penalty(matrix, target, rng, lambdas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    best, best_error = lambdas[0], float("inf")
    splits = folds(len(target), 4, rng)
    for penalty in lambdas:
        errors = []
        for test in splits:
            train = np.setdiff1d(np.arange(len(target)), test)
            scaled, mean, scale = standardise(matrix[train])
            intercept, coefficients = fit_ridge(scaled, target[train], penalty)
            held = (matrix[test] - mean) / scale
            errors.append(np.mean((target[test] - predict_ridge(held, intercept, coefficients)) ** 2))
        error = float(np.mean(errors))
        if error < best_error:
            best, best_error = penalty, error
    return best


def cross_validate(matrix, target, rng, repeats):
    scores = []
    for _ in range(repeats):
        splits = folds(len(target), 5, rng)
        predicted = np.full(len(target), np.nan)
        for test in splits:
            train = np.setdiff1d(np.arange(len(target)), test)
            if matrix.shape[1] == 0:
                predicted[test] = float(target[train].mean())
                continue
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


def load_roles(n):
    role_of = np.full(n, -1, dtype=np.int64)
    cached = np.load(ROLE_CACHE)
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
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values))
    return np.minimum((rank * buckets) // max(len(values), 1), buckets - 1)


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
        extension = [w for w in neighbours[arm[0]] if w != centre and not adjacent(w, centre)]
        if not extension:
            continue
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


def instance_distances(pattern, nodes, distance):
    if pattern == "K5":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in combinations(range(5), 2)]}
    if pattern == "K4_tail":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in combinations(range(4), 2)],
                "far": [distance(nodes[4], nodes[a]) for a in range(3)]}
    if pattern == "double_triangle":
        return {"near": [distance(nodes[a], nodes[b])
                         for a, b in [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]],
                "far": [distance(nodes[a], nodes[b]) for a, b in [(1, 3), (1, 4), (2, 3), (2, 4)]]}
    if pattern == "K2_3":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in
                         [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]],
                "far": [distance(nodes[a], nodes[b]) for a, b in
                        [(0, 1), (2, 3), (2, 4), (3, 4)]]}
    if pattern == "P5_centre":
        return {"near": [distance(nodes[2], nodes[1]), distance(nodes[2], nodes[3])],
                "far": [distance(nodes[2], nodes[0]), distance(nodes[2], nodes[4])]}
    if pattern == "P5_hops":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]],
                "far": [distance(nodes[a], nodes[b]) for a, b in [(0, 2), (1, 3), (2, 4)]],
                "D1": [distance(nodes[a], nodes[b]) for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]],
                "D2": [distance(nodes[a], nodes[b]) for a, b in [(0, 2), (1, 3), (2, 4)]],
                "D3": [distance(nodes[a], nodes[b]) for a, b in [(0, 3), (1, 4)]],
                "D4": [distance(nodes[0], nodes[4])]}
    if pattern == "C5":
        return {"near": [distance(nodes[a], nodes[b])
                         for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]],
                "far": [distance(nodes[a], nodes[b])
                        for a, b in [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]]}
    if pattern == "K1_4":
        return {"near": [distance(nodes[0], nodes[i]) for i in range(1, 5)],
                "far": [distance(nodes[i], nodes[j]) for i in range(1, 5) for j in range(i + 1, 5)]}
    if pattern == "T5":
        return {"near": [distance(nodes[0], nodes[i]) for i in (1, 2, 3)],
                "far": [distance(nodes[0], nodes[4])]}
    raise KeyError(pattern)


def per_instance(pattern, instance, draws, distance):
    nodes = tuple(instance["nodes"])
    if pattern == "K5":
        inside_real = float(np.mean([distance(nodes[a], nodes[b])
                                     for a, b in combinations(range(5), 2)]))
        inside_null = [float(np.mean([distance(drawn[a], drawn[b])
                                      for a, b in combinations(range(5), 2)])) for drawn in draws]
        if not inside_null:
            return None
        real = float(np.mean([psi(inside_real, value) for value in inside_null]))
        substituted = [psi(value, float(np.mean([item for index, item in enumerate(inside_null)
                                                 if index != position])))
                       for position, value in enumerate(inside_null)
                       if len(inside_null) > 1]
        if not substituted:
            return None
        return real, float(np.mean(substituted))
    real = hypothesis(pattern, instance_distances(pattern, nodes, distance))
    responses = [hypothesis(pattern, instance_distances(pattern, drawn, distance))
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
    items = sorted(incidence)
    if not items:
        return np.full(n, np.nan)
    item_index = np.concatenate([np.full(len(incidence[item]), item, dtype=np.int64) for item in items])
    inst_index = np.concatenate([np.asarray(incidence[item], dtype=np.int64) for item in items])
    counts = np.bincount(item_index, minlength=n)
    sums = np.bincount(item_index, weights=values[inst_index], minlength=n)
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
    role_of = load_roles(n)
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
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def cmd_distance(args):
    n, adjacency, _ = load_graph()
    role_of = load_roles(n)
    u, v = behaviour_edges(adjacency, n)
    role_of_u = role_of[u]
    rng = np.random.default_rng(args.seed)
    partner = np.full(u.shape[0], -1, dtype=np.int64)
    for index in range(u.shape[0]):
        anchor = int(u[index])
        for _ in range(32):
            candidate = int(rng.integers(n))
            if candidate != anchor and candidate not in adjacency[anchor]:
                partner[index] = candidate
                break
    keep_roles = {}
    for role in range(6):
        indices = np.flatnonzero(role_of_u == role)
        if args.edges and indices.size > args.edges:
            picks = np.sort(rng.choice(indices.size, size=args.edges, replace=False))
            indices = indices[picks]
        keep_roles[role] = indices
    candidates = {role: rng.integers(0, n, size=(indices.size, CANDS))
                  for role, indices in keep_roles.items()}
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
            candidate_distance = 1.0 - lcp_arrays(tokens, np.repeat(anchors[:, None], CANDS, axis=1),
                                                  draws) / 3.0
            closer = (candidate_distance < truth_distance[:, None]).sum(axis=1)
            equal = (candidate_distance == truth_distance[:, None]).sum(axis=1)
            entry["roles"][str(role)] = {
                "edges": int(mask.sum()),
                "win_rate": float(win[indices][mask].mean()),
                "score": float(1.0 - np.mean((closer + 0.5 * equal) / CANDS))}
        rows.append(entry)
        print(f"  {arm['number']} recall {arm['recall']:.6f}", flush=True)
    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] if "collision" in row else row["l3_collision"]
                           for row in rows])
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
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
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
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
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
        print(f"  {name:<16} instances {len(instances):>5} distinct items {len(incidence):>5}", flush=True)
    return patterns, nulls, incidences


def score_arm(patterns, nulls, incidences, distance, n):
    arrays = {}
    for name in patterns:
        real_values, null_values = [], []
        for instance, draws in zip(patterns[name], nulls[name]):
            if not draws:
                continue
            computed = per_instance(name, instance, draws, distance)
            if computed is None:
                continue
            real_values.append(computed[0])
            null_values.append(computed[1])
        arrays[name] = (np.asarray(real_values), np.asarray(null_values))
    return arrays


def role_scores(arrays, incidences, role_of, role, n):
    rates, gains = [], []
    for name, (real_values, null_values) in arrays.items():
        if real_values.size == 0:
            continue
        incidence = incidences[name]
        items = [item for item in incidence if role_of[item] == role]
        if len(items) < MIN_ITEMS:
            continue
        real_means = item_means(real_values, incidence, n)
        null_means = item_means(null_values, incidence, n)
        mask = np.zeros(n, dtype=bool)
        mask[items] = True
        mask &= np.isfinite(real_means)
        if not mask.any():
            continue
        rates.append(float(np.nanmean(real_means[mask])))
        gains.append(float(np.nanmean(real_means[mask]) - np.nanmean(null_means[mask])))
    if not rates:
        return None, None
    return float(np.mean(rates)), float(np.mean(gains))


def cmd_patterns(args):
    n, adjacency, _ = load_graph()
    role_of = load_roles(n)
    patterns, nulls, incidences = prepare_instances(args, n, adjacency)
    arms = load_arms(args.arms)
    rows = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        arrays = score_arm(patterns, nulls, incidences, lcp_distance(tokens), n)
        entry = {"number": arm["number"], "recall": arm["recall"],
                 "collision": collision_of(tokens), "gain": {}, "rate": {}}
        for role in range(6):
            rate, gain = role_scores(arrays, incidences, role_of, role, n)
            entry["gain"][str(role)] = gain
            entry["rate"][str(role)] = rate
        entry["patterns"] = {name: {"real": float(np.mean(values[0])) if values[0].size else None,
                                    "null": float(np.mean(values[1])) if values[1].size else None}
                             for name, values in arrays.items()}
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
    print(f"  {'role':<7}{'rate mean':>10}{'gain mean':>10}{'rho(rate)':>11}{'rho(gain)':>11}"
          f"{'r(gain)':>9}{'p':>7}")
    for role in range(6):
        rate = np.array([row["rate"][str(role)] if row["rate"][str(role)] is not None else np.nan
                         for row in rows])
        gain = np.array([row["gain"][str(role)] if row["gain"][str(role)] is not None else np.nan
                         for row in rows])
        result["roles"][str(role)] = {"rate": correlate(rate, recalls, collisions, args.permutations),
                                      "gain": correlate(gain, recalls, collisions, args.permutations),
                                      "rate_mean": float(np.nanmean(rate)) if np.isfinite(rate).any() else None,
                                      "gain_mean": float(np.nanmean(gain)) if np.isfinite(gain).any() else None}
        block = result["roles"][str(role)]
        if block["rate"].get("spearman") is None or block["gain"].get("spearman") is None:
            print(f"  Role {role + 1:<2}  n={block['rate']['n']} arms with a rate, too few to correlate")
            continue
        print(f"  Role {role + 1:<2}{block['rate_mean']:>10.3f}{block['gain_mean']:>10.3f}"
              f"{block['rate']['spearman']:>+11.3f}{block['gain']['spearman']:>+11.3f}"
              f"{block['gain']['pearson']:>+9.3f}{block['gain']['p_pearson']:>7.3f}")
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
                arrays = score_arm(patterns, nulls, incidences, distance, n)
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
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def cmd_independence(args):
    if not args.in_path:
        raise SystemExit("independence reads the per-arm role rates: pass --in <patterns.json>")
    payload = json.loads(Path(args.in_path).read_text())
    rows = payload["rows"]
    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["collision"] if "collision" in row else row["l3_collision"]
                           for row in rows])
    roles = [str(role) for role in range(6)]
    key = "rate" if "rate" in rows[0] else "role_rate"
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
    rng = np.random.default_rng(args.seed)
    full = cross_validate(features, recalls, rng, args.repeats)
    print(f"all six roles: CV R^2 {full['r2_mean']:+.4f} (sd {full['r2_sd']:.4f})")
    result = {"n_arms": len(rows), "collinearity": {"mean": float(np.mean(off)),
                                                    "max": float(max(off)),
                                                    "first_component": float(eigenvalues[-1] / eigenvalues.sum())},
              "full": full, "marginal": {}}
    print(f"  {'removed':<9}{'without':>10}{'drop':>9}{'p':>8}")
    for position, role in enumerate(roles):
        keep = [index for index in range(6) if index != position]
        without = cross_validate(features[:, keep], recalls, rng, args.repeats)
        drop = full["r2_mean"] - without["r2_mean"]
        shuffle_rng = np.random.default_rng(args.seed + 101 * (position + 1))
        null_drops = []
        for _ in range(args.permutations):
            shuffled = features.copy()
            shuffled[:, position] = shuffle_rng.permutation(features[:, position])
            null_drops.append(cross_validate(shuffled, recalls, rng, args.repeats)["r2_mean"]
                              - without["r2_mean"])
        p_value = float((np.sum(np.array(null_drops) >= drop) + 1) / (args.permutations + 1))
        result["marginal"][f"Role {int(role) + 1}"] = {"without": without["r2_mean"],
                                                       "drop": drop, "p": p_value}
        print(f"  Role {int(role) + 1:<4}{without['r2_mean']:>10.4f}{drop:>+9.4f}{p_value:>8.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def build_features(n, adjacency):
    import itertools

    import networkx as nx

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
    overlap = {}
    for node in range(n):
        near = list(adjacency[node])
        if len(near) < 2:
            overlap[node] = 1.0
            continue
        total = 0.0
        for first, second in itertools.combinations(near, 2):
            union = len(adjacency[first] | adjacency[second])
            total += len(adjacency[first] & adjacency[second]) / union if union else 0.0
        overlap[node] = total / (len(near) * (len(near) - 1) / 2)
    orbits = np.loadtxt(ORBITS, dtype=np.float64)
    rows = []
    for node in range(n):
        rows.append([np.log1p(degrees[node]), clustering.get(node, 0.0), core.get(node, 0),
                     np.log1p(mean_neighbour_degree[node]), overlap[node],
                     *[np.log1p(value) for value in orbits[node]]])
    matrix = np.asarray(rows, dtype=np.float64)
    return (matrix - matrix.mean(0)) / matrix.std(0).clip(min=1e-9)


def fit_roles(matrix, k, seed):
    from sklearn.decomposition import NMF

    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=800)
    return np.argmax(model.fit_transform(matrix.clip(min=0.0)), axis=1)


def cmd_k(args):
    ks = [int(value) for value in args.k.split(",")]
    n, adjacency, _ = load_graph()
    print("building the ORCA feature block ...", flush=True)
    matrix = build_features(n, adjacency)
    assignments = {k: fit_roles(matrix, k, args.seed) for k in ks}
    cached = load_roles(n)
    assignments["cached"] = cached
    patterns, nulls, incidences = prepare_instances(args, n, adjacency)
    arms = load_arms(args.arms)
    scored = []
    for arm in arms:
        tokens = tokens_of(arm["sid_path"], n)
        arrays = score_arm(patterns, nulls, incidences, lcp_distance(tokens), n)
        scored.append((arm, arrays))
        print(f"  scored {arm['number']}", flush=True)
    recalls = np.array([arm["recall"] for arm, _ in scored])
    result = {"ks": ks, "n_arms": len(scored), "granularities": {}}
    for key, assignment in assignments.items():
        roles = int(assignment.max()) + 1
        sizes = np.bincount(assignment[assignment >= 0], minlength=roles)
        kept = [role for role in range(roles) if sizes[role] >= MIN_ITEMS]
        table = np.zeros((len(scored), len(kept)))
        for row, (_, arrays) in enumerate(scored):
            for column, role in enumerate(kept):
                rate, _ = role_scores(arrays, incidences, assignment, role, n)
                table[row, column] = rate if rate is not None else np.nan
        rates = np.where(np.isnan(table), np.nanmean(table, axis=0), table)
        rng = np.random.default_rng(args.seed)
        full = cross_validate(rates, recalls, rng, args.repeats)
        marginals = {}
        best, best_drop = None, -np.inf
        for column, role in enumerate(kept):
            keep = [index for index in range(len(kept)) if index != column]
            without = cross_validate(rates[:, keep], recalls, rng, args.repeats)
            drop = full["r2_mean"] - without["r2_mean"]
            marginals[role + 1] = {"without": without["r2_mean"], "drop": drop}
            if drop > best_drop:
                best, best_drop = role + 1, drop
        base = np.flatnonzero(assignments["cached"] == 2)
        split = np.bincount(assignment[base][assignment[base] >= 0], minlength=roles).tolist()
        result["granularities"][str(key)] = {"sizes": sizes.tolist(), "kept": [role + 1 for role in kept],
                                             "cv": full, "marginal": marginals,
                                             "best_role": best, "best_drop": best_drop,
                                             "cached_role3_split": split}
        print(f"K={key}: kept {len(kept)} roles, CV R^2 {full['r2_mean']:+.4f}, "
              f"strongest marginal role {best} ({best_drop:+.4f}), "
              f"cached Role 3 largest share {max(split) / max(len(base), 1):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def cmd_split(args):
    ks = [int(value) for value in args.k.split(",")]
    n, adjacency, _ = load_graph()
    cached = load_roles(n)
    base = np.flatnonzero(cached == args.role)
    matrix = build_features(n, adjacency)
    result = {"cached_role": args.role + 1, "size": int(len(base)), "splits": {}}
    for k in ks:
        assignment = fit_roles(matrix, k, args.seed)
        sizes = np.bincount(assignment[base][assignment[base] >= 0], minlength=k).tolist()
        result["splits"][str(k)] = {"sizes": sizes,
                                    "shares": [value / max(len(base), 1) for value in sizes]}
        print(f"K={k}: cached Role {args.role + 1} splits as {sizes}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def cmd_profile(args):
    import itertools

    import networkx as nx

    n, adjacency, edges = load_graph()
    role_of = load_roles(n)
    members = {role: int((role_of == role).sum()) for role in range(6)}
    print("role sizes:", {f"Role {role + 1}": size for role, size in members.items()})
    cliques4 = enumerate_cliques(adjacency, 4)
    cliques5 = enumerate_cliques(adjacency, 5)
    k4 = np.zeros(n, dtype=np.int64)
    for clique in cliques4:
        for node in clique:
            k4[node] += 1
    k5 = np.zeros(n, dtype=np.int64)
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
    of_role = np.flatnonzero(role_of == args.role)
    for name, incidence in incidences.items():
        touched = sum(1 for node in of_role if node in incidence)
        participation[name] = touched / max(of_role.size, 1)
        print(f"  {name:<16}{participation[name]:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"profile": table, "orbits": top,
                                              "participation": participation, "sizes": members},
                                             indent=1, default=float))
        print(f"\nwritten {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("roles", "distance", "patterns", "independence", "k", "profile", "split", "sid"):
        block = sub.add_parser(name)
        block.add_argument("--arms", type=int, default=None)
        block.add_argument("--instances", type=int, default=1000)
        block.add_argument("--null-draws", type=int, default=5)
        block.add_argument("--permutations", type=int, default=2000)
        block.add_argument("--repeats", type=int, default=5)
        block.add_argument("--edges", type=int, default=None)
        block.add_argument("--k", type=str, default="6")
        block.add_argument("--role", type=int, default=2)
        block.add_argument("--seed", type=int, default=SEED)
        block.add_argument("--in", dest="in_path", type=str, default=None)
        block.add_argument("--representations", action="store_true")
        block.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    handlers = {"roles": cmd_roles, "distance": cmd_distance, "patterns": cmd_patterns,
                "independence": cmd_independence, "k": cmd_k, "profile": cmd_profile,
                "split": cmd_split, "sid": cmd_sid}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
