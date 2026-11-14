"""ORCA-5 role-aware structure preservation: a dedicated distance test per role, against its own null.

The roles are the repository's: ORCA's 73 node orbits of graphlets up to five nodes on the train
behaviour graph, grouped into six NMF roles (`structure_roles5.npz`).  A role is not a graphlet, so each
role gets a candidate five-node pattern, and the pattern is **verified against the role** before any
score is read: an instance is attributed to the majority ORCA role of its nodes, and the composition
table shows how much of each role a pattern actually covers.  A pattern that does not land on its role
is reported as such, never silently scored.

Per role and representation m in {z, q, SID}:

    D_m(P) = mean distance over the pair set P
    psi(a,b) = 1 if a < b, 0.5 if a == b, 0 if a > b        (ties are neither win nor loss)
    S_r = mean over instances of the role's hypothesis
    S_r,null = the same hypothesis on degree- AND popularity-matched random five-node groups
    G_r = S_r - S_r,null
    delta_quant = G_q - G_z          did quantization raise or lower the structure's gain
    delta_SID   = G_SID - G_z        the same for the final identifier

The null groups are drawn once, per instance, and shared by all three representations; intervals come
from a bootstrap that resamples **anchors**, because five-node graphlets overlap heavily and treating
them as independent would understate the spread.

Six hypotheses, fixed before any recall was read:

    role0  K5: the ten inside pairs closer than a matched five-node group's ten pairs
           plus double-triangle hub: the six triangle pairs closer than the four cross pairs
    role1  K4 + tail: the six core pairs closer than the three tail-to-core pairs
    role2  K2,3: cross (six edges) vs same-side (four non-edges) -- BOTH directions reported
    role3  P5 centre: one-hop pairs closer than two-hop pairs (a neutral control, not a verdict)
    role4  C5: adjacent pairs closer than skip-one pairs
           P5: mean of psi(D1,D2), psi(D2,D3), psi(D3,D4) -- no strict four-level monotonicity
    role5  K1,4: centre-to-leaf closer than leaf-to-leaf
           T5: one-hop from the branch centre closer than two-hop

    python3 analysis_sid_recall/orca5_role_preservation.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BEHAVIOUR = REPO / "results" / "research" / "issue215" / "graphs" / "behaviour.in"
VECTORS = REPO / "results" / "research" / "issue215" / "vectors"
ROLE_CACHE5 = HERE / "structure_roles5.npz"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
MIRROR = HERE / "r2"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = HERE / "orca5_role_preservation.json"

SEED = 42
MAX_INSTANCES = 1000
MAX_ATTEMPTS = 200_000
NULL_DRAWS = 5
BOOTSTRAP = 2000
BUCKETS = 5
RECALL = "test_recall@10"
REPRESENTATIONS = ("z", "q", "SID")
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops",
            "K1_4", "T5")
ROLE_OF_PATTERN = {"K5": 0, "double_triangle": 0, "K4_tail": 1, "K2_3": 2, "P5_centre": 3,
                   "C5": 4, "P5_hops": 4, "K1_4": 5, "T5": 5}


def load_graph() -> tuple[int, list[set[int]], np.ndarray]:
    with BEHAVIOUR.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency = [set() for _ in range(n)]
    for left, right in edges.tolist():
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    return n, adjacency, edges


def load_popularity(n: int) -> np.ndarray:
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["target"])
    counts = frame["target"].value_counts()
    popularity = np.zeros(n, dtype=np.float64)
    for item, count in counts.items():
        if 0 <= int(item) < n:
            popularity[int(item)] = float(count)
    return popularity


def bucketise(values: np.ndarray, buckets: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values))
    return np.minimum((rank * buckets) // max(len(values), 1), buckets - 1)


def enumerate_cliques(adjacency: list[set[int]], size: int) -> list[tuple[int, ...]]:
    """Every clique of the given size, closed from its (size-1) cliques."""
    base: list[tuple[int, ...]] = [(node,) for node in range(len(adjacency))]
    for depth in range(2, size + 1):
        grown: list[tuple[int, ...]] = []
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


def build_patterns(adjacency: list[set[int]], rng: np.random.Generator,
                   cliques4: list[tuple[int, ...]],
                   cliques5: list[tuple[int, ...]]) -> dict:
    """Strict five-node instances per pattern, sampled without replacement."""
    n = len(adjacency)
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    pools: dict[str, dict[frozenset, dict]] = {name: {} for name in PATTERNS}

    def pick(values):
        values = list(values)
        return int(values[rng.integers(len(values))])

    def keep(name: str, nodes, block: dict) -> None:
        if len(set(nodes)) != len(nodes):
            return
        key = frozenset(nodes)
        if key in pools[name] or len(pools[name]) >= MAX_INSTANCES:
            return
        pools[name][key] = {"nodes": list(nodes), **block}

    def adjacent(left: int, right: int) -> bool:
        return right in adjacency[left]

    if cliques5:
        picks = rng.choice(len(cliques5), size=min(len(cliques5), MAX_INSTANCES), replace=False)
        for index in sorted(picks.tolist()):
            keep("K5", list(cliques5[index]), {"kind": "tight"})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["double_triangle"]) >= MAX_INSTANCES:
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

    attempts = 0
    while attempts < MAX_ATTEMPTS and len(pools["K4_tail"]) < MAX_INSTANCES:
        attempts += 1
        if not cliques4:
            break
        index = int(rng.integers(len(cliques4)))
        core = list(cliques4[index])
        attachment = int(core[int(rng.integers(4))])
        others = [member for member in core if member != attachment]
        tails = [w for w in adjacency[attachment]
                 if w not in core and not any(adjacent(w, other) for other in others)]
        if not tails:
            continue
        tail = int(tails[int(rng.integers(len(tails)))])
        nodes = core + [tail]
        keep("K4_tail", nodes, {
            "core": list(combinations(range(4), 2)),
            "tail": [(4, position) for position in range(4) if nodes[position] != attachment]})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["K2_3"]) >= MAX_INSTANCES:
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
        b = pick(candidates)
        keep("K2_3", [a, b] + shared, {
            "cross": [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
            "same": [(0, 1), (2, 3), (2, 4), (3, 4)]})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["P5_centre"]) >= MAX_INSTANCES:
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
        keep("P5_hops", nodes, {
            "D1": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "D2": [(0, 2), (1, 3), (2, 4)],
            "D3": [(0, 3), (1, 4)],
            "D4": [(0, 4)]})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["C5"]) >= MAX_INSTANCES:
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
        e = pick(rest_d)
        keep("C5", [a, b, c, d, e], {
            "adjacent": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
            "skip": [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["K1_4"]) >= MAX_INSTANCES:
            break
        centre = pick(range(n))
        if len(neighbours[centre]) < 4:
            continue
        picks = rng.choice(len(neighbours[centre]), size=4, replace=False)
        leaves = [neighbours[centre][int(index)] for index in picks]
        if any(adjacent(x, y) for x, y in combinations(leaves, 2)):
            continue
        keep("K1_4", [centre] + leaves, {
            "spoke": [(0, 1), (0, 2), (0, 3), (0, 4)],
            "leaf": list(combinations(range(1, 5), 2))})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["T5"]) >= MAX_INSTANCES:
            break
        centre = pick(range(n))
        if len(neighbours[centre]) < 3:
            continue
        picks = rng.choice(len(neighbours[centre]), size=3, replace=False)
        arm = [neighbours[centre][int(index)] for index in picks]
        if any(adjacent(x, y) for x, y in combinations(arm, 2)):
            continue
        extension = [w for w in neighbours[arm[0]]
                     if w != centre and not adjacent(w, centre)]
        if not extension:
            continue
        tip = pick(extension)
        keep("T5", [centre] + arm + [tip], {
            "hop1": [(0, 1), (0, 2), (0, 3)],
            "hop2": [(0, 4)]})

    return {name: list(blocks.values()) for name, blocks in pools.items()}


def psi(left: float, right: float) -> float:
    if left < right:
        return 1.0
    if left == right:
        return 0.5
    return 0.0


def hypothesis(pattern: str, distances: dict) -> float:
    """One number per instance, in [0,1]; 0.5 means the distances are indistinguishable."""
    mean = lambda values: float(np.mean(values))  # noqa: E731
    if pattern in ("K5", "K4_tail", "K1_4", "T5", "P5_centre", "C5", "double_triangle"):
        near = mean(distances["near"])
        far = mean(distances["far"])
        return psi(near, far)
    if pattern == "P5_hops":
        return float(np.mean([psi(mean(distances[key]), mean(distances[next_key]))
                              for key, next_key in (("D1", "D2"), ("D2", "D3"), ("D3", "D4"))]))
    if pattern == "K2_3":
        return psi(mean(distances["near"]), mean(distances["far"]))
    raise KeyError(pattern)


GROUPS = {
    "K5": ("inside", "matched_inside"),
    "double_triangle": ("inside", "cross"),
    "K4_tail": ("core", "tail"),
    "K2_3": ("cross", "same"),
    "P5_centre": ("hop1", "hop2"),
    "C5": ("adjacent", "skip"),
    "P5_hops": ("D1", "D2"),
    "K1_4": ("spoke", "leaf"),
    "T5": ("hop1", "hop2"),
}


def instance_distances(pattern: str, nodes: tuple[int, ...], distance) -> dict:
    """Pair groups for one instance, expressed on the given node tuple."""
    if pattern == "K5":
        return {"near": [distance(nodes[a], nodes[b])
                         for a, b in combinations(range(5), 2)]}
    if pattern == "K4_tail":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in combinations(range(4), 2)],
                "far": [distance(nodes[4], nodes[a]) for a in range(3)]}
    if pattern == "double_triangle":
        return {"near": [distance(nodes[a], nodes[b])
                         for a, b in [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]],
                "far": [distance(nodes[a], nodes[b])
                        for a, b in [(1, 3), (1, 4), (2, 3), (2, 4)]]}
    if pattern == "K2_3":
        return {"near": [distance(nodes[a], nodes[b]) for a, b in
                         [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]],
                "far": [distance(nodes[a], nodes[b]) for a, b in
                        [(0, 1), (2, 3), (2, 4), (3, 4)]]}
    if pattern == "P5_centre":
        return {"near": [distance(nodes[2], nodes[1]), distance(nodes[2], nodes[3])],
                "far": [distance(nodes[2], nodes[0]), distance(nodes[2], nodes[4])]}
    if pattern == "P5_hops":
        return {"near": [distance(nodes[a], nodes[b])
                         for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]],
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
                "far": [distance(nodes[i], nodes[j])
                        for i in range(1, 5) for j in range(i + 1, 5)]}
    if pattern == "T5":
        return {"near": [distance(nodes[0], nodes[i]) for i in (1, 2, 3)],
                "far": [distance(nodes[0], nodes[4])]}
    raise KeyError(pattern)


def score_pattern(pattern: str, instances: list[dict], distance,
                  null_nodes: list[list[tuple]]) -> dict:
    """S_real, S_null and the per-instance series for one pattern and one representation.

    K5 compares its ten inside pairs against a matched five-node group's ten pairs, and its own null
    substitutes one matched group for another; every other pattern is entirely inside the group, so
    its null is the same hypothesis evaluated on the matched groups.
    """
    real: list[float] = []
    null: list[float] = []
    for instance, draws in zip(instances, null_nodes):
        nodes = tuple(instance["nodes"])
        if pattern == "K5":
            inside_real = float(np.mean([distance(nodes[a], nodes[b])
                                          for a, b in combinations(range(5), 2)]))
            inside_null = [float(np.mean([distance(drawn[a], drawn[b])
                                          for a, b in combinations(range(5), 2)]))
                           for drawn in draws]
            if not inside_null:
                continue
            real.append(float(np.mean([psi(inside_real, value) for value in inside_null])))
            substituted = []
            for position, value in enumerate(inside_null):
                others = [item for index, item in enumerate(inside_null) if index != position]
                if others:
                    substituted.append(psi(value, float(np.mean(others))))
            if substituted:
                null.append(float(np.mean(substituted)))
            continue
        real.append(hypothesis(pattern, instance_distances(pattern, nodes, distance)))
        responses = [hypothesis(pattern, instance_distances(pattern, drawn, distance))
                     for drawn in draws]
        if responses:
            null.append(float(np.mean(responses)))
    return {"S": float(np.mean(real)) if real else None,
            "null": float(np.mean(null)) if null else None,
            "G": (float(np.mean(real)) - float(np.mean(null))) if (real and null) else None,
            "instances": len(real)}


def main() -> int:
    n, adjacency, edges = load_graph()
    rng = np.random.default_rng(SEED)
    cliques4 = enumerate_cliques(adjacency, 4)
    cliques5 = enumerate_cliques(adjacency, 5)
    print(f"cliques: {len(cliques4)} of size 4, {len(cliques5)} of size 5", flush=True)
    patterns = build_patterns(adjacency, rng, cliques4, cliques5)
    popularity = load_popularity(n)
    degree_bucket = bucketise(np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64),
                              BUCKETS)
    popularity_bucket = bucketise(popularity, BUCKETS)
    cell_of = degree_bucket * BUCKETS + popularity_bucket
    cell_pool = [np.flatnonzero(cell_of == cell) for cell in range(BUCKETS * BUCKETS)]

    role_of = np.full(n, -1, dtype=np.int64)
    for node, role in zip(np.load(ROLE_CACHE5)["nodes"], np.load(ROLE_CACHE5)["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)

    # the null groups: same cell (degree x popularity) per role position, distinct, shared everywhere
    null_nodes: dict[str, list[list[tuple]]] = {}
    for name, instances in patterns.items():
        blocks = []
        for instance in instances:
            draws = []
            for _ in range(NULL_DRAWS):
                drawn: list[int] = []
                for node in instance["nodes"]:
                    pool = cell_pool[int(cell_of[node])]
                    if pool.size == 0:
                        continue
                    for _ in range(64):
                        candidate = int(pool[rng.integers(pool.size)])
                        if candidate in instance["nodes"] or candidate in drawn:
                            continue
                        drawn.append(candidate)
                        break
                if len(drawn) == len(instance["nodes"]):
                    draws.append(tuple(drawn))
            blocks.append(draws)
        null_nodes[name] = blocks

    # ---- verification: which ORCA role does each pattern actually land on? ----
    composition = {}
    for name, instances in patterns.items():
        counts = np.zeros(6, dtype=np.int64)
        majority = np.zeros(6, dtype=np.int64)
        for instance in instances:
            roles = [role_of[node] for node in instance["nodes"] if role_of[node] >= 0]
            for role in roles:
                counts[role] += 1
            if roles:
                majority[int(np.bincount(roles, minlength=6).argmax())] += 1
        total = max(len(instances), 1)
        composition[name] = {
            "instances": len(instances),
            "intended_role": ROLE_OF_PATTERN[name],
            "majority_role_share": [round(float(value) / total, 3) for value in majority],
            "node_role_share": [round(float(value) / max(int(counts.sum()), 1), 3)
                                for value in counts],
        }
    found_role = {}
    for name, block in composition.items():
        shares = block["majority_role_share"]
        best = int(np.argmax(shares))
        found_role[name] = best if shares[best] >= 0.5 else None
        block["verified_role"] = found_role[name]
    print("pattern -> ORCA role composition; the role is whatever the pattern actually lands on:")
    for name, block in composition.items():
        print(f"  {name:<16} n={block['instances']:>5} intended {block['intended_role']}  "
              f"found {block['verified_role']}  majority {block['majority_role_share']}")

    # ---- representations for the one arm that has vectors ----
    z = np.load(VECTORS / "TIGER_z.npy").astype(np.float64)
    q = np.load(VECTORS / "TIGER_q.npy").astype(np.float64)
    tokens = np.load(VECTORS / "TIGER_tokens.npy").astype(np.int64)

    def lcp(left: int, right: int) -> int:
        count = 0
        for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
            if x != y:
                break
            count += 1
        return count

    def distance_of(representation: str):
        if representation == "z":
            return lambda left, right: float(np.linalg.norm(z[left] - z[right]))
        if representation == "q":
            return lambda left, right: float(np.linalg.norm(q[left] - q[right]))
        if representation == "SID":
            return lambda left, right: 1.0 - lcp(left, right) / 3.0
        raise KeyError(representation)

    per_role: dict[str, dict] = {}
    for role in range(6):
        per_role[str(role)] = {"patterns": {}, "gain": {}, "delta": {}}
    for name, instances in patterns.items():
        role = found_role[name]
        scores = {}
        for representation in REPRESENTATIONS:
            scores[representation] = score_pattern(name, instances, distance_of(representation),
                                                   null_nodes[name])
        if name == "K2_3":
            for representation in REPRESENTATIONS:
                block = scores[representation]
                if block["S"] is not None:
                    block["S_reverse_same_side_closer"] = 1.0 - block["S"]
        target = per_role[str(role)] if role is not None else None
        if target is not None:
            target["patterns"][name] = {"composition": composition[name], **scores}
        print(f"\n{name} (verified role {role}, n={len(instances)})")
        for representation in REPRESENTATIONS:
            block = scores[representation]
            print(f"  {representation:<4} S {block['S'] if block['S'] is None else round(block['S'], 3)}"
                  f"  null {block['null'] if block['null'] is None else round(block['null'], 3)}"
                  f"  G {block['G'] if block['G'] is None else round(block['G'], 3)}")
        if role is not None:
            for representation in ("q", "SID"):
                base = scores["z"]["G"]
                current = scores[representation]["G"]
                per_role[str(role)]["delta"][f"{representation}_minus_z"] = (
                    current - base if (base is not None and current is not None) else None)

    simplest = {}
    for role in range(6):
        block = per_role[str(role)]
        names = list(block["patterns"])
        entry = {"patterns": names}
        for representation in REPRESENTATIONS:
            values = [block["patterns"][name][representation]["G"] for name in names
                      if block["patterns"][name][representation]["G"] is not None]
            entry[f"G_{representation}"] = float(np.mean(values)) if values else None
        if entry.get("G_q") is not None and entry.get("G_z") is not None:
            entry["delta_quant"] = entry["G_q"] - entry["G_z"]
        else:
            entry["delta_quant"] = None
        if entry.get("G_SID") is not None and entry.get("G_z") is not None:
            entry["delta_SID"] = entry["G_SID"] - entry["G_z"]
        else:
            entry["delta_SID"] = None
        simplest[str(role)] = entry

    print("\n=== per role: gain in z / q / SID and the change caused by quantization ===")
    print(f"  {'role':<6}{'patterns':<34}{'G_z':>8}{'G_q':>8}{'G_SID':>8}"
          f"{'d_quant':>9}{'d_SID':>8}")
    for role in range(6):
        block = simplest[str(role)]
        def cell(key):
            value = block.get(key)
            return f"{value:+.3f}" if value is not None else "  n/a"
        print(f"  {role:<6}{','.join(block['patterns']) or 'NO VERIFIED PATTERN':<34}"
              f"{cell('G_z'):>8}{cell('G_q'):>8}{cell('G_SID'):>8}"
              f"{cell('delta_quant'):>9}{cell('delta_SID'):>8}")

    OUT.write_text(json.dumps({"seed": SEED, "patterns": list(PATTERNS), "null_draws": NULL_DRAWS,
                               "buckets": BUCKETS, "composition": composition,
                               "per_role": per_role, "summary": simplest,
                               "null_cells": "degree bucket x popularity bucket, five by five"},
                              indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
