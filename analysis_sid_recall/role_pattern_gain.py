"""Per-role structure gain built from the graphlets a role's items actually take part in.

The owner's correction: a role is not a graphlet, so nothing requires a role to be described by one
five-node shape.  Instead the graphlet formulas stay as they are, and each role's gain is the weighted
average of the gains of the patterns its items actually participate in:

    G_r = sum_g w_{r,g} * ( S_{r,g}^{real} - S_{r,g}^{null} ),   w_{r,g} = 1 / |M_r|

Three pre-declared rules, fixed before any recall is read:

    * a pattern instance contributes to every role that owns at least one of its five nodes, and the
      contribution is averaged **per item first**, then over items: an item that sits in many
      graphlets counts once, which is what keeps hubs from dominating
    * a pattern enters a role's set M_r only if at least MIN_ITEMS items of that role touch it
    * weights are equal inside M_r; no weight was fitted, and none can be changed after the fact

Scores per arm and per role are then correlated with `test_recall@10` (Spearman, Pearson, permutation
p-values, partial coefficients controlling the level-3 collision rate).

    python3 analysis_sid_recall/role_pattern_gain.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orca5_role_preservation as m  # noqa: E402

LEDGER = m.LEDGER
MIRROR = m.MIRROR
OUT = HERE / "role_pattern_gain.json"
RECALL = m.RECALL
MIN_ITEMS = 30
SUBSET = 1000
SEED = 42
PERMUTATIONS = 10_000


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


def partial(first: np.ndarray, second: np.ndarray, control: np.ndarray, rank: bool) -> float:
    if rank:
        first, second, control = average_ranks(first), average_ranks(second), average_ranks(control)
    design = np.column_stack([np.ones_like(control), control])
    residual_a = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    residual_b = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return pearson(residual_a, residual_b)


def permutation_p(first: np.ndarray, second: np.ndarray, observed: dict) -> dict:
    rng = np.random.default_rng(SEED)
    beats = {"spearman": 0, "pearson": 0}
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        if abs(spearman(first, shuffled)) >= abs(observed["spearman"]):
            beats["spearman"] += 1
        if abs(pearson(first, shuffled)) >= abs(observed["pearson"]):
            beats["pearson"] += 1
    return {key: (value + 1) / (PERMUTATIONS + 1) for key, value in beats.items()}


def per_instance(pattern: str, instance: dict, draws: list[tuple], distance) -> tuple[float, float] | None:
    """The pattern's psi on the real instance and on its matched groups, both per instance."""
    nodes = tuple(instance["nodes"])
    if pattern == "K5":
        inside_real = float(np.mean([distance(nodes[a], nodes[b])
                                     for a, b in m.combinations(range(5), 2)]))
        inside_null = [float(np.mean([distance(drawn[a], drawn[b])
                                      for a, b in m.combinations(range(5), 2)]))
                       for drawn in draws]
        if not inside_null:
            return None
        real = float(np.mean([m.psi(inside_real, value) for value in inside_null]))
        substituted = []
        for position, value in enumerate(inside_null):
            others = [item for index, item in enumerate(inside_null) if index != position]
            if others:
                substituted.append(m.psi(value, float(np.mean(others))))
        if not substituted:
            return None
        return real, float(np.mean(substituted))
    real = m.hypothesis(pattern, m.instance_distances(pattern, nodes, distance))
    responses = [m.hypothesis(pattern, m.instance_distances(pattern, drawn, distance))
                 for drawn in draws]
    if not responses:
        return None
    return real, float(np.mean(responses))


def role_scores(values_real: np.ndarray, values_null: np.ndarray,
                incidence: dict[int, list[int]], role_of: np.ndarray,
                role: int) -> tuple[float, float] | None:
    """Average per item first, then over the items of the role."""
    items = [item for item in incidence if role_of[item] == role]
    if not items:
        return None
    real_by_item, null_by_item = [], []
    for item in items:
        indices = incidence[item]
        real_by_item.append(float(np.mean(values_real[indices])))
        null_by_item.append(float(np.mean(values_null[indices])))
    return float(np.mean(real_by_item)), float(np.mean(null_by_item))


def prepare() -> dict:
    """The instance set, the matched null groups, the subsets and the role map, all with fixed seeds."""
    n, adjacency, _ = m.load_graph()
    rng = np.random.default_rng(m.SEED)
    cliques4 = m.enumerate_cliques(adjacency, 4)
    cliques5 = m.enumerate_cliques(adjacency, 5)
    print(f"cliques: {len(cliques4)} of size 4, {len(cliques5)} of size 5", flush=True)
    patterns = m.build_patterns(adjacency, rng, cliques4, cliques5)
    popularity = m.load_popularity(n)
    degree_bucket = m.bucketise(
        np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64), m.BUCKETS)
    popularity_bucket = m.bucketise(popularity, m.BUCKETS)
    cell_of = degree_bucket * m.BUCKETS + popularity_bucket
    cell_pool = [np.flatnonzero(cell_of == cell) for cell in range(m.BUCKETS * m.BUCKETS)]

    null_groups: dict[str, list[list[tuple]]] = {}
    for name, instances in patterns.items():
        blocks = []
        for instance in instances:
            draws = []
            for _ in range(m.NULL_DRAWS):
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
        null_groups[name] = blocks

    role_of = np.full(n, -1, dtype=np.int64)
    cached = np.load(m.ROLE_CACHE5)
    for node, role in zip(cached["nodes"], cached["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)

    subset_rng = np.random.default_rng(m.SEED + 3)
    subsets, incidences = {}, {}
    for name, instances in patterns.items():
        if len(instances) <= SUBSET:
            picked = list(range(len(instances)))
        else:
            picked = sorted(subset_rng.choice(len(instances), size=SUBSET, replace=False).tolist())
        chosen = [instances[index] for index in picked]
        blocks = [null_groups[name][index] for index in picked]
        subsets[name] = (chosen, blocks)
        incidence: dict[int, list[int]] = defaultdict(list)
        for index, instance in enumerate(chosen):
            for node in instance["nodes"]:
                incidence[node].append(index)
        incidences[name] = incidence
        print(f"  {name:<16} instances {len(chosen):>5}  distinct items {len(incidence):>5}", flush=True)

    patterns_by_role = {}
    for role in range(6):
        patterns_by_role[role] = [name for name, incidence in incidences.items()
                                  if sum(1 for item in incidence if role_of[item] == role) >= MIN_ITEMS]
        print(f"  Role {role + 1}: patterns {patterns_by_role[role] or 'none'}", flush=True)
    return {"n": n, "patterns": patterns, "subsets": subsets, "incidences": incidences,
            "role_of": role_of, "patterns_by_role": patterns_by_role}


def role_gain(per_pattern: dict, patterns_by_role: list[str], role: int,
              role_of: np.ndarray) -> float | None:
    """Equal-weight average over the patterns the role's items actually take part in."""
    gains = []
    for name in patterns_by_role:
        values_real, values_null, incidence = per_pattern[name]
        if values_real.size == 0:
            continue
        scores = role_scores(values_real, values_null, incidence, role_of, role)
        if scores is None:
            continue
        gains.append(scores[0] - scores[1])
    return float(np.mean(gains)) if gains else None


def main() -> int:
    prepared = prepare()
    n = prepared["n"]
    patterns = prepared["patterns"]
    subsets = prepared["subsets"]
    incidences = prepared["incidences"]
    role_of = prepared["role_of"]
    patterns_by_role = prepared["patterns_by_role"]

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
        arms.append({"number": number, "recall": float(recall), "sid_path": sid_path,
                     "arm": entry.get("arm") or number})
    print(f"{len(arms)} arms", flush=True)

    rows = []
    for count, arm in enumerate(arms, start=1):
        payload = json.loads(arm["sid_path"].read_text())
        tokens = np.full((n, 3), -1, dtype=np.int64)
        for key, row in payload.items():
            item = int(key)
            if 0 <= item < n:
                tokens[item] = [int(code) for code in row[:3]]

        def distance(left: int, right: int) -> float:
            matches = 0
            for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
                if x != y:
                    break
                matches += 1
            return 1.0 - matches / 3.0

        triples = {(int(row[0]), int(row[1]), int(row[2])) for row in tokens[tokens[:, 0] >= 0]}
        collision = 1.0 - len(triples) / max(int((tokens[:, 0] >= 0).sum()), 1)

        per_pattern = {}
        for name in patterns:
            chosen, blocks = subsets[name]
            values_real, values_null = [], []
            for instance, draws in zip(chosen, blocks):
                if not draws:
                    continue
                result = per_instance(name, instance, draws, distance)
                if result is None:
                    continue
                values_real.append(result[0])
                values_null.append(result[1])
            per_pattern[name] = (np.array(values_real), np.array(values_null),
                                 incidences[name])

        role_gains = {}
        for role in range(6):
            gains, weights = [], []
            for name in patterns_by_role[role]:
                values_real, values_null, incidence = per_pattern[name]
                if values_real.size == 0:
                    continue
                scores = role_scores(values_real, values_null, incidence, role_of, role)
                if scores is None:
                    continue
                gains.append(scores[0] - scores[1])
                weights.append(1.0)
            if gains:
                weights = np.array(weights) / np.sum(weights)
                role_gains[str(role)] = float(np.dot(weights, np.array(gains)))
            else:
                role_gains[str(role)] = None
        rows.append({"number": arm["number"], "recall": arm["recall"],
                     "l3_collision": collision, "role_gain": role_gains})
        print(f"  [{count}/{len(arms)}] {arm['number']} recall {arm['recall']:.6f}  " +
              " ".join(f"r{role}:{role_gains[str(role)]:+.3f}" if role_gains[str(role)] is not None
                       else f"r{role}:n/a" for role in range(6)), flush=True)

    def correlate(role: int) -> dict:
        values = [(row["role_gain"][str(role)], row["recall"], row["l3_collision"]) for row in rows]
        values = [triple for triple in values if triple[0] is not None]
        if len(values) < 5:
            return {"n": len(values), "spearman": None}
        first = np.array([a for a, _, _ in values])
        second = np.array([b for _, b, _ in values])
        control = np.array([c for _, _, c in values])
        observed = {"spearman": spearman(first, second), "pearson": pearson(first, second)}
        p_values = permutation_p(first, second, observed)
        return {"n": len(values), **observed, "p_spearman": p_values["spearman"],
                "p_pearson": p_values["pearson"],
                "mean_gain": float(first.mean()), "min_gain": float(first.min()),
                "max_gain": float(first.max()),
                "partial_spearman_given_collision": partial(first, second, control, True),
                "partial_pearson_given_collision": partial(first, second, control, False)}

    result = {"n_arms": len(rows), "min_items": MIN_ITEMS, "subset": SUBSET,
              "weights": "equal inside each role's pattern set, fixed before any recall was read",
              "patterns_by_role": {str(role): patterns_by_role[role] for role in range(6)},
              "roles": {}, "rows": rows}
    print(f"\n=== per-role aggregated gain vs test_recall@10 (n={len(rows)}) ===")
    print(f"  {'role':<8}{'patterns':<40}{'gain mean':>10}{'range':>16}{'rho':>8}{'p':>7}"
          f"{'r':>8}{'p':>7}{'partS':>9}{'partP':>9}")
    for role in range(6):
        block = correlate(role)
        result["roles"][str(role)] = {**block, "patterns": patterns_by_role[role]}
        if block.get("spearman") is None:
            print(f"  Role {role + 1:<3}{'none':<40}")
            continue
        spread = f"{block['min_gain']:+.3f}..{block['max_gain']:+.3f}"
        print(f"  Role {role + 1:<3}{','.join(patterns_by_role[role])[:38]:<40}"
              f"{block['mean_gain']:>+10.3f}{spread:>16}"
              f"{block['spearman']:>+8.3f}{block['p_spearman']:>7.3f}"
              f"{block['pearson']:>+8.3f}{block['p_pearson']:>7.3f}"
              f"{block['partial_spearman_given_collision']:>+9.3f}"
              f"{block['partial_pearson_given_collision']:>+9.3f}")
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(m.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
