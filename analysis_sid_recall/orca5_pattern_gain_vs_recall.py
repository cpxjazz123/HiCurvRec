"""Per-pattern SID structure gain across the archived arms, and its correlation with recall.

The role-aware evaluation (`orca5_role_preservation.py`) showed that the guessed role-to-pattern map
does not hold: every dense five-node pattern it tested lands on ORCA role 0, and the medium-degree and
star patterns land on no role at all.  The patterns themselves are still measurable questions about the
geometry, so this script keeps them and asks the cross-arm question the owner wants in the last column of
the table: for each pattern, does the SID structure gain of an arm predict that arm's `test_recall@10`?

    G_pattern(arm) = mean_instance psi( mean_near, mean_far ) - the same on degree x popularity
                     matched random five-node groups

Instances and null groups are rebuilt with the same seeds as the role-aware run, so the two reports share
one instance set; the per-pattern subset is the first 300 instances to keep 67 arms inside a few minutes.
Correlations: Spearman and Pearson with permutation p-values, plus partial coefficients controlling the
level-3 collision rate.

    python3 analysis_sid_recall/orca5_pattern_gain_vs_recall.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orca5_role_preservation as m  # noqa: E402

LEDGER = HERE / ".." / ".." / "results" / "research" / "issue139" / "r2_ledger.json"
MIRROR = HERE / "r2"
OUT = HERE / "orca5_pattern_gain_vs_recall.json"
SUBSET = 300
RECALL = "test_recall@10"
PERMUTATIONS = 10_000


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    average = (starts + ends) / 2.0
    ranks_sorted = np.repeat(average, ends - starts + 1)
    ranks = np.empty(len(values))
    ranks[order] = ranks_sorted
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a = average_ranks(first)
    b = average_ranks(second)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = first - first.mean()
    b = second - second.mean()
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
    rng = np.random.default_rng(7)
    beats = {"spearman": 0, "pearson": 0}
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        if abs(spearman(first, shuffled)) >= abs(observed["spearman"]):
            beats["spearman"] += 1
        if abs(pearson(first, shuffled)) >= abs(observed["pearson"]):
            beats["pearson"] += 1
    return {key: (value + 1) / (PERMUTATIONS + 1) for key, value in beats.items()}


def main() -> int:
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

    null_nodes: dict[str, list[list[tuple]]] = {}
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
        null_nodes[name] = blocks

    subset_rng = np.random.default_rng(m.SEED + 1)
    subsets = {}
    for name, instances in patterns.items():
        if len(instances) <= SUBSET:
            subsets[name] = (instances, null_nodes[name])
        else:
            picks = np.sort(subset_rng.choice(len(instances), size=SUBSET, replace=False))
            subsets[name] = ([instances[int(index)] for index in picks],
                             [null_nodes[name][int(index)] for index in picks])

    ledger = json.loads(m.LEDGER.read_text())
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
    print(f"{len(arms)} arms with a ledger-verified SID and a recall", flush=True)

    names = list(patterns)
    rows = []
    for count, arm in enumerate(arms, start=1):
        payload = json.loads(arm["sid_path"].read_text())
        tokens = np.full((n, 3), -1, dtype=np.int64)
        for key, row in payload.items():
            item = int(key)
            if 0 <= item < n:
                tokens[item] = [int(code) for code in row[:3]]

        def distance(left: int, right: int) -> float:
            count_matches = 0
            for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
                if x != y:
                    break
                count_matches += 1
            return 1.0 - count_matches / 3.0

        triples = {(int(row[0]), int(row[1]), int(row[2]))
                   for row in tokens[tokens[:, 0] >= 0]}
        collision = 1.0 - len(triples) / max(int((tokens[:, 0] >= 0).sum()), 1)
        entry = {"number": arm["number"], "arm": arm["arm"], "recall": arm["recall"],
                 "l3_collision": collision, "gains": {}}
        for name in names:
            instances, nulls = subsets[name]
            result = m.score_pattern(name, instances, distance, nulls)
            entry["gains"][name] = result["G"]
        rows.append(entry)
        print(f"  [{count}/{len(arms)}] {arm['number']} recall {arm['recall']:.6f}  " +
              " ".join(f"{name}={entry['gains'][name]:+.3f}"
                       for name in names if entry["gains"][name] is not None), flush=True)

    def correlate(extract) -> dict:
        values = [(extract(row), row["recall"], row["l3_collision"]) for row in rows]
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
                "partial_spearman_given_collision": partial(first, second, control, True),
                "partial_pearson_given_collision": partial(first, second, control, False)}

    result = {"n_arms": len(rows), "subset": SUBSET, "recall_metric": RECALL,
              "patterns": names, "correlations": {}, "rows": rows}
    for name in names:
        result["correlations"][name] = correlate(lambda row, name=name: row["gains"][name])
    verified = [name for name, block in
                json.loads((HERE / "orca5_role_preservation.json").read_text())["composition"].items()
                if block.get("verified_role") == 0]
    result["verified_role0"] = verified
    result["correlations"]["role0_mean_of_verified"] = correlate(
        lambda row: float(np.mean([row["gains"][name] for name in verified
                                   if row["gains"][name] is not None])))
    OUT.write_text(json.dumps(result, indent=1, default=float))

    print(f"\n=== per-pattern SID gain vs test_recall@10 (n={len(rows)}, subset {SUBSET}/pattern) ===")
    print(f"  {'pattern':<24}{'spearman':>10}{'p':>8}{'pearson':>10}{'p':>8}"
          f"{'partialS':>10}{'partialP':>10}")
    for name, value in result["correlations"].items():
        if value.get("spearman") is None:
            continue
        print(f"  {name:<24}{value['spearman']:>+10.3f}{value['p_spearman']:>8.3f}"
              f"{value['pearson']:>+10.3f}{value['p_pearson']:>8.3f}"
              f"{value['partial_spearman_given_collision']:>+10.3f}"
              f"{value['partial_pearson_given_collision']:>+10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
