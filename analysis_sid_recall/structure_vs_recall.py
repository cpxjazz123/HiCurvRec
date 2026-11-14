"""Does a structure-preserving SID predict the downstream recall? Per-structure support vs test_recall@10.

The owner asks for the correlation between each structure's retention and the downstream recommendation
number, across every SID-recall pair the repository already holds. The pairs are the archived ones under
`analysis_sid_recall/r2/<NN>/`: a Stage2 `item_sids.json` next to the Stage3 `recall.json` produced from
it. Only the SID table is needed, so no checkpoint has to be reloaded and nothing is retrained.

For every pair, and for the SID tree alone (the only representation a SID table defines):

    support   the share of real instances that satisfy the structure's own distance hypothesis
              (the hypotheses of section 5.16, on the SID tree's closeness distance)
    gain      support minus the same statistic on degree-matched random role groups
              (the matched null of section 5.17)

and for context the SID-level quantities already in `analysis_sid_recall/report.json`: level-1 codes,
level-2 prefixes, level-3 distinct triples, level-3 collision.

Then Spearman correlations against `test_recall@10`, with a permutation p-value and a bootstrap interval
over pairs. Two readings are reported, as the pairing rule requires: every pair, and one row per distinct
SID (duplicates by sha256 collapse).

    python3 analysis_sid_recall/structure_vs_recall.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_sid_recall"
ARCHIVE = OUT / "r2"
JOURNAL = OUT / "structure_vs_recall.json"
BEHAVIOUR = REPO / "results" / "research" / "issue215" / "graphs" / "behaviour.in"
sys.path.insert(0, str(REPO / "results" / "research" / "issue215"))

import structure_hypothesis_tests as sht  # noqa: E402  (instance builders, validation, metric)

SEED = 42
MAX_INSTANCES = 1000
NULL_DRAWS = 3
STRUCTURES = tuple(name for name in sht.STRUCTURES if name != "edge")
PERMUTATIONS = 10_000


def load_pairs() -> list[dict]:
    pairs: list[dict] = []
    for folder in sorted(ARCHIVE.iterdir()):
        if not folder.is_dir():
            continue
        sid_path, recall_path = folder / "item_sids.json", folder / "recall.json"
        if not (sid_path.is_file() and recall_path.is_file()):
            continue
        recall = json.loads(recall_path.read_text())
        if isinstance(recall, dict) and "test_recall@10" in recall:
            value = float(recall["test_recall@10"])
            source = "recall.json"
        else:
            block = recall.get("test_final", recall)
            value = float(block["test_recall@10"]) if "test_recall@10" in block else None
            source = "recall.json:test_final"
        if value is None:
            continue
        pairs.append({"number": folder.name, "sid_path": str(sid_path), "recall": value,
                      "recall_source": source,
                      "sid_sha256": hashlib.sha256(sid_path.read_bytes()).hexdigest()})
    return pairs


def sid_tokens(path: str, n: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    tokens = np.full((n, 3), -1, dtype=np.int64)
    for key, row in payload.items():
        item = int(key)
        if 0 <= item < n:
            tokens[item] = [int(value) for value in row[:3]]
    return tokens


def tree_distance(tokens: np.ndarray, left: int, right: int) -> float:
    count = 0
    for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
        if x != y:
            break
        count += 1
    return 1.0 - count / 3.0


def structure_scores(tokens: np.ndarray, instances: dict, patterns: dict,
                     null_sets: dict) -> dict:
    """Support and gain per structure, for one SID table's tree distance."""
    scores: dict[str, dict] = {}
    missing = int(np.sum(tokens[:, 0] < 0))
    for name in STRUCTURES:
        support: list[float] = []
        null_support: list[float] = []
        for index, instance in enumerate(instances[name]):
            nodes = instance["nodes"]
            if any(tokens[node, 0] < 0 for node in nodes):
                continue
            pattern = patterns[name]
            if name in ("triangle", "K4"):
                inside = float(np.mean([tree_distance(tokens, a, b)
                                        for a, b in combinations(nodes, 2)]))
                sets = null_sets[name][index]
                references = []
                for drawn in sets:
                    if any(tokens[item, 0] < 0 for item in drawn):
                        continue
                    references.append(float(np.mean([tree_distance(tokens, a, b)
                                                     for a, b in combinations(drawn, 2)])))
                if not references:
                    continue
                support.append(1.0 if inside < float(np.mean(references)) else 0.0)
                # the null substitutes one matched group for another, keeping the same reference
                null_responses = []
                for position in range(len(references)):
                    others = [value for index_other, value in enumerate(references)
                              if index_other != position]
                    if others:
                        null_responses.append(
                            1.0 if references[position] < float(np.mean(others)) else 0.0)
                if null_responses:
                    null_support.append(float(np.mean(null_responses)))
            elif name == "path4":
                hop1 = float(np.mean([tree_distance(tokens, nodes[a], nodes[b])
                                      for a, b in pattern["hop1"]]))
                hop2 = float(np.mean([tree_distance(tokens, nodes[a], nodes[b])
                                      for a, b in pattern["hop2"]]))
                hop3 = float(np.mean([tree_distance(tokens, nodes[a], nodes[b])
                                      for a, b in pattern["hop3"]]))
                support.append(1.0 if hop1 < hop2 < hop3 else 0.0)
                sets = null_sets[name][index]
                responses = []
                for drawn in sets:
                    if any(tokens[item, 0] < 0 for item in drawn):
                        continue
                    first = float(np.mean([tree_distance(tokens, drawn[a], drawn[b])
                                           for a, b in pattern["hop1"]]))
                    second = float(np.mean([tree_distance(tokens, drawn[a], drawn[b])
                                            for a, b in pattern["hop2"]]))
                    third = float(np.mean([tree_distance(tokens, drawn[a], drawn[b])
                                           for a, b in pattern["hop3"]]))
                    responses.append(1.0 if first < second < third else 0.0)
                if responses:
                    null_support.append(float(np.mean(responses)))
            else:
                closer = float(np.mean([tree_distance(tokens, nodes[a], nodes[b])
                                        for a, b in pattern["closer"]]))
                farther = float(np.mean([tree_distance(tokens, nodes[a], nodes[b])
                                         for a, b in pattern["farther"]]))
                support.append(1.0 if closer < farther else 0.0)
                sets = null_sets[name][index]
                responses = []
                for drawn in sets:
                    if any(tokens[item, 0] < 0 for item in drawn):
                        continue
                    near_null = float(np.mean([tree_distance(tokens, drawn[a], drawn[b])
                                               for a, b in pattern["closer"]]))
                    far_null = float(np.mean([tree_distance(tokens, drawn[a], drawn[b])
                                              for a, b in pattern["farther"]]))
                    responses.append(1.0 if near_null < far_null else 0.0)
                if responses:
                    null_support.append(float(np.mean(responses)))
        scores[name] = {
            "instances": len(support),
            "support": float(np.mean(support)) if support else None,
            "null": float(np.mean(null_support)) if null_support else None,
            "gain": (float(np.mean(support)) - float(np.mean(null_support))
                     if support and null_support else None),
            "missing_items": missing,
        }
    return scores


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank 1..n with ties sharing the average position: a support rate is full of ties."""
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
    if len(first) < 3:
        return float("nan")
    rank_a = average_ranks(first) - (len(first) - 1) / 2.0
    rank_b = average_ranks(second) - (len(second) - 1) / 2.0
    denominator = float(np.sqrt((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator else float("nan")


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3:
        return float("nan")
    centred_a = first - first.mean()
    centred_b = second - second.mean()
    denominator = float(np.sqrt((centred_a ** 2).sum() * (centred_b ** 2).sum()))
    return float((centred_a * centred_b).sum() / denominator) if denominator else float("nan")


def partial_pearson(first: np.ndarray, second: np.ndarray, control: np.ndarray) -> float:
    """Linear correlation of first and second after regressing the control out of both."""
    design = np.column_stack([np.ones_like(control), control])
    residual_a = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    residual_b = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return pearson(residual_a, residual_b)


def permutation_p(first: np.ndarray, second: np.ndarray,
                  observed_spearman: float, observed_pearson: float) -> dict:
    """Both statistics in one pass: one shuffle buys one p-value per statistic."""
    rng = np.random.default_rng(SEED)
    beats_spearman = 0
    beats_pearson = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        if abs(spearman(first, shuffled)) >= abs(observed_spearman):
            beats_spearman += 1
        if abs(pearson(first, shuffled)) >= abs(observed_pearson):
            beats_pearson += 1
    return {"spearman": (beats_spearman + 1) / (PERMUTATIONS + 1),
            "pearson": (beats_pearson + 1) / (PERMUTATIONS + 1)}


def partial_spearman(first: np.ndarray, second: np.ndarray, control: np.ndarray) -> float:
    """Spearman of first and second after removing the rank of the control variable from both."""
    rank_a = average_ranks(first)
    rank_b = average_ranks(second)
    rank_c = average_ranks(control)
    design = np.column_stack([np.ones_like(rank_c), rank_c])
    residual_a = rank_a - design @ np.linalg.lstsq(design, rank_a, rcond=None)[0]
    residual_b = rank_b - design @ np.linalg.lstsq(design, rank_b, rcond=None)[0]
    denominator = float(np.sqrt((residual_a ** 2).sum() * (residual_b ** 2).sum()))
    return float((residual_a * residual_b).sum() / denominator) if denominator else float("nan")


def bootstrap_ci(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 1)
    values = []
    for _ in range(2000):
        index = rng.integers(0, len(first), size=len(first))
        values.append(spearman(first[index], second[index]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> int:
    n, adjacency = sht.load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = sht.sample_instances(adjacency, rng)
    cliques = sht.enumerate_cliques(n, adjacency)
    cliques.sort()
    chosen = rng.choice(len(cliques), size=min(len(cliques), MAX_INSTANCES), replace=False)
    instances["K4"] = [{"nodes": list(cliques[int(index)])} for index in sorted(chosen)]
    violations = sht.validate(instances, adjacency)
    if any(violations.values()):
        raise SystemExit(f"structural violations: {violations}")

    patterns = {name: role_partition(instances[name]) for name in STRUCTURES}

    # the null is a property of the graph and the role partition, not of the SID table
    degrees = np.array([len(adjacency[i]) for i in range(n)])
    order = np.argsort(degrees, kind="mergesort")
    bucket_of = np.empty(n, dtype=np.int64)
    bucket_of[order] = np.minimum((np.arange(n) * 5) // max(n, 1), 4)
    bucket_pool = [np.flatnonzero(bucket_of == b) for b in range(5)]
    null_sets: dict[str, list[list[tuple[int, ...]]]] = {}
    for name in STRUCTURES:
        blocks: list[list[tuple[int, ...]]] = []
        for instance in instances[name]:
            real = set(instance["nodes"])
            sets: list[tuple[int, ...]] = []
            for _ in range(NULL_DRAWS):
                drawn: list[int] = []
                for node in instance["nodes"]:
                    pool = bucket_pool[int(bucket_of[node])]
                    for _ in range(64):
                        candidate = int(pool[rng.integers(pool.size)])
                        if candidate in real or candidate in drawn:
                            continue
                        drawn.append(candidate)
                        break
                if len(drawn) == len(instance["nodes"]):
                    sets.append(tuple(drawn))
            blocks.append(sets)
        null_sets[name] = blocks

    pairs = load_pairs()
    print(f"{len(pairs)} archived SID-recall pairs")
    rows: list[dict] = []
    for pair in pairs:
        tokens = sid_tokens(pair["sid_path"], n)
        present = tokens[tokens[:, 0] >= 0]
        l1 = len({int(row[0]) for row in present})
        l2 = len({(int(row[0]), int(row[1])) for row in present})
        triples = {(int(row[0]), int(row[1]), int(row[2])) for row in present}
        collision = 1.0 - len(triples) / max(len(present), 1)
        scores = structure_scores(tokens, instances, patterns, null_sets)
        rows.append({"number": pair["number"], "recall": pair["recall"],
                     "sid_sha256": pair["sid_sha256"], "items": int(len(present)),
                     "l1_codes": l1, "l2_prefixes": l2,
                     "l3_distinct": len(triples), "l3_collision": collision,
                     "structures": scores})
        print(f"  {pair['number']}: recall {pair['recall']:.6f}  l1 {l1}  l2 {l2}  "
              f"collision {collision:.4f}  " +
              " ".join(f"{name}={scores[name]['support']:.3f}"
                       for name in STRUCTURES if scores[name]["support"] is not None))

    def correlate(subset: list[dict], extract, control: str = "l3_collision") -> dict:
        values = [(extract(row), row["recall"], row[control]) for row in subset]
        values = [triple for triple in values if triple[0] is not None]
        if len(values) < 4:
            return {"n": len(values), "rho": None}
        first = np.array([a for a, _, _ in values])
        second = np.array([b for _, b, _ in values])
        control_values = np.array([c for _, _, c in values])
        rho = spearman(first, second)
        linear = pearson(first, second)
        low, high = bootstrap_ci(first, second)
        p_values = permutation_p(first, second, rho, linear)
        # a structure metric could simply track how fragmented the codebook is: report the partial
        return {"n": len(values), "rho": rho, "pearson": linear,
                "p_permutation": p_values["spearman"], "p_permutation_pearson": p_values["pearson"],
                "ci95": [low, high],
                "rho_vs_collision": spearman(first, control_values),
                "pearson_vs_collision": pearson(first, control_values),
                "partial_rho_given_collision": partial_spearman(first, second, control_values),
                "partial_pearson_given_collision": partial_pearson(first, second, control_values)}

    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["sid_sha256"], row)
    unique_rows = list(unique.values())

    result = {"recall_metric": "test_recall@10", "n_pairs": len(rows),
              "n_unique_sid": len(unique_rows), "null_draws": NULL_DRAWS,
              "structures": list(STRUCTURES), "rows": rows}
    for label, subset in (("all", rows), ("unique", unique_rows)):
        result[label] = {"context": {}, "structures": {}}
        for key in ("l1_codes", "l2_prefixes", "l3_distinct", "l3_collision"):
            result[label]["context"][key] = correlate(subset, lambda row, key=key: row[key])
        for name in STRUCTURES:
            result[label]["structures"][name] = {
                "support": correlate(subset, lambda row, name=name:
                                     row["structures"][name]["support"]),
                "gain": correlate(subset, lambda row, name=name:
                                  row["structures"][name]["gain"]),
            }

    JOURNAL.write_text(json.dumps(result, indent=1, default=float))
    for label in ("all", "unique"):
        block = result[label]
        print(f"\n=== Spearman against test_recall@10, {label} pairs "
              f"(n={block['context']['l3_collision'].get('n')}) ===")
        for key, value in block["context"].items():
            if value.get("rho") is not None:
                print(f"  {key:<14} spearman {value['rho']:+.3f} (p {value['p_permutation']:.4f})"
                      f"  pearson {value['pearson']:+.3f} (p {value['p_permutation_pearson']:.4f})"
                      f"  CI [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]")
        print(f"  {'structure':<14}{'sup rho':>9}{'p':>7}{'sup r':>9}{'p':>7}"
              f"{'gain rho':>10}{'p':>7}{'gain r':>9}{'p':>7}")
        for name in STRUCTURES:
            support = block["structures"][name]["support"]
            gain = block["structures"][name]["gain"]
            def cell(entry, key):
                if entry.get(key) is None:
                    return "n/a", ""
                return f"{entry[key]:+.3f}", f"{entry.get('p_permutation' if key == 'rho' else 'p_permutation_pearson', 0):.3f}"
            support_rho, support_rho_p = cell(support, "rho")
            support_r, support_r_p = cell(support, "pearson")
            gain_rho, gain_rho_p = cell(gain, "rho")
            gain_r, gain_r_p = cell(gain, "pearson")
            print(f"  {name:<14}{support_rho:>9}{support_rho_p:>7}{support_r:>9}{support_r_p:>7}"
                  f"{gain_rho:>10}{gain_rho_p:>7}{gain_r:>9}{gain_r_p:>7}")
    return 0


def role_partition(instances: list[dict]) -> dict:
    pattern: dict | None = None
    for instance in instances:
        index = {node: position for position, node in enumerate(instance["nodes"])}
        current = {key: sorted((index[a], index[b]) for a, b in instance.get(key, []))
                   for key in ("closer", "farther", "hop1", "hop2", "hop3")}
        if pattern is None:
            pattern = current
        elif current != pattern:
            raise AssertionError("role partition is not constant across instances")
    return pattern


if __name__ == "__main__":
    raise SystemExit(main())
