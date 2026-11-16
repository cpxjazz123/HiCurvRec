"""Do different purchase contexts of the same item have different futures and structures?

For each selected target item X:

    context   global text cluster of the item bought immediately before X
    outcome   global text cluster of the item bought immediately after X
    structure the TRAIN edge (predecessor, X): graphlet pattern / triangle count / roles

TRAIN discovers P(outcome | context) and P(structure | context); TEST is the independent future
data.  The context never sees X's outcome.  A matched permutation test shuffles context labels
inside the same target (and, for the structure test, inside predecessor-degree quintiles).

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/context_structure_validation/validate.py
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OCC = HERE / "occurrences.parquet"
TARGETS = HERE / "targets.json"
CLUSTERS = HERE / "text_clusters.json"
ITEM_FEATURES = REPO / "results" / "research" / "graphlet_relation_sid" / "item_features.parquet"
ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
OUT_JSON = HERE / "analysis.json"
OUT_MD = HERE / "tables.md"

MIN_CONTEXT = 40
MIN_OUTCOME = 20
KEEP_OUTCOMES = 12
ALPHA = 1.0
PERMUTATIONS = 500
SEED = 42


def mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    _, left_codes = np.unique(left, return_inverse=True)
    _, right_codes = np.unique(right, return_inverse=True)
    rows = int(left_codes.max()) + 1
    columns = int(right_codes.max()) + 1
    counts = np.bincount(left_codes * columns + right_codes, minlength=rows * columns)
    table = counts.reshape(rows, columns).astype(np.float64)
    total = table.sum()
    if total == 0:
        return 0.0
    probability = table / total
    row = probability.sum(axis=1, keepdims=True)
    column = probability.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = probability > 0
    return float((probability[mask] * np.log(probability[mask] / independent[mask])).sum())


def permutation_p(left: np.ndarray, right: np.ndarray, rng: np.random.Generator,
                  permutations: int = PERMUTATIONS) -> float:
    observed = mutual_information(left, right)
    null = np.array([mutual_information(rng.permutation(left), right) for _ in range(permutations)])
    return float((1.0 + (null >= observed).sum()) / (permutations + 1.0))


def cramers_v(left: np.ndarray, right: np.ndarray) -> float:
    table = pd.crosstab(left, right).to_numpy(dtype=np.float64)
    total = table.sum()
    if total == 0 or min(table.shape) < 2:
        return 0.0
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
    chi2 = ((table - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    return float(math.sqrt(chi2 / (total * (min(table.shape) - 1))))


def stratified_permutation(left: np.ndarray, right: np.ndarray, strata: np.ndarray,
                           rng: np.random.Generator, permutations: int = PERMUTATIONS) -> dict:
    """MI weighted by stratum size, with context shuffled inside each stratum."""
    def statistic(a):
        total = 0.0
        weight = 0.0
        for value in np.unique(strata):
            mask = strata == value
            if mask.sum() < 10:
                continue
            total += mutual_information(a[mask], right[mask]) * mask.sum()
            weight += mask.sum()
        return total / weight if weight else 0.0

    observed = statistic(left)
    null = np.array([statistic(np.concatenate([
        rng.permutation(left[strata == value]) for value in np.unique(strata)
    ])) for _ in range(permutations)])
    return {"mi": float(observed), "p": float((1.0 + (null >= observed).sum()) / (permutations + 1.0))}


def jsd(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / max(first.sum(), 1e-12)
    second = second / max(second.sum(), 1e-12)
    middle = 0.5 * (first + second)
    value = 0.0
    for left, right in ((first, middle), (second, middle)):
        mask = left > 0
        value += 0.5 * float((left[mask] * np.log(left[mask] / right[mask])).sum())
    return value


def weighted_jsd(table: np.ndarray) -> tuple[float, float]:
    rows = len(table)
    if rows < 2:
        return 0.0, 0.0
    total_weight = 0.0
    total = 0.0
    maximum = 0.0
    for first in range(rows):
        for second in range(first + 1, rows):
            value = jsd(table[first], table[second])
            weight = table[first].sum() * table[second].sum()
            total += value * weight
            total_weight += weight
            maximum = max(maximum, value)
    return float(total / total_weight) if total_weight else 0.0, float(maximum)


def conditional(table: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    table = table.astype(np.float64)
    return (table + alpha) / (table.sum(axis=1, keepdims=True) + alpha * table.shape[1])


def marginal(table: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    counts = table.sum(axis=0).astype(np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    picked = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def top_titles(values, titles, k=5):
    counts = pd.Series(values).value_counts().head(k)
    return [{"item": int(item), "count": int(count), "title": titles[int(item)]} for item, count in counts.items()]


def incremental_structure_gain(
    train_seen: pd.DataFrame,
    eval_context: np.ndarray,
    eval_outcome: np.ndarray,
    eval_pattern: np.ndarray,
    context_order: list[int],
    outcome_order: list[int],
    pattern_order: list[str],
    context_probability: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train P(outcome|context,pattern) with backoff and evaluate it on the same TEST rows.

    When ``rng`` is given, pattern labels are permuted **within each context** in TRAIN, which keeps
    the context marginals and the pattern marginals but breaks the context->pattern association.  The
    resulting gain distribution is the null for "structure adds information beyond context".
    """
    context_map = {value: index for index, value in enumerate(context_order)}
    outcome_map = {value: index for index, value in enumerate(outcome_order + [-1])}
    pattern_map = {value: index for index, value in enumerate(pattern_order)}
    contexts = train_seen["context_c"].to_numpy()
    outcomes = train_seen["outcome_c"].to_numpy()
    patterns = train_seen["pattern"].to_numpy().copy()
    if rng is not None:
        for context in np.unique(contexts):
            mask = contexts == context
            patterns[mask] = rng.permutation(patterns[mask])
    c_index = np.array([context_map[int(value)] for value in contexts])
    o_index = np.array([outcome_map[int(value)] for value in outcomes])
    p_index = np.array([pattern_map[value] for value in patterns])
    n_contexts = len(context_order)
    n_patterns = len(pattern_order)
    n_outcomes = len(outcome_order) + 1
    codes = (c_index * n_patterns + p_index) * n_outcomes + o_index
    counts = np.bincount(codes, minlength=n_contexts * n_patterns * n_outcomes).reshape(
        n_contexts, n_patterns, n_outcomes
    ).astype(np.float64)
    probability = np.zeros_like(counts)
    for context in range(n_contexts):
        if counts[context].sum() > 0:
            probability[context] = conditional(counts[context])
    predictions = context_probability[eval_context].copy()
    for row, (context, pattern) in enumerate(zip(eval_context, eval_pattern)):
        cell = counts[context, pattern_map[pattern]]
        if cell.sum() >= 10:
            predictions[row] = probability[context, pattern_map[pattern]]
    context_loss = log_loss(context_probability[eval_context], eval_outcome)
    final_loss = log_loss(predictions, eval_outcome)
    return {"gain": context_loss - final_loss, "context_loss": context_loss, "final_loss": final_loss}


def collapse_context(series: pd.Series, keep: set[int]) -> pd.Series:
    return series.where(series.isin(keep), -1)


def collapse_outcome(series: pd.Series, keep: set[int]) -> pd.Series:
    return series.where(series.isin(keep), -1)


def main() -> int:
    occurrences = pd.read_parquet(OCC)
    targets = json.loads(TARGETS.read_text())
    clusters = json.loads(CLUSTERS.read_text())
    items = pd.read_parquet(ITEMS).sort_values("item_id").reset_index(drop=True)
    titles = items["title"].tolist()
    item_features = pd.read_parquet(ITEM_FEATURES).set_index("item_id")

    occurrences = occurrences.merge(
        item_features[["degree", "popularity"]].rename(
            columns={"degree": "predecessor_degree", "popularity": "predecessor_popularity"}
        ),
        left_on="predecessor", right_index=True, how="left",
    )
    occurrences["degree_quintile"] = pd.qcut(
        occurrences["predecessor_degree"].rank(method="first"), 5, labels=False, duplicates="drop"
    )

    rng = np.random.default_rng(SEED)
    result = {
        "config": {
            "occ": str(OCC.relative_to(REPO)),
            "targets": targets,
            "context_clusters": clusters,
            "min_context": MIN_CONTEXT,
            "min_outcome": MIN_OUTCOME,
            "keep_outcomes": KEEP_OUTCOMES,
            "permutations": PERMUTATIONS,
        },
        "targets": {},
    }

    for target_text, meta in targets.items():
        target = int(target_text)
        frame = occurrences[occurrences["target"] == target].copy()
        train = frame[frame["split"] == "train"].copy()
        test = frame[frame["split"] == "test"].copy()

        context_counts = train["context"].value_counts()
        kept_contexts = set(context_counts[context_counts >= MIN_CONTEXT].index.astype(int))
        outcome_counts = train["outcome"].value_counts()
        kept_outcomes = set(outcome_counts.head(KEEP_OUTCOMES).index.astype(int))

        for part in (train, test):
            part["context_c"] = collapse_context(part["context"], kept_contexts)
            part["outcome_c"] = collapse_outcome(part["outcome"], kept_outcomes)

        context_order = sorted(kept_contexts)
        outcome_order = sorted(kept_outcomes)
        train_table = pd.crosstab(train["context_c"], train["outcome_c"]).reindex(
            index=context_order + [-1], columns=outcome_order + [-1], fill_value=0
        ).to_numpy()
        test_table = pd.crosstab(test["context_c"], test["outcome_c"]).reindex(
            index=context_order + [-1], columns=outcome_order + [-1], fill_value=0
        ).to_numpy()

        train_mi = mutual_information(train["context_c"].to_numpy(), train["outcome_c"].to_numpy())
        test_mi = mutual_information(test["context_c"].to_numpy(), test["outcome_c"].to_numpy())
        first_test = test.sort_values(["user", "position"]).drop_duplicates("user")
        first_mi = mutual_information(first_test["context_c"].to_numpy(), first_test["outcome_c"].to_numpy())
        first_p = permutation_p(first_test["context_c"].to_numpy(), first_test["outcome_c"].to_numpy(), rng)
        train_v = cramers_v(train["context_c"].to_numpy(), train["outcome_c"].to_numpy())
        test_v = cramers_v(test["context_c"].to_numpy(), test["outcome_c"].to_numpy())
        train_p = permutation_p(train["context_c"].to_numpy(), train["outcome_c"].to_numpy(), rng)
        test_p = permutation_p(test["context_c"].to_numpy(), test["outcome_c"].to_numpy(), rng)
        train_jsd, train_jsd_max = weighted_jsd(train_table)
        test_jsd, test_jsd_max = weighted_jsd(test_table)

        probability = conditional(train_table)
        baseline_probability = marginal(train_table)
        context_index = {value: index for index, value in enumerate(context_order + [-1])}
        outcome_index = {value: index for index, value in enumerate(outcome_order + [-1])}
        test_context = np.array([context_index[int(value)] for value in test["context_c"]])
        test_outcome = np.array([outcome_index[int(value)] for value in test["outcome_c"]])
        test_logloss = log_loss(probability[test_context], test_outcome)
        baseline_logloss = log_loss(np.tile(baseline_probability, (len(test_outcome), 1)), test_outcome)
        context_accuracy = float((probability[test_context].argmax(axis=1) == test_outcome).mean())
        baseline_accuracy = float((baseline_probability.argmax() == test_outcome).mean())

        # train/test profile stability per context
        profile = []
        for context in context_order:
            train_row = train[train["context_c"] == context]
            test_row = test[test["context_c"] == context]
            if len(train_row) < 20 or len(test_row) < 20:
                continue
            a = train_row["outcome_c"].value_counts().reindex(outcome_order + [-1], fill_value=0).to_numpy()
            b = test_row["outcome_c"].value_counts().reindex(outcome_order + [-1], fill_value=0).to_numpy()
            profile.append((int(context), len(train_row), len(test_row), jsd(a, b)))
        profile_jsd = (
            float(np.average([row[3] for row in profile], weights=[min(row[1], row[2]) for row in profile]))
            if profile else float("nan")
        )

        # structure association on seen TRAIN edges
        train_seen = train[train["seen_edge"] == 1].copy()
        test_seen = test[test["seen_edge"] == 1].copy()
        pattern_mi = mutual_information(train_seen["context_c"].to_numpy(), train_seen["pattern"].to_numpy())
        pattern_p = permutation_p(train_seen["context_c"].to_numpy(), train_seen["pattern"].to_numpy(), rng)
        pattern_v = cramers_v(train_seen["context_c"].to_numpy(), train_seen["pattern"].to_numpy())
        tri_mi = mutual_information(train_seen["context_c"].to_numpy(), train_seen["tri_bin"].to_numpy())
        tri_p = permutation_p(train_seen["context_c"].to_numpy(), train_seen["tri_bin"].to_numpy(), rng)
        tri_v = cramers_v(train_seen["context_c"].to_numpy(), train_seen["tri_bin"].to_numpy())
        strat_pattern = stratified_permutation(
            train_seen["context_c"].to_numpy(), train_seen["pattern"].to_numpy(),
            train_seen["degree_quintile"].to_numpy(), rng,
        )
        strat_tri = stratified_permutation(
            train_seen["context_c"].to_numpy(), train_seen["tri_bin"].to_numpy(),
            train_seen["degree_quintile"].to_numpy(), rng,
        )

        # does structure add predictive value beyond context?
        use = test_seen[test_seen["pattern"] != "unseen"].copy()
        use = use[use["context_c"].isin(context_order)]
        use = use[use["pattern"].isin(["none", "K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5", "multi"])]
        eval_context = np.array([context_index[int(value)] for value in use["context_c"]])
        eval_outcome = np.array([outcome_index[int(value)] for value in use["outcome_c"]])
        eval_pattern = use["pattern"].to_numpy()

        train_seen_context = train[train["context_c"].isin(context_order)]
        table_context = pd.crosstab(train_seen_context["context_c"], train_seen_context["outcome_c"]).reindex(
            index=context_order, columns=outcome_order + [-1], fill_value=0
        ).to_numpy()
        prob_context = conditional(table_context)

        pattern_order = sorted(train_seen["pattern"].unique().tolist())
        pattern_map = {value: index for index, value in enumerate(pattern_order)}
        # a test edge can carry a pattern absent from this target's TRAIN edges; it cannot be scored
        # by a train conditional table, so it is excluded from the incremental comparison
        present = use["pattern"].isin(pattern_map)
        use = use[present]
        eval_context = np.array([context_index[int(value)] for value in use["context_c"]])
        eval_outcome = np.array([outcome_index[int(value)] for value in use["outcome_c"]])
        eval_pattern = use["pattern"].to_numpy()
        train_seen_model = train_seen[train_seen["context_c"].isin(context_order)]
        observed = incremental_structure_gain(
            train_seen_model, eval_context, eval_outcome, eval_pattern,
            context_order, outcome_order, pattern_order, prob_context,
        )
        null = np.array([
            incremental_structure_gain(
                train_seen_model, eval_context, eval_outcome, eval_pattern,
                context_order, outcome_order, pattern_order, prob_context, rng=rng,
            )["gain"]
            for _ in range(200)
        ])
        increment_p = float((1.0 + (null >= observed["gain"]).sum()) / (len(null) + 1.0))

        train_tri = train_seen_model.copy()
        train_tri["pattern"] = train_tri["tri_bin"]
        tri_order = sorted(train_tri["pattern"].unique().tolist())
        eval_tri = use["tri_bin"].to_numpy()
        observed_tri = incremental_structure_gain(
            train_tri, eval_context, eval_outcome, eval_tri,
            context_order, outcome_order, tri_order, prob_context,
        )
        null_tri = np.array([
            incremental_structure_gain(
                train_tri, eval_context, eval_outcome, eval_tri,
                context_order, outcome_order, tri_order, prob_context, rng=rng,
            )["gain"]
            for _ in range(200)
        ])
        tri_increment_p = float((1.0 + (null_tri >= observed_tri["gain"]).sum()) / (len(null_tri) + 1.0))

        ll_context = observed["context_loss"]
        ll_final = observed["final_loss"]

        structure_only = np.zeros((len(pattern_order), len(outcome_order) + 1))
        for pattern in pattern_order:
            group = train_seen[train_seen["pattern"] == pattern]
            structure_only[pattern_map[pattern]] = group["outcome_c"].value_counts().reindex(
                outcome_order + [-1], fill_value=0
            ).to_numpy()
        structure_prob = conditional(structure_only)
        ll_structure = log_loss(structure_prob[[pattern_map[p] for p in eval_pattern]], eval_outcome)
        baseline_use = log_loss(np.tile(baseline_probability, (len(eval_outcome), 1)), eval_outcome)

        # context signatures and structure signatures
        context_rows = []
        for context in context_order:
            group = train[train["context_c"] == context]
            if len(group) == 0:
                continue
            context_rows.append({
                "context": int(context),
                "n_train": int(len(group)),
                "n_test": int((test["context_c"] == context).sum()),
                "top_predecessors": top_titles(group["predecessor"], titles, 4),
                "top_next": top_titles(group["next"], titles, 4),
                "top_outcomes": [
                    {"cluster": int(value), "count": int(count)}
                    for value, count in group["outcome_c"].value_counts().head(3).items()
                ],
                "pattern_share": {
                    value: int(count) for value, count in group["pattern"].value_counts().items()
                },
                "mean_triangles": float(group["triangles"].mean()) if group["triangles"].notna().any() else None,
                "mean_predecessor_degree": float(group["predecessor_degree"].mean()),
            })

        context_pairs = []
        for first, second in combinations(context_order, 2):
            first_rows = train[train["context_c"] == first]
            second_rows = train[train["context_c"] == second]
            if len(first_rows) < MIN_CONTEXT or len(second_rows) < MIN_CONTEXT:
                continue
            first_distribution = first_rows["outcome_c"].value_counts().reindex(
                outcome_order + [-1], fill_value=0
            ).to_numpy()
            second_distribution = second_rows["outcome_c"].value_counts().reindex(
                outcome_order + [-1], fill_value=0
            ).to_numpy()
            context_pairs.append({
                "first": int(first),
                "second": int(second),
                "n_first": int(len(first_rows)),
                "n_second": int(len(second_rows)),
                "jsd": jsd(first_distribution, second_distribution),
                "first_next": top_titles(first_rows["next"], titles, 4),
                "second_next": top_titles(second_rows["next"], titles, 4),
            })
        context_pairs.sort(key=lambda row: row["jsd"], reverse=True)

        pattern_rows = []
        for pattern in pattern_order:
            group = train_seen[train_seen["pattern"] == pattern]
            if len(group) < 10:
                continue
            pattern_rows.append({
                "pattern": pattern,
                "n_train": int(len(group)),
                "top_contexts": [
                    {"context": int(value), "count": int(count)}
                    for value, count in group["context_c"].value_counts().head(3).items()
                ],
                "top_next": top_titles(group["next"], titles, 3),
                "mean_triangles": float(group["triangles"].mean()),
            })

        result["targets"][target_text] = {
            "title": meta["title"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "seen_edge_share_test": float(test["seen_edge"].mean()),
            "train": {
                "mi_context_outcome": train_mi,
                "mi_p": train_p,
                "cramers_v": train_v,
                "weighted_jsd": train_jsd,
                "max_jsd": train_jsd_max,
            },
            "test": {
                "mi_context_outcome": test_mi,
                "mi_p": test_p,
                "first_event_mi": first_mi,
                "first_event_p": first_p,
                "first_event_n": int(len(first_test)),
                "cramers_v": test_v,
                "weighted_jsd": test_jsd,
                "max_jsd": test_jsd_max,
                "logloss_context": test_logloss,
                "logloss_baseline": baseline_logloss,
                "logloss_gain": baseline_logloss - test_logloss,
                "accuracy_context": context_accuracy,
                "accuracy_baseline": baseline_accuracy,
                "accuracy_gain": context_accuracy - baseline_accuracy,
                "profile_jsd_train_test": profile_jsd,
                "n_profile_contexts": len(profile),
            },
            "structure": {
                "pattern_mi": pattern_mi,
                "pattern_p": pattern_p,
                "pattern_cramers_v": pattern_v,
                "pattern_mi_degree_stratified": strat_pattern["mi"],
                "pattern_p_degree_stratified": strat_pattern["p"],
                "tri_mi": tri_mi,
                "tri_p": tri_p,
                "tri_cramers_v": tri_v,
                "tri_mi_degree_stratified": strat_tri["mi"],
                "tri_p_degree_stratified": strat_tri["p"],
                "n_seen_train": int(len(train_seen)),
                "n_seen_test": int(len(test_seen)),
                "n_eval_structure": int(len(use)),
                "logloss_context_only": ll_context,
                "logloss_structure_only": ll_structure,
                "logloss_context_plus_structure": ll_final,
                "logloss_gain_structure_only": baseline_use - ll_structure,
                "logloss_gain_structure_beyond_context": ll_context - ll_final,
                "increment_p": increment_p,
                "increment_null_mean": float(null.mean()),
                "tri_increment_gain": observed_tri["gain"],
                "tri_increment_p": tri_increment_p,
                "tri_increment_null_mean": float(null_tri.mean()),
            },
            "contexts": context_rows,
            "context_pairs": context_pairs,
            "patterns": pattern_rows,
        }
        print(f"{target}: test MI {test_mi:.4f} p={test_p:.4f}; pattern MI {pattern_mi:.4f} p={pattern_p:.4f}; "
              f"structure gain {ll_context - ll_final:+.4f}", flush=True)

    train_all = occurrences[
        (occurrences["split"] == "train")
        & (occurrences["target"].isin([int(key) for key in targets]))
        & (occurrences["seen_edge"] == 1)
    ]
    pooled = {}
    for pattern in sorted(train_all["pattern"].unique().tolist()):
        group = train_all[train_all["pattern"] == pattern]
        pooled[pattern] = {
            "n": int(len(group)),
            "n_targets": int(group["target"].nunique()),
            "mean_triangles": float(group["triangles"].mean()),
            "top_contexts": [
                {"context": int(value), "count": int(count)}
                for value, count in group["context"].value_counts().head(4).items()
            ],
            "top_next": top_titles(group["next"], titles, 5),
        }
    result["pooled_patterns"] = pooled

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    # Markdown summary
    lines = ["# Context / structure validation tables\n"]
    lines.append(
        "| target | title | train n | test n | test MI | test p | first MI | first p | Cramér V | test JSD | "
        "log-loss gain | pattern MI | pattern p | degree-strat. p | structure gain beyond context | gain p | null gain |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for key, row in result["targets"].items():
        lines.append(
            f"| {key} | {row['title'][:55]} | {row['n_train']:,} | {row['n_test']:,} | "
            f"{row['test']['mi_context_outcome']:.4f} | {row['test']['mi_p']:.4f} | "
            f"{row['test']['first_event_mi']:.4f} | {row['test']['first_event_p']:.4f} | "
            f"{row['test']['cramers_v']:.3f} | {row['test']['weighted_jsd']:.3f} | "
            f"{row['test']['logloss_gain']:+.4f} | {row['structure']['pattern_mi']:.4f} | "
            f"{row['structure']['pattern_p']:.4f} | {row['structure']['pattern_p_degree_stratified']:.4f} | "
            f"{row['structure']['logloss_gain_structure_beyond_context']:+.4f} | "
            f"{row['structure']['increment_p']:.4f} | {row['structure']['increment_null_mean']:+.4f} |\n"
        )
    lines.append("\n## Does the named graphlet beat a simple triangle-count bin?\n\n")
    lines.append("| target | graphlet gain | graphlet p | triangle-bin gain | triangle-bin p |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for key, value in result["targets"].items():
        structure = value["structure"]
        lines.append(
            f"| {key} | {structure['logloss_gain_structure_beyond_context']:+.4f} | "
            f"{structure['increment_p']:.4f} | {structure['tri_increment_gain']:+.4f} | "
            f"{structure['tri_increment_p']:.4f} |\n"
        )
    primary = next(iter(result["targets"]))
    row = result["targets"][primary]
    lines.append(f"\n## Context signatures: target {primary}\n\n")
    lines.append("| context | n train | n test | top predecessors | top next | pattern share | mean triangles | mean pred degree |\n")
    lines.append("|---|---:|---:|---|---|---|---:|---:|\n")
    for context in row["contexts"]:
        predecessors = "; ".join(item["title"][:42] for item in context["top_predecessors"])
        following = "; ".join(item["title"][:42] for item in context["top_next"])
        patterns = ", ".join(f"{key}:{value}" for key, value in list(context["pattern_share"].items())[:6])
        lines.append(
            f"| C{context['context']} | {context['n_train']:,} | {context['n_test']:,} | {predecessors} | "
            f"{following} | {patterns} | {context['mean_triangles']} | {context['mean_predecessor_degree']:.1f} |\n"
        )
    lines.append(f"\n## Largest context contrasts: target {primary}\n\n")
    lines.append("| context A | n A | context B | n B | JSD | A top next | B top next |\n")
    lines.append("|---|---:|---|---:|---:|---|---|\n")
    for pair in row["context_pairs"][:6]:
        first_next = "; ".join(item["title"][:38] for item in pair["first_next"])
        second_next = "; ".join(item["title"][:38] for item in pair["second_next"])
        lines.append(
            f"| C{pair['first']} | {pair['n_first']:,} | C{pair['second']} | {pair['n_second']:,} | "
            f"{pair['jsd']:.3f} | {first_next} | {second_next} |\n"
        )
    lines.append(f"\n## Structure signatures: target {primary}\n\n")
    lines.append("| pattern | n train | top contexts | top next | mean triangles |\n")
    lines.append("|---|---:|---|---|---:|\n")
    for pattern in row["patterns"]:
        contexts = ", ".join(f"C{item['context']}:{item['count']}" for item in pattern["top_contexts"])
        following = "; ".join(item["title"][:42] for item in pattern["top_next"])
        lines.append(
            f"| {pattern['pattern']} | {pattern['n_train']:,} | {contexts} | {following} | {pattern['mean_triangles']:.2f} |\n"
        )
    lines.append("\n## Pooled structure signatures (all targets, TRAIN)\n\n")
    lines.append("| pattern | n | targets | mean triangles | top contexts | top next |\n")
    lines.append("|---|---:|---:|---:|---|---|\n")
    for pattern, value in result["pooled_patterns"].items():
        contexts = ", ".join(f"C{item['context']}:{item['count']}" for item in value["top_contexts"])
        following = "; ".join(item["title"][:42] for item in value["top_next"])
        lines.append(
            f"| {pattern} | {value['n']:,} | {value['n_targets']} | {value['mean_triangles']:.1f} | "
            f"{contexts} | {following} |\n"
        )
    OUT_MD.write_text("".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
