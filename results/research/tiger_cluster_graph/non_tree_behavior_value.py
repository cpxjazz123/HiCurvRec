"""Does the non-tree part of text similarity predict held-out behaviour?

Experiment:
  * hierarchy-only baseline: tree relation (same fine / same coarse) + popularity,
    degree, category-size and history-length controls;
  * add the non-tree residual of sentence-T5 cosine similarity, after subtracting
    the mean cosine of the same tree relation estimated on random item pairs;
  * fit on TRAIN next-item pairs, evaluate on VALID and TEST future targets.

Positives: (last item of history -> target) in each split.  Negatives: items from
the same training-popularity bin, excluding the user's history and target.
Test is evaluation only; no test pair is used to fit or to estimate residual means.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/non_tree_behavior_value.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
LABELS = REPO / "results" / "research" / "semantic_tree_hypothesis" / "semantic_tree_labels.csv"
BEHAVIOUR = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
DATA = REPO / "results" / "stage0_build_parquet"
OUT_DIR = REPO / "results" / "research" / "non_tree_behavior_value"
OUT_JSON = OUT_DIR / "non_tree_behavior_value.json"
OUT_MD = OUT_DIR / "non_tree_behavior_value.md"
SEED = 20261121
NEG_TRAIN = 3
NEG_EVAL = 5
BOOTSTRAP = 300
FEATURE_NAMES = [
    "same_fine", "same_coarse",
    "log_deg_source", "log_deg_target", "log_pop_source", "log_pop_target",
    "log_fine_source", "log_coarse_source", "log_fine_target", "log_coarse_target",
    "history_length", "cosine", "non_tree_residual",
]
BASE_FEATURES = FEATURE_NAMES[:11]
COSINE_FEATURES = BASE_FEATURES + ["cosine"]
RESIDUAL_FEATURES = BASE_FEATURES + ["non_tree_residual"]
BOTH_FEATURES = BASE_FEATURES + ["cosine", "non_tree_residual"]


def relation(left: int, right: int, paths: dict[str, str], coarse: dict[str, str]) -> str:
    if paths[left] == paths[right]:
        return "same_fine"
    if coarse[left] == coarse[right]:
        return "same_coarse_diff_fine"
    return "cross_coarse"


def load_data():
    labels = pd.read_csv(LABELS).drop_duplicates("item_id")
    paths = {int(row.item_id): str(row.path) for row in labels.itertuples(index=False)}
    coarse = {int(row.item_id): str(row.coarse) for row in labels.itertuples(index=False)}
    fine_size = {item: 0 for item in paths}
    coarse_size = {item: 0 for item in paths}
    for counts, target in ((pd.Series([paths[i] for i in paths]).value_counts(), fine_size),
                           (pd.Series([coarse[i] for i in paths]).value_counts(), coarse_size)):
        for item in paths:
            target[item] = int(counts[paths[item]] if target is fine_size else counts[coarse[item]])
    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)

    edges = []
    with BEHAVIOUR.open() as handle:
        for line in handle:
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            edges.append((left, right))
    edge_array = np.asarray(edges, dtype=np.int64)
    degree = np.bincount(edge_array.ravel(), minlength=len(embedding))
    train = pd.read_parquet(DATA / "train.parquet")
    pop = np.bincount(train["target"].to_numpy(dtype=np.int64), minlength=len(embedding))
    return paths, coarse, fine_size, coarse_size, embedding, degree, pop, edge_array, train


def category_residual_embeddings(paths: dict, embedding: np.ndarray) -> np.ndarray:
    """Remove each item's fine-category mean direction.

    The remaining item vector is semantic variation that the category leaf does
    not explain; pair cosine on these residuals is the non-tree feature.  A
    singleton fine category has zero residual.
    """
    groups = defaultdict(list)
    for item, path in paths.items():
        groups[path].append(item)
    residual = embedding.copy()
    for members in groups.values():
        if len(members) < 2:
            residual[members] = 0.0
            continue
        centroid = embedding[members].mean(axis=0)
        residual[members] = embedding[members] - centroid
    norm = np.linalg.norm(residual, axis=1, keepdims=True)
    residual /= np.maximum(norm, 1e-12)
    return residual


def candidates_by_bin(items: list[int], degree: np.ndarray) -> tuple[list[np.ndarray], dict[int, int]]:
    values = np.asarray([degree[item] for item in items], dtype=float)
    bins = pd.qcut(values, 5, labels=False, duplicates="drop")
    output = []
    mapping = {}
    for index, item in enumerate(items):
        mapping[int(item)] = int(bins[index])
    for raw in sorted(set(int(x) for x in bins)):
        output.append(np.asarray([item for index, item in enumerate(items) if int(bins[index]) == raw], dtype=np.int64))
    # raw bins may be 0..4; keep a dense index
    dense = {raw: index for index, raw in enumerate(sorted(set(int(x) for x in bins)))}
    mapping = {item: dense[raw] for item, raw in mapping.items()}
    return output, mapping


def build_pairs(frame: pd.DataFrame, split: str, paths: dict, coarse: dict,
                fine_size: dict, coarse_size: dict, embedding: np.ndarray,
                degree: np.ndarray, pop: np.ndarray,
                group_list: list[np.ndarray], group_of: dict[int, int],
                residual_embedding: np.ndarray, n_neg: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    all_labeled = np.fromiter(paths.keys(), dtype=np.int64)
    for row in frame.itertuples(index=False):
        history = [int(x) for x in row.history]
        if not history:
            continue
        source, target = history[-1], int(row.target)
        if source not in paths or target not in paths or target in history:
            continue
        seen = set(history)
        group = group_of[target]
        pool = group_list[group]
        candidates = [(source, target, 1)]
        for _ in range(n_neg):
            for _attempt in range(100):
                negative = int(pool[rng.integers(0, len(pool))])
                if negative not in seen and negative != target:
                    break
            else:
                negative = int(all_labeled[rng.integers(0, len(all_labeled))])
            candidates.append((source, negative, 0))
        for left, right, label in candidates:
            rel = relation(left, right, paths, coarse)
            cosine = float(np.dot(embedding[left], embedding[right]))
            residual_cosine = float(np.dot(residual_embedding[left], residual_embedding[right]))
            rows.append({
                "split": split, "user": int(row.user), "left": left, "right": right,
                "label": label, "relation": rel,
                "same_fine": float(rel == "same_fine"),
                "same_coarse": float(rel != "cross_coarse"),
                "log_deg_source": float(np.log1p(degree[left])),
                "log_deg_target": float(np.log1p(degree[right])),
                "log_pop_source": float(np.log1p(pop[left])),
                "log_pop_target": float(np.log1p(pop[right])),
                "log_fine_source": float(np.log1p(fine_size[left])),
                "log_coarse_source": float(np.log1p(coarse_size[left])),
                "log_fine_target": float(np.log1p(fine_size[right])),
                "log_coarse_target": float(np.log1p(coarse_size[right])),
                "history_length": float(len(history)),
                "cosine": cosine,
                "non_tree_residual": residual_cosine,
            })
    return pd.DataFrame(rows)


def fit_models(train: pd.DataFrame) -> dict:
    models = {}
    for name, features in (
        ("hierarchy_controls", BASE_FEATURES),
        ("+ raw cosine", COSINE_FEATURES),
        ("+ non-tree residual", RESIDUAL_FEATURES),
        ("+ both", BOTH_FEATURES),
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
        model.fit(train[features], train["label"])
        models[name] = (model, features)
    return models


def metrics(model, frame: pd.DataFrame, features: list[str]) -> dict:
    probability = model.predict_proba(frame[features])[:, 1]
    return {
        "n": int(len(frame)),
        "positives": int(frame["label"].sum()),
        "auc": float(roc_auc_score(frame["label"], probability)),
        "ap": float(average_precision_score(frame["label"], probability)),
        "log_loss": float(log_loss(frame["label"], probability)),
    }


def bootstrap_delta(models: dict, frame: pd.DataFrame, users: np.ndarray,
                    repeats: int = BOOTSTRAP) -> dict:
    predictions = {name: model.predict_proba(frame[features])[:, 1]
                   for name, (model, features) in models.items()}
    groups = defaultdict(list)
    for index, user in enumerate(users):
        groups[int(user)].append(index)
    unique_users = np.asarray(list(groups.keys()))
    rng = np.random.default_rng(20261122)
    output = {}
    for candidate in ("+ raw cosine", "+ non-tree residual", "+ both"):
        deltas = []
        for _ in range(repeats):
            chosen = unique_users[rng.integers(0, len(unique_users), len(unique_users))]
            indices = np.concatenate([groups[int(user)] for user in chosen])
            labels = frame["label"].to_numpy()[indices]
            if labels.min() == labels.max():
                continue
            base_auc = roc_auc_score(labels, predictions["hierarchy_controls"][indices])
            cand_auc = roc_auc_score(labels, predictions[candidate][indices])
            deltas.append(cand_auc - base_auc)
        output[candidate] = {
            "mean": float(np.mean(deltas)),
            "ci_low": float(np.quantile(deltas, 0.025)),
            "ci_high": float(np.quantile(deltas, 0.975)),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
        }
    return output


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths, coarse, fine_size, coarse_size, embedding, degree, pop, edges, train = load_data()
    residual_embedding = category_residual_embeddings(paths, embedding)
    items = list(paths.keys())
    group_list, group_of = candidates_by_bin(items, degree)
    rng = np.random.default_rng(SEED)

    valid = pd.read_parquet(DATA / "valid.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    train_pairs = build_pairs(train, "train", paths, coarse, fine_size, coarse_size,
                              embedding, degree, pop, group_list, group_of,
                              residual_embedding, NEG_TRAIN, rng)
    valid_pairs = build_pairs(valid, "valid", paths, coarse, fine_size, coarse_size,
                              embedding, degree, pop, group_list, group_of,
                              residual_embedding, NEG_EVAL, rng)
    test_pairs = build_pairs(test, "test", paths, coarse, fine_size, coarse_size,
                             embedding, degree, pop, group_list, group_of,
                             residual_embedding, NEG_EVAL, rng)
    print("train pairs", len(train_pairs), "valid", len(valid_pairs), "test", len(test_pairs), flush=True)

    models = fit_models(train_pairs)
    results = {
        "pair_counts": {"train": len(train_pairs), "valid": len(valid_pairs), "test": len(test_pairs)},
        "models": {},
        "bootstrap_test": {},
    }
    for split, frame in (("valid", valid_pairs), ("test", test_pairs)):
        results["models"][split] = {
            name: metrics(model, frame, features) for name, (model, features) in models.items()
        }
    results["bootstrap_test"] = bootstrap_delta(models, test_pairs, test_pairs["user"].to_numpy())

    # Within the same tree relation, does the residual model still beat the base?
    within = {}
    for name in ("same_fine", "same_coarse_diff_fine", "cross_coarse"):
        subset = test_pairs[test_pairs["relation"] == name].reset_index(drop=True)
        if subset["label"].nunique() < 2:
            continue
        within[name] = {
            model_name: metrics(model, subset, features)
            for model_name, (model, features) in models.items()
        }
    results["within_tree_relation_test"] = within

    OUT_JSON.write_text(json.dumps(results, indent=2))
    lines = [
        "# Non-tree text similarity and held-out behaviour",
        "",
        "Positives: (last history item -> target) in TRAIN/VALID/TEST.",
        "Negatives: same training-popularity bin, excluding the user's history and target.",
        "Non-tree residual = cosine of item vectors after subtracting each item's fine-category mean.",
        "This removes the category leaf direction; singleton fine categories have zero residual.",
        "",
        f"Pairs: train **{len(train_pairs)}**, valid **{len(valid_pairs)}**, test **{len(test_pairs)}**.",
        "",
        "## Models",
        "",
        "| split | model | n | AUC | AP | log loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        for name, value in results["models"][split].items():
            lines.append(f"| {split} | {name} | {value['n']} | {value['auc']:.4f} | "
                         f"{value['ap']:.4f} | {value['log_loss']:.4f} |")
    lines += [
        "",
        "## Paired user bootstrap on test",
        "",
        "| candidate | mean ΔAUC | 95% CI | fraction positive |",
        "|---|---:|---:|---:|",
    ]
    for name, value in results["bootstrap_test"].items():
        lines.append(f"| {name} | {value['mean']:+.4f} | "
                     f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] | "
                     f"{value['positive_fraction']:.3f} |")
    lines += [
        "",
        "## Within tree relation (test AUC)",
        "",
        "| relation | base | + raw cosine | + non-tree residual | + both |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in results["within_tree_relation_test"].items():
        lines.append(f"| {name} | {values['hierarchy_controls']['auc']:.4f} | "
                     f"{values['+ raw cosine']['auc']:.4f} | "
                     f"{values['+ non-tree residual']['auc']:.4f} | "
                     f"{values['+ both']['auc']:.4f} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
