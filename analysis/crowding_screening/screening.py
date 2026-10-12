"""Fixed seed-42 crowding screen on the held-out validation pair set."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "results/crowding_screening"
BASE_DIR = ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
TIGER_DIR = ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE"
TRAINER_DIR = ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
PAIR_PATH = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
TRAIN_PATH = ROOT / "results/stage0_build_parquet/train.parquet"
VALID_PATH = ROOT / "results/stage0_build_parquet/valid.parquet"
ITEM_PATH = ROOT / "results/stage0_build_parquet/items.parquet"
EMBEDDING_PATH = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
K_VALUES = (5, 20, 50)
BOOTSTRAPS = 200
SEED = 42
STRENGTHS = (1, 2, 3)
MECHANISMS = ("posattract", "couser", "simweight", "density", "softassign",
              "prefixshare", "random", "scale", "parammatch")
CROWDING_NAMES = tuple(f"crowd_{mech}_w{strength}" for mech in MECHANISMS for strength in STRENGTHS)


def _load_modules():
    diagnostic = ROOT / "analysis/crowding_diagnostic"
    sys.path.insert(0, str(diagnostic))
    import distances
    fork_scripts = ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"
    sys.path.insert(0, str(fork_scripts))
    import analyze_prefix_behavior as AP
    import eval_behavior_metrics as EM
    return distances, AP, EM


def _co_purchase_graph(n_items: int) -> np.ndarray:
    """Build the full symmetric train-only user co-purchase graph."""
    train = pd.read_parquet(TRAIN_PATH)
    adjacency = np.zeros((n_items, n_items), dtype=bool)
    for _, interactions in train.groupby("user", sort=False):
        basket = np.unique(interactions.target.to_numpy(dtype=np.int64))
        basket = basket[(basket >= 0) & (basket < n_items)]
        adjacency[np.ix_(basket, basket)] = True
    np.fill_diagonal(adjacency, False)
    print("BEHAVIOUR_GRAPH=train.parquet full user co-purchase adjacency; graph_s42.npz stores only top-50 edges")
    return adjacency


def _relations(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    adjacency = _co_purchase_graph(n_items)
    successors = np.zeros_like(adjacency)
    for frame in (pd.read_parquet(TRAIN_PATH), pd.read_parquet(VALID_PATH)):
        for row in frame.itertuples(index=False):
            history = np.asarray(row.history, dtype=np.int64)
            if len(history) and 0 <= int(history[-1]) < n_items and 0 <= int(row.target) < n_items:
                successors[int(history[-1]), int(row.target)] = True
    return adjacency | successors | successors.T, successors



def load_compatible_state(model, state_dict: dict):
    """Load model parameters and ignore auxiliary heads absent from inference."""
    expected = model.state_dict()
    missing = expected.keys() - state_dict.keys()
    if missing:
        raise RuntimeError(f"Checkpoint is missing model parameters: {sorted(missing)}")
    compatible = {key: value for key, value in state_dict.items() if key in expected}
    return model.load_state_dict(compatible, strict=True)
def _load_residuals(checkpoint: Path, n_items: int) -> np.ndarray:
    """Extract the three residual vectors with the existing TIGER trainer/model."""
    import torch
    trainer_path = TRAINER_DIR / "train_rqvae.py"
    sys.path.insert(0, str(TRAINER_DIR))
    spec = importlib.util.spec_from_file_location("crowding_screen_trainer", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    from model import RQVAE
    embeddings = np.load(EMBEDDING_PATH, mmap_mode="r")
    if embeddings.shape[0] != n_items:
        raise ValueError(f"Embedding/item alignment mismatch: {embeddings.shape[0]} != {n_items}")
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_compatible_state(model, state["state_dict"])
    model.eval()
    vectors = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    with torch.no_grad():
        trace = trainer._residuals_with_trace(model, vectors)
    return trace.cpu().numpy().transpose(1, 0, 2)


def _cluster_ci(values: np.ndarray, anchors: np.ndarray, seed: int) -> list[float] | None:
    from screening_metrics import anchor_bootstrap_ci
    return anchor_bootstrap_ci(values, anchors, seed=seed, replicates=BOOTSTRAPS)

def _bootstrap_stat_ci(values: np.ndarray, statistic, seed: int) -> list[float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)
    for draw in range(BOOTSTRAPS):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[draw] = statistic(values[sample])
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _code_stats_ci(codes: np.ndarray, seed: int) -> tuple[dict, dict]:
    from screening_metrics import item_bootstrap_ci, occupancy_stats
    stats = occupancy_stats(codes, capacity=256)
    intervals = {}
    for offset, key in enumerate(("entropy", "gini")):
        bound = (0.0, float(np.log2(256)) if key == "entropy" else 1.0)
        result = item_bootstrap_ci(
            codes, lambda sample, key=key: occupancy_stats(sample, capacity=256)[key],
            seed + offset, BOOTSTRAPS, bound)
        intervals[f"{key}_ci"] = result["ci"]
    return stats, intervals


def _metric_record(residual: np.ndarray, levels: np.ndarray, pairs: dict,
                   relations: np.ndarray, successors: np.ndarray, popularity: np.ndarray,
                   distances, EM, AP) -> tuple[dict, list[dict]]:
    a, p, n = pairs["anchors"], pairs["positives"], pairs["negatives"]
    metrics = {}
    rows = []
    for level in range(3):
        vectors = residual[:, level, :]
        pos = np.linalg.norm(vectors[a] - vectors[p], axis=1)
        neg_matrix = np.linalg.norm(vectors[a, None, :] - vectors[n], axis=2)
        auc_values = (neg_matrix > pos[:, None]).astype(float) + 0.5 * (neg_matrix == pos[:, None])
        err_values = (neg_matrix < pos[:, None]).astype(float) + 0.5 * (neg_matrix == pos[:, None])
        auc_per_pair = auc_values.mean(axis=1)
        err_per_pair = err_values.mean(axis=1)
        nearest_successor_distance = np.full(len(a), np.inf)
        for i, anchor in enumerate(a):
            positive_ids = np.flatnonzero(successors[anchor])
            if len(positive_ids):
                nearest_successor_distance[i] = np.linalg.norm(vectors[positive_ids] - vectors[anchor], axis=1).min()
        intrusion = (neg_matrix < nearest_successor_distance[:, None]).mean(axis=1)
        rec = {"auc": distances.auc_neg_farther_than_pos(pos, neg_matrix),
               "auc_ci": _cluster_ci(auc_per_pair, a, SEED + level),
               "ordering_error": distances.ordering_error_rate(pos, neg_matrix),
               "ordering_error_ci": _cluster_ci(err_per_pair, a, SEED + 10 + level),
               "intrusion": float(intrusion.mean()),
               "intrusion_ci": _cluster_ci(intrusion, a, SEED + 20 + level)}
        from screening_metrics import local_purity, occupancy_stats
        for k in K_VALUES:
            purity = local_purity(vectors, relations, popularity, k=k, seed=SEED + level * 100 + k)
            rec[f"purity_{k}"] = purity["purity"]
            rec[f"null_purity_{k}"] = purity["null_purity"]
            rec[f"purity_lift_{k}"] = purity["purity_lift"]
            rec[f"purity_lift_{k}_ci"] = _cluster_ci(purity["per_item"] - purity["null_per_item"],
                                                       np.arange(len(vectors)), SEED + 30 + level * 100 + k)
            top = purity["neighbours"]
            successors_share = successors[np.arange(len(top))[:, None], top].mean(axis=1)
            candidate_sets = {}
            for anchor, negatives in zip(pairs["anchors"], pairs["negatives"]):
                candidate_sets.setdefault(int(anchor), set()).update(map(int, negatives))
            matched_share = np.asarray([
                np.mean([int(item) in candidate_sets.get(anchor, ()) for item in row])
                for anchor, row in enumerate(top)
            ])
            successor_ratio = float(successors_share.mean() / matched_share.mean()) if matched_share.mean() else None
            multiplicity = successors.sum(axis=1)
            for label, select in (("one_successor", multiplicity == 1),
                                  ("multiple_successors", multiplicity >= 2)):
                if select.any():
                    group_purity = purity["per_item"][select]
                    group_null = purity["null_per_item"][select]
                    rec[f"purity_{k}_{label}"] = float(group_purity.mean())
                    rec[f"null_purity_{k}_{label}"] = float(group_null.mean())
                    rec[f"purity_lift_{k}_{label}"] = float((group_purity - group_null).mean())
                    rec[f"purity_lift_{k}_{label}_ci"] = _cluster_ci(
                        group_purity - group_null, np.flatnonzero(select), SEED + 50 + k + level)
                else:
                    for prefix in ("purity", "null_purity", "purity_lift"):
                        rec[f"{prefix}_{k}_{label}"] = None
                    rec[f"purity_lift_{k}_{label}_ci"] = None
            rec[f"successor_share_{k}"] = float(successors_share.mean())
            rec[f"successor_share_{k}_ci"] = _cluster_ci(successors_share, np.arange(len(top)), SEED + 60 + k + level)
            rec[f"matched_negative_share_{k}"] = float(matched_share.mean())
            rec[f"matched_negative_share_{k}_ci"] = _cluster_ci(matched_share, np.arange(len(top)), SEED + 70 + k + level)
            rec[f"successor_negative_ratio_{k}"] = successor_ratio
            rec[f"successor_negative_ratio_{k}_ci"] = None if successor_ratio is None else _bootstrap_stat_ci(
                np.arange(len(top)), lambda sample: float(successors_share[sample].mean() /
                    matched_share[sample].mean()) if matched_share[sample].mean() else np.nan, SEED + 80 + k + level)
        code_stats, code_ci = _code_stats_ci(levels[:, level], SEED + level)
        rec.update({f"{key}_{level + 1}": value for key, value in code_stats.items()})
        rec.update(code_ci)
        rows.append({"level": level + 1, **rec})
        metrics[f"L{level + 1}"] = rec
    code_tuples = np.unique(levels[:, :3], axis=0)
    metrics["raw_sid_distinct"] = int(len(code_tuples))
    metrics["raw_sid_collision"] = int(len(levels) - len(code_tuples))
    metrics["raw_sid_collision_rate"] = float(1 - len(code_tuples) / len(levels))
    _, inverse, counts = np.unique(levels[:, :3], axis=0, return_inverse=True, return_counts=True)
    collision_contribution = 1.0 - 1.0 / counts[inverse]
    from screening_metrics import item_bootstrap_ci
    collision_result = item_bootstrap_ci(
        collision_contribution, np.mean, SEED + 90, BOOTSTRAPS, (0.0, 1.0))
    metrics["raw_sid_collision_rate_ci"] = collision_result["ci"]
    behavior = EM.metrics_for(levels, pairs)
    behavior_ci = {}
    for depth, key in ((1, "L1"), (2, "L1_L2"), (3, "L1_L2_L3")):
        prefix = AP.prefix_ids(levels, depth)
        pos_share = (prefix[p] == prefix[a]).astype(float)
        neg_share = (prefix[n] == prefix[a, None]).astype(float)
        unique, inverse = np.unique(a, return_inverse=True)
        pos_sum = np.bincount(inverse, weights=pos_share, minlength=len(unique))
        neg_sum = np.bincount(inverse, weights=neg_share.mean(axis=1), minlength=len(unique))
        anchor_count = np.bincount(inverse, minlength=len(unique))
        estimates = np.empty((BOOTSTRAPS, 3), dtype=np.float64)
        rng = np.random.default_rng(SEED + 100 + depth)
        for draw in range(BOOTSTRAPS):
            sampled = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            denom = np.dot(sampled, anchor_count)
            ps = np.dot(sampled, pos_sum) / denom
            ns = np.dot(sampled, neg_sum) / denom
            estimates[draw] = (ps, ns, ps / ns if ns > 0 else np.nan)
        behavior_ci[key] = {name: ([float(np.nanpercentile(estimates[:, col], 2.5)),
                                    float(np.nanpercentile(estimates[:, col], 97.5))]
                                   if np.isfinite(estimates[:, col]).any() else None)
                            for col, name in enumerate(("p_share_pos", "p_share_neg", "er"))}
    metrics["prefix_behavior"] = behavior
    metrics["prefix_behavior_ci"] = behavior_ci
    return metrics, rows


def _screen_record(metrics: dict, level: str, strength: int | None) -> dict:
    row = metrics[level]
    result = {"purity_lift": row.get("purity_lift_20"),
              "purity_lift_ci": row.get("purity_lift_20_ci"),
              "auc": row.get("auc"), "auc_ci": row.get("auc_ci"),
              "ordering_error": row.get("ordering_error"), "ordering_error_ci": row.get("ordering_error_ci"),
              "intrusion": row.get("intrusion"), "intrusion_ci": row.get("intrusion_ci"),
              "entropy": row.get("entropy_" + level[1:]), "entropy_ci": row.get("entropy_ci"),
              "gini": row.get("gini_" + level[1:]), "gini_ci": row.get("gini_ci"),
              "collision": metrics.get("raw_sid_collision_rate"),
              "collision_ci": metrics.get("raw_sid_collision_rate_ci")}
    return result


def _detailed_rows(arm: str, metrics: dict) -> list[dict]:
    behavior = metrics["prefix_behavior"]
    behavior_ci = metrics["prefix_behavior_ci"]
    levels = (("L1", "L1"), ("L2", "L1_L2"), ("L3", "L1_L2_L3"))
    output = []
    for level, prefix in levels:
        row = {"arm": arm, "level": level, **metrics[level]}
        for key in ("p_share_pos", "p_share_neg", "er"):
            row[f"{prefix}_{key}"] = behavior[prefix].get(key)
            row[f"{prefix}_{key}_ci"] = behavior_ci[prefix].get(key)
        row.update({key: metrics[key] for key in
                    ("raw_sid_distinct", "raw_sid_collision", "raw_sid_collision_rate",
                     "raw_sid_collision_rate_ci")})
        output.append(row)
    return output

def run() -> list[dict]:
    from screening_metrics import classify_arm
    distances, AP, EM = _load_modules()
    pair_data = np.load(PAIR_PATH)
    pairs = {k: pair_data[k].astype(np.int64) for k in ("anchors", "positives", "negatives")}
    levels_data = pd.read_parquet(ITEM_PATH)
    n_items = int(levels_data.item_id.max()) + 1
    relations, successors = _relations(n_items)
    train = pd.read_parquet(TRAIN_PATH)
    popularity = np.bincount(np.concatenate((
        train.target.to_numpy(dtype=np.int64),
        np.concatenate(train.history.to_list()).astype(np.int64)
    )), minlength=n_items)[:n_items]
    arms = {"TIGER": TIGER_DIR, "Exp1a": BASE_DIR / "resample"}
    for name in ("resample_l1only", "resample_l2only", "resample_l3only", "resample_proj_sym",
                 "resample_proj_dir", "resample_fn_mask", "resample_s43", "resample_s44"):
        arms[name] = BASE_DIR / name
    for name in CROWDING_NAMES:
        arms[name] = BASE_DIR / name
    loaded = {}
    detailed_rows = []
    for name, directory in arms.items():
        sid_path = directory / "item_sids.json"
        checkpoint = directory / "rqvae_best.pth"
        if not sid_path.is_file() or not checkpoint.is_file():
            continue
        levels, _ = AP.load_levels(sid_path)
        if len(levels) != n_items:
            raise ValueError(f"{name}: SID count {len(levels)} != item count {n_items}")
        cache_label = {"resample_l1only": "l1only", "resample_l2only": "l2only",
                       "resample_l3only": "l3only"}.get(name, name)
        cache = ROOT / f"results/crowding_diagnostic/residuals_{cache_label}.npz"
        if cache.is_file():
            with np.load(cache) as data:
                residual = data["residuals"]
        else:
            residual = _load_residuals(checkpoint, n_items)
        metrics, table_rows = _metric_record(residual, levels, pairs, relations, successors,
                                             popularity, distances, EM, AP)
        loaded[name] = {"metrics": metrics, "levels": levels}
        detailed_rows.extend(_detailed_rows(name, metrics))
        print("ARM_METRICS," + json.dumps({"arm": name, "prefix_behavior": metrics["prefix_behavior"],
                                           "prefix_behavior_ci": metrics["prefix_behavior_ci"],
                                           "raw_sid_collision_rate_ci": metrics["raw_sid_collision_rate_ci"]},
                                          sort_keys=True))
        for row in table_rows:
            row["arm"] = name
            print("METRIC," + json.dumps(row, sort_keys=True))
        del residual
    base = loaded.get("Exp1a")
    table = []
    missing = [name for name in CROWDING_NAMES if name not in loaded]
    for name, value in loaded.items():
        for level in ("L1", "L2", "L3"):
            result = None
            if name not in CROWDING_NAMES:
                classification = "UNCLASSIFIED"
                reason = ["matched_strength_random_control"]
                arm_cmp = _screen_record(value["metrics"], level, None)
                base_cmp = _screen_record(base["metrics"], level, None) if base else None
                random_cmp = None
            else:
                strength = int(name.rsplit("w", 1)[1])
                control_name = f"crowd_random_w{strength}"
                arm_cmp = _screen_record(value["metrics"], level, strength)
                base_cmp = _screen_record(base["metrics"], level, strength) if base else None
                random_cmp = (_screen_record(loaded[control_name]["metrics"], level, strength)
                              if control_name in loaded else None)
                if base is None or random_cmp is None:
                    missing_controls = []
                    if base is None:
                        missing_controls.append("Exp1a")
                    if random_cmp is None:
                        missing_controls.append(f"matched_strength_random_control:{control_name}")
                    result = {"classification": "UNCLASSIFIED", "missing_metrics": missing_controls}
                else:
                    result = classify_arm(arm_cmp, base_cmp, random_cmp, name)
                classification, reason = result["classification"], result.get("missing_metrics", [])
            row = {"arm": name, "level": level, "strength": (int(name.rsplit("w", 1)[1])
                   if name in CROWDING_NAMES else None), "classification": classification,
                   "missing_metrics": ";".join(reason),
                   "arm_metrics_ci": json.dumps(arm_cmp, sort_keys=True),
                   "baseline_metrics_ci": json.dumps(base_cmp, sort_keys=True),
                   "random_control_metrics_ci": json.dumps(random_cmp, sort_keys=True),
                   "comparison": json.dumps(result.get("comparisons", {}) if result else {}, sort_keys=True)}
            table.append(row)
            print("SCREEN," + json.dumps(row, sort_keys=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "screening_table_complete.csv"
    with table_path.open("w", newline="") as stream:
        fields = ("arm", "level", "strength", "classification", "missing_metrics",
                  "arm_metrics_ci", "baseline_metrics_ci", "random_control_metrics_ci", "comparison")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    screens = {(row["arm"], row["level"]): row for row in table}
    for row in detailed_rows:
        screen = screens.get((row["arm"], row["level"]), {})
        row["classification"] = screen.get("classification")
        row["comparison_cis"] = {key: screen.get(key) for key in
                                 ("arm_metrics_ci", "baseline_metrics_ci", "random_control_metrics_ci")}
    metrics_json = OUT_DIR / "metrics_by_arm_level.json"
    metrics_csv = OUT_DIR / "metrics_by_arm_level.csv"
    metrics_json.write_text(json.dumps(detailed_rows, indent=2, sort_keys=True, allow_nan=False))
    metric_fields = list(dict.fromkeys(key for row in detailed_rows for key in row))
    with metrics_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    print(f"ARM_LEVEL_METRICS_JSON={metrics_json}")
    print(f"ARM_LEVEL_METRICS_CSV={metrics_csv}")
    print("MISSING_CROWDING_ARMS=" + ",".join(missing))
    print(f"SCREENING_TABLE={table_path}")
    return table


if __name__ == "__main__":
    run()
