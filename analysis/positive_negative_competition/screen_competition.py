"""Run direct competition measurements on existing seed-42 Stage-2 arms."""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / "analysis/positive_negative_competition"))
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"))
sys.path.insert(0, str(ROOT / "analysis/crowding_diagnostic"))
import competition as C

PAIRS_PATH = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
VALID_PATH = ROOT / "results/stage0_build_parquet/valid.parquet"
EMBEDDINGS_PATH = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
OUTPUT = ROOT / "results/positive_negative_competition"
REFERENCE_SAMPLE_SIZE = 512
SEED = 42
EPSILONS = C.EPSILONS


def _load_pairs():
    with np.load(PAIRS_PATH) as f:
        return {key: f[key] for key in ("anchors", "positives", "negatives")}


def _successors(frame, pairs):
    successors = {}
    for history, target in zip(frame["history"], frame["target"]):
        for item in history:
            successors.setdefault(int(item), set()).add(int(target))
    anchors = np.asarray(pairs["anchors"], dtype=np.int64)
    positives = np.asarray(pairs["positives"], dtype=np.int64).reshape(len(anchors), -1)
    for anchor, targets in zip(anchors, positives):
        successors.setdefault(int(anchor), set()).update(map(int, targets))
    return successors


def _arm_dirs():
    roots = (ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE",
             ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE")
    arms = {"TIGER": roots[0], "Exp1a": roots[1] / "resample"}
    for parent in (roots[1],):
        for sid in sorted(parent.glob("*/item_sids.json")):
            label = "Exp1a" if sid.parent.name == "resample" else sid.parent.name
            arms[label] = sid.parent
    priority = ("Exp1a", "TIGER", "crowd_couser_w1", "crowd_couser_w2", "crowd_couser_w3",
                "crowd_random_w2", "crowd_random_w3", "crowd_scale_w1", "crowd_scale_w2")
    rank = {label: index for index, label in enumerate(priority)}
    return dict(sorted(arms.items(), key=lambda pair: (rank.get(pair[0], len(rank)), pair[0])))

def _load_compatible_checkpoint(label, arm_dir, checkpoint_override=None):
    import torch
    utility_path = ROOT / "analysis/crowding_diagnostic/residuals.py"
    spec = importlib.util.spec_from_file_location("competition_residual_extractor", utility_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = module._trainer_module()
    model_module = importlib.import_module("model")
    model_class = model_module.RQVAE
    expected_model = model_class(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(expected_model)
    expected_keys = set(expected_model.state_dict())
    checkpoint_path = Path(checkpoint_override) if checkpoint_override else arm_dir / "rqvae_best.pth"
    mirror_path = Path("/tmp/ckpt_mirror") / arm_dir.name
    if mirror_path.is_file():
        checkpoint_path = mirror_path
    print(f"resolved_checkpoint_path arm={label} path={checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    dropped_keys = sorted(key for key in state_dict
                          if key.startswith("crowding_control.") and key not in expected_keys)
    dropped_set = set(dropped_keys)
    filtered_state = {key: value for key, value in state_dict.items() if key not in dropped_set}
    missing_keys = sorted(expected_keys.difference(filtered_state))
    if missing_keys:
        raise RuntimeError(f"{label}: missing RQVAE model keys after known-extra filtering: {missing_keys}")
    checked = expected_model.load_state_dict(filtered_state, strict=True)
    if checked.missing_keys:
        raise RuntimeError(f"{label}: strict checkpoint load returned missing keys: {checked.missing_keys}")
    if checked.unexpected_keys:
        raise RuntimeError(f"{label}: strict checkpoint load returned unexpected keys: {checked.unexpected_keys}")
    return module, torch, model_class, checkpoint, filtered_state, dropped_keys, checkpoint_path


def _extract(label, arm_dir, output_dir=None, force=False, checkpoint_override=None):
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT / "residuals"
    output = output_dir / f"residuals_{label}.npz"
    if output.exists() and not force:
        print(f"loader_compatibility arm={label} checkpoint_validation=skipped_cached_residual")
        return output, None
    module, torch, model_class, checkpoint, filtered_state, dropped_keys, checkpoint_path = (
        _load_compatible_checkpoint(label, arm_dir, checkpoint_override))
    original_torch_load = torch.load
    original_load_state_dict = model_class.load_state_dict

    def load_compatible(path, *args, **kwargs):
        try:
            same_checkpoint = Path(path).resolve() == checkpoint_path.resolve()
        except TypeError:
            same_checkpoint = False
        return checkpoint | {"state_dict": filtered_state} if same_checkpoint else original_torch_load(
            path, *args, **kwargs)

    def load_strict(self, state_dict, strict=True, **kwargs):
        kwargs.pop("strict", None)
        result = original_load_state_dict(self, state_dict, strict=True, **kwargs)
        if result.missing_keys:
            raise RuntimeError(f"{label}: strict model load returned missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            raise RuntimeError(f"{label}: strict model load returned unexpected keys: {result.unexpected_keys}")
        return result

    module.OUT_DIR = output_dir
    sid_json = arm_dir / "item_sids.json"
    module.ARMS = {label: {"checkpoint": checkpoint_path, "sid_json": sid_json,
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}}
    module._trainer_module()
    torch.load = load_compatible
    model_class.load_state_dict = load_strict
    try:
        module.extract(label)
    finally:
        torch.load = original_torch_load
        model_class.load_state_dict = original_load_state_dict
    print(f"loader_compatibility arm={label} dropped_crowding_control_keys={len(dropped_keys)} missing_keys=0")
    return output, dropped_keys


def _load_levels(label, residual_path):
    with np.load(residual_path) as f:
        item_ids = f["item_ids"].astype(np.int64)
        residuals = f["residuals"]
        codes = f["codes"].astype(np.int64)
    if residuals.shape[1] != 3:
        raise ValueError(f"{label}: expected three residual levels, got {residuals.shape}")
    order = np.argsort(item_ids)
    if not np.array_equal(item_ids[order], np.arange(len(item_ids))):
        raise ValueError(f"{label}: item IDs are not contiguous item-index coordinates")
    vectors = {f"L{i + 1}": residuals[order, i] for i in range(3)}
    levels = codes[order]
    return item_ids[order], vectors, levels


def _reference_ids(n_items):
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_items, size=min(REFERENCE_SAMPLE_SIZE, n_items), replace=False))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else "undefined"
    if isinstance(value, float) and not np.isfinite(value):
        return "undefined"
    return value



def _metrics_for_arm(label, item_ids, vector_spaces, levels, pairs, successors, ref_ids, knn_workers=-1):
    from eval_behavior_metrics import metrics_for
    rows = []
    for name, vectors in vector_spaces.items():
        neighbors = C._knn(vectors, (5, 10, 20, 50), workers=knn_workers)
        measured = C.measure_level(vectors, pairs, ref_ids, epsilons=EPSILONS)
        neighborhood = C.neighborhood_metrics(vectors, pairs, k_values=(5, 10, 20, 50),
                                              successor_lists=successors, precomputed_neighbors=neighbors)
        # Margins include sampled reference distances only for the observed pair anchors.
        anchors = np.asarray(pairs["anchors"], dtype=np.int64)
        neg = np.asarray(pairs["negatives"], dtype=np.int64)
        positives = np.asarray(pairs["positives"], dtype=np.int64)
        if positives.ndim == 1:
            positives = positives[:, None]
        successor_counts = np.array([len(successors.get(int(a), ())) for a in anchors])
        multiplicity_rows = {}
        for group, selected in C.multiplicity_groups(successor_counts).items():
            sub = {"anchors": anchors[selected], "positives": positives[selected], "negatives": neg[selected]}
            part = C.measure_level(vectors, sub, ref_ids, epsilons=EPSILONS,
                                   local_scales=measured["local_scales"][selected])
            multiplicity_rows[group] = {
                "anchor_count": int(len(np.unique(anchors[selected]))), "triple_count": part["triple_count"],
                "pncr": part["pncr"], "error_rate_margin": part["error_rate_margin"],
                "behavioral_auc": part["behavioral_auc"],
                "margin_distribution": part["margin_distribution"],
                "tightest_negative_margin": part["tightest_negative_margin"],
                "loosest_negative_margin": part["loosest_negative_margin"],
                "neighborhood": C.neighborhood_metrics(vectors, sub, k_values=(5, 10, 20, 50),
                                                       successor_lists=successors, precomputed_neighbors=neighbors),
                "positive_distance": C._distribution(part["positive_distances"]),
                "negative_distance": C._distribution(part["negative_distances"]),
                "positive_distance_over_local_scale": C._distribution(part["positive_distances"] / part["local_scales"]),
                "negative_distance_over_local_scale": C._distribution(part["negative_distances"] / part["local_scales"][:, None]),
            }
        code_metrics = metrics_for(levels, pairs)
        prefix = {key: code_metrics[key] for key in ("L1", "L1_L2", "L1_L2_L3", "L2_given_L1", "L3_given_L2", "codebook")}
        prefix["sample_counts"] = {"positive_pairs": int(len(anchors)), "negative_pairs": int(len(neg) * neg.shape[1]),
                                   "unique_anchors": int(len(np.unique(anchors))), "items": int(len(levels))}
        for key in ("L1", "L1_L2", "L1_L2_L3"):
            prefix[key]["n_positive_pairs"] = len(anchors)
            prefix[key]["n_negative_pairs"] = len(neg) * neg.shape[1]
        for value in prefix["codebook"].values():
            value["n_items"] = len(levels)
        rows.append({"arm": label, "level": name,
                     "triples": {key: value for key, value in measured.items()
                                 if key not in ("margins", "local_scales", "positive_distances", "negative_distances")},
                     "positive_distance": C._distribution(measured["positive_distances"]),
                     "negative_distance": C._distribution(measured["negative_distances"]),
                     "local_scale": C._distribution(measured["local_scales"]),
                     "positive_distance_over_local_scale": C._distribution(measured["positive_distances"] / measured["local_scales"]),
                     "negative_distance_over_local_scale": C._distribution(measured["negative_distances"] / measured["local_scales"][:, None]),
                     "neighborhood": neighborhood,
                     "successor_multiplicity": C._distribution(
                         [len(successors.get(int(a), ())) for a in np.unique(anchors)]),
                     "multiplicity_strata": multiplicity_rows,
                     "successor_neighborhood_overlap": _successor_overlap(vectors, successors, anchors, neighbors),
                     "prefix_sharing_and_er": prefix})
    return rows

def _warm_checkpoint(task):
    label, arm_dir = task
    _, _, _, _, _, dropped_keys, _ = _load_compatible_checkpoint(label, arm_dir)
    return label, len(dropped_keys)


def _process_arm(task):
    label, arm_dir, knn_workers, preserve_existing, staged_checkpoint = task
    residual_file, dropped_keys = _extract(label, arm_dir, checkpoint_override=staged_checkpoint)
    compatibility = ({"checkpoint_validation": "skipped_cached_residual", "dropped_key_count": None,
                      "dropped_keys": None, "missing_keys": None}
                     if dropped_keys is None else
                     {"checkpoint_validation": "strict_loaded", "dropped_key_count": len(dropped_keys),
                      "dropped_keys": dropped_keys, "missing_keys": []})
    if preserve_existing:
        rows = []
    else:
        pairs = _load_pairs()
        frame = pd.read_parquet(VALID_PATH)
        successors = _successors(frame, pairs)
        embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        ref_ids = _reference_ids(len(embeddings))
        ids, spaces, levels = _load_levels(label, residual_file)
        if len(ids) != len(embeddings):
            raise ValueError(f"{label}: item count differs from embedding space")
        rows = _metrics_for_arm(label, ids, spaces, levels, pairs, successors, ref_ids,
                                knn_workers=knn_workers)
    worker_dir = OUTPUT / "worker_rows"
    worker_dir.mkdir(parents=True, exist_ok=True)
    row_path = worker_dir / f"{label}.json"
    row_path.write_text(json.dumps({"rows": _jsonable(rows), "loader_compatibility": compatibility},
                                   allow_nan=False) + "\n")
    return label, str(row_path)


def _successor_overlap_reference(successors, anchors, precomputed_neighbors):
    neighborhoods = {item: set(map(int, row[:10])) for item, row in enumerate(precomputed_neighbors)}
    overlap_anchors, overlap_by_anchor = [], []
    for anchor in np.unique(anchors):
        succ = sorted(successors.get(int(anchor), ()))
        if len(succ) < 2:
            continue
        pair_scores = []
        for left in range(len(succ)):
            left_neighbors = neighborhoods[succ[left]]
            for right in range(left + 1, len(succ)):
                right_neighbors = neighborhoods[succ[right]]
                union = left_neighbors | right_neighbors
                pair_scores.append(len(left_neighbors & right_neighbors) / len(union) if union else 0.0)
        overlap_anchors.append(int(anchor))
        overlap_by_anchor.append(float(np.mean(pair_scores)))
    return np.asarray(overlap_anchors, dtype=np.int64), np.asarray(overlap_by_anchor, dtype=np.float64)


def _successor_pair_index(successors, anchors):
    group_anchors = np.unique(anchors)
    groups = [np.asarray(sorted(successors.get(int(anchor), ())), dtype=np.int64)
              for anchor in group_anchors]
    counts = np.fromiter((len(group) for group in groups), dtype=np.int64, count=len(groups))
    keep = counts >= 2
    group_anchors = group_anchors[keep]
    counts = counts[keep]
    groups = [group for group, use in zip(groups, keep) if use]
    if not len(groups):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    flat_items = np.concatenate(groups)
    successor_group_ids = np.repeat(np.arange(len(groups), dtype=np.int64), counts)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    left_parts, right_parts, group_parts = [], [], []
    for size in np.unique(counts):
        group_rows = np.flatnonzero(counts == size)
        triangle_left, triangle_right = np.triu_indices(int(size), k=1)
        offsets = starts[group_rows, None]
        left_positions = (offsets + triangle_left[None, :]).reshape(-1)
        right_positions = (offsets + triangle_right[None, :]).reshape(-1)
        left_parts.append(left_positions)
        right_parts.append(right_positions)
        group_parts.append(np.repeat(group_rows, len(triangle_left)))
    left_positions = np.concatenate(left_parts)
    right_positions = np.concatenate(right_parts)
    pair_groups = np.concatenate(group_parts)
    order = np.argsort(pair_groups, kind="stable")
    left_positions = left_positions[order]
    right_positions = right_positions[order]
    pair_groups = successor_group_ids[left_positions]
    return group_anchors[pair_groups], flat_items[left_positions], flat_items[right_positions]


def _successor_overlap_values(successors, anchors, precomputed_neighbors):
    neighborhoods = np.asarray(precomputed_neighbors, dtype=np.int64)[:, :10]
    pair_anchors, left_items, right_items = _successor_pair_index(successors, anchors)
    if not len(pair_anchors):
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=np.float64)
    scores = np.empty(len(pair_anchors), dtype=np.float64)
    chunk_size = 200_000
    for start in range(0, len(pair_anchors), chunk_size):
        stop = min(start + chunk_size, len(pair_anchors))
        left = neighborhoods[left_items[start:stop]]
        right = neighborhoods[right_items[start:stop]]
        intersection = np.count_nonzero(left[:, :, None] == right[:, None, :], axis=(1, 2))
        union = 20 - intersection
        scores[start:stop] = np.divide(intersection, union, out=np.zeros(len(intersection), dtype=np.float64),
                                       where=union != 0)
    group_starts = np.r_[0, np.flatnonzero(pair_anchors[1:] != pair_anchors[:-1]) + 1]
    group_counts = np.diff(np.r_[group_starts, len(pair_anchors)])
    per_anchor = np.empty(len(group_starts), dtype=np.float64)
    for pair_count in np.unique(group_counts):
        group_rows = np.flatnonzero(group_counts == pair_count)
        positions = group_starts[group_rows, None] + np.arange(pair_count, dtype=np.int64)[None, :]
        per_anchor[group_rows] = scores[positions].mean(axis=1)
    return pair_anchors[group_starts], per_anchor


def _successor_overlap(vectors, successors, anchors, precomputed_neighbors):
    overlap_anchors, overlap_values = _successor_overlap_reference(successors, anchors, precomputed_neighbors)
    return {"mean_pairwise_successor_neighborhood_jaccard_at_10": C._distribution(overlap_values)}


def _verify_overlap_vectorization(prior_rows, pairs, successors, reference_count, arm_count):
    _, vector_spaces, _ = _load_levels("Exp1a", OUTPUT / "residuals" / "residuals_Exp1a.npz")
    vectors = vector_spaces["L1"]
    anchors = np.asarray(pairs["anchors"], dtype=np.int64)
    neighbors = C._knn(vectors, (5, 10, 20, 50), workers=-1)
    sample_anchors = np.unique(anchors)[:2000]
    sample_ids, sample_values = _successor_overlap_reference(successors, sample_anchors, neighbors)
    if len(sample_ids) != len(sample_values):
        raise RuntimeError("Original successor overlap loop returned inconsistent sample arrays")
    _write_incremental(prior_rows, arm_count, len(anchors), reference_count)
    with (OUTPUT / "competition_metrics.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] == "Exp1a" and row["level"] == "L1"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one saved Exp1a L1 overlap row, found {len(rows)}")
    saved_value = float(rows[0]["successor_neighborhood_jaccard_mean"])
    current = C._distribution(sample_values)
    print(f"successor_overlap_original_loop_sample_anchors={len(sample_ids)} mean={current['mean']:.17g}")
    if not np.isfinite(saved_value):
        raise RuntimeError("Saved Exp1a overlap is not finite")

def _decomposition_statement(label, rows):
    def average(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
        return float(np.mean(values)) if values else "undefined"

    closeness = average(("triples", "pncr", "value"))
    intrusion = average(("neighborhood", "hard_negative_intrusion", "value"))
    p_mean = average(("positive_distance", "mean"))
    p_median = average(("positive_distance", "median"))
    n_mean = average(("negative_distance", "mean"))
    n_median = average(("negative_distance", "median"))
    p_ratio = average(("positive_distance_over_local_scale", "mean"))
    n_ratio = average(("negative_distance_over_local_scale", "mean"))
    return (f"{label}: closeness PNCR@0.10={closeness:.6g}; intrusion={intrusion:.6g}; "
            f"compression positive mean/median={p_mean:.6g}/{p_median:.6g}, "
            f"negative mean/median={n_mean:.6g}/{n_median:.6g}, "
            f"positive/negative mean distance-to-local-scale={p_ratio:.6g}/{n_ratio:.6g}")


def _verify_loader_wrapper(arm_dir):
    baseline_path = OUTPUT / "residuals" / "residuals_Exp1a.npz"
    check_dir = OUTPUT / "loader_wrapper_check"
    check_path, dropped_keys = _extract("Exp1a", arm_dir, output_dir=check_dir, force=True)
    try:
        with np.load(baseline_path) as baseline, np.load(check_path) as checked:
            compared = ("item_ids", "residuals", "codes", "cumulative_reconstruction", "sid",
                        "checkpoint_sha256")
            for key in compared:
                if not np.array_equal(baseline[key], checked[key]):
                    raise RuntimeError(f"Loader wrapper changed Exp1a {key}")
    finally:
        if check_path.exists():
            check_path.unlink()
        if check_dir.exists():
            check_dir.rmdir()
    if dropped_keys:
        raise RuntimeError("Exp1a unexpectedly contained crowding_control keys")
    print("loader_wrapper_bit_identical=Exp1a arrays=item_ids,residuals,codes,cumulative_reconstruction,sid,checkpoint_sha256")




def _write_incremental(rows, arm_count, pair_count, reference_count, loader_compatibility=None, quarantined=None):
    result = {"protocol": {"pair_split": "validation", "seed": SEED, "reference_sample_size": reference_count,
                          "local_scale": "per-anchor median distance to fixed reference sample, excluding self",
                          "pair_count": pair_count, "arm_count": arm_count, "rows": len(rows)},
              "loader_compatibility": loader_compatibility or {},
              "quarantined_arms": quarantined or [], "rows": _jsonable(rows)}
    (OUTPUT / "competition_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ("arm", "level", "n_triples", "pncr_epsilon_0.05", "pncr_epsilon_0.10", "pncr_epsilon_0.20",
              "error_rate_margin", "behavioral_auc", "intrusion", "positive_mean_distance", "negative_mean_distance",
              "successor_neighborhood_jaccard_mean", "successor_neighborhood_jaccard_n")
    with (OUTPUT / "competition_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            triple = row["triples"]
            writer.writerow({
                "arm": row["arm"], "level": row["level"], "n_triples": triple["triple_count"],
                **{f"pncr_epsilon_{epsilon:.2f}": triple["pncr_by_epsilon"][str(epsilon)]["value"]
                   for epsilon in EPSILONS},
                "error_rate_margin": triple["error_rate_margin"]["value"],
                "behavioral_auc": triple["behavioral_auc"]["value"],
                "intrusion": row["neighborhood"]["hard_negative_intrusion"]["value"],
                "positive_mean_distance": row["positive_distance"]["mean"],
                "negative_mean_distance": row["negative_distance"]["mean"],
                "successor_neighborhood_jaccard_mean": row["successor_neighborhood_overlap"]["mean_pairwise_successor_neighborhood_jaccard_at_10"]["mean"],
                "successor_neighborhood_jaccard_n": row["successor_neighborhood_overlap"]["mean_pairwise_successor_neighborhood_jaccard_at_10"]["n"]})


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = _load_pairs()
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    n_items = len(embeddings)
    ref_ids = _reference_ids(n_items)
    arms = _arm_dirs()
    previous_file = OUTPUT / "competition_metrics.json"
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else {}
    prior_rows = previous.get("rows", [])
    prior_labels = {row["arm"] for row in prior_rows}
    completed = {label: [row for row in prior_rows if row["arm"] == label] for label in prior_labels}
    all_rows = [row for label in arms for row in completed.get(label, ())]
    _write_incremental(all_rows, len(arms), len(pairs["anchors"]), len(ref_ids))


    compatibility = {}
    quarantined = []
    missing_labels = [label for label in arms
                      if not (OUTPUT / "residuals" / f"residuals_{label}.npz").exists()]
    warm_label = missing_labels[0] if missing_labels else None
    if warm_label:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as warm_pool:
            warm_label, warm_dropped_keys = warm_pool.submit(
                _warm_checkpoint, (warm_label, arms[warm_label])).result()
        print(f"checkpoint_warmup arm={warm_label} dropped_crowding_control_keys={warm_dropped_keys} missing_keys=0")

    with ProcessPoolExecutor(max_workers=8, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_process_arm, (label, arm_dir, -1, label in prior_labels, None)): label
                   for label, arm_dir in arms.items()}
        for future in as_completed(futures):
            label = futures[future]
            try:
                result_label, row_path = future.result()
                payload = json.loads(Path(row_path).read_text())
                compatibility[result_label] = payload["loader_compatibility"]
                if payload["loader_compatibility"]["missing_keys"]:
                    raise RuntimeError(f"Nonempty missing-key list: {payload['loader_compatibility']['missing_keys']}")
                if payload["rows"]:
                    completed[result_label] = payload["rows"]
            except Exception as exc:
                completed[label] = []
                quarantine = {"arm": label, "reason": str(exc)}
                quarantined.append(quarantine)
                print(f"arm_quarantined={label} reason={exc}")
            all_rows = [row for arm in arms for row in completed.get(arm, ())]
            _write_incremental(all_rows, len(arms), len(pairs["anchors"]), len(ref_ids),
                               compatibility, quarantined)
            if label not in {entry["arm"] for entry in quarantined}:
                print(f"completed_arm={label} rows={len(all_rows)}")
    worker_dir = OUTPUT / "worker_rows"
    if worker_dir.exists():
        for path in worker_dir.glob("*.json"):
            path.unlink()
        worker_dir.rmdir()
    result = {"protocol": {"pair_file": str(PAIRS_PATH), "pair_split": "validation", "seed": SEED,
                            "reference_sample_size": len(ref_ids), "reference_sampling": "uniform without replacement, RNG seed 42; same item sample across arms",
                            "local_scale": "per-anchor median Euclidean distance from the anchor to the fixed 512-item reference sample, excluding self",
                            "epsilon_values": list(EPSILONS), "primary_PNCR_epsilon": EPSILONS[1],
                            "distance": "Euclidean residual-space distance; AUC uses crowding_diagnostic.distances.auc_neg_farther_than_pos",
                            "arm_count": len(arms), "rows": len(all_rows)},
              "loader_compatibility": compatibility, "quarantined_arms": quarantined,
              "rows": _jsonable(all_rows)}
    out_json = OUTPUT / "competition_metrics.json"
    out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = ["arm", "level", "n_triples"]
    for epsilon in EPSILONS:
        columns.extend((f"pncr_epsilon_{epsilon:.2f}", f"pncr_epsilon_{epsilon:.2f}_n"))
    columns.extend(("error_rate_margin", "error_rate_margin_n", "error_count", "behavioral_auc", "behavioral_auc_n",
                    "intrusion", "intrusion_n", "tightest_margin_median", "tightest_margin_p05", "tightest_margin_p95",
                    "tightest_margin_n", "loosest_margin_median", "loosest_margin_p05", "loosest_margin_p95",
                    "loosest_margin_n", "positive_mean_distance", "positive_median_distance", "positive_distance_n",
                    "negative_mean_distance", "negative_median_distance", "negative_distance_n",
                    "local_scale_mean", "local_scale_median", "local_scale_n", "positive_mean_distance_over_local_scale",
                    "positive_median_distance_over_local_scale", "positive_ratio_n", "negative_mean_distance_over_local_scale",
                    "negative_median_distance_over_local_scale", "negative_ratio_n"))
    for k in (5, 10, 20, 50):
        columns.extend((f"positive_coverage_at_{k}", f"positive_coverage_at_{k}_n",
                        f"negative_coverage_at_{k}", f"negative_coverage_at_{k}_n"))
    for prefix in ("L1", "L1_L2", "L1_L2_L3"):
        columns.extend((f"{prefix}_p_share_pos", f"{prefix}_p_share_pos_n",
                        f"{prefix}_p_share_neg", f"{prefix}_p_share_neg_n", f"{prefix}_er"))
    columns.extend(("multiplicity_1_anchors", "multiplicity_1_triples", "multiplicity_1_pncr",
                    "multiplicity_1_error_rate", "multiplicity_1_auc", "multiplicity_2plus_anchors",
                    "multiplicity_2plus_triples", "multiplicity_2plus_pncr", "multiplicity_2plus_error_rate",
                    "multiplicity_2plus_auc", "successor_multiplicity_mean", "successor_multiplicity_n",
                    "successor_neighborhood_jaccard_mean", "successor_neighborhood_jaccard_n"))
    with (OUTPUT / "competition_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["rows"]:
            tri = row["triples"]
            output = {"arm": row["arm"], "level": row["level"], "n_triples": tri["triple_count"],
                      "error_rate_margin": tri["error_rate_margin"]["value"],
                      "error_rate_margin_n": tri["error_rate_margin"]["n"], "error_count": tri["error_count"],
                      "behavioral_auc": tri["behavioral_auc"]["value"], "behavioral_auc_n": tri["behavioral_auc"]["n"],
                      "intrusion": row["neighborhood"]["hard_negative_intrusion"]["value"],
                      "intrusion_n": row["neighborhood"]["hard_negative_intrusion"]["n"],
                      "successor_multiplicity_mean": row["successor_multiplicity"]["mean"],
                      "successor_multiplicity_n": row["successor_multiplicity"]["n"],
                      "successor_neighborhood_jaccard_mean": row["successor_neighborhood_overlap"]["mean_pairwise_successor_neighborhood_jaccard_at_10"]["mean"],
                      "successor_neighborhood_jaccard_n": row["successor_neighborhood_overlap"]["mean_pairwise_successor_neighborhood_jaccard_at_10"]["n"]}
            for epsilon in EPSILONS:
                stat = tri["pncr_by_epsilon"][str(epsilon)]
                output[f"pncr_epsilon_{epsilon:.2f}"] = stat["value"]
                output[f"pncr_epsilon_{epsilon:.2f}_n"] = stat["n"]
            for key, name in (("tightest_negative_margin", "tightest_margin"),
                              ("loosest_negative_margin", "loosest_margin")):
                distribution = tri[key]
                output.update({f"{name}_median": distribution["median"], f"{name}_p05": distribution["p05"],
                               f"{name}_p95": distribution["p95"], f"{name}_n": distribution["n"]})
            distributions = (
                ("positive_distance", "positive_mean_distance", "positive_median_distance", "positive_distance_n"),
                ("negative_distance", "negative_mean_distance", "negative_median_distance", "negative_distance_n"),
                ("local_scale", "local_scale_mean", "local_scale_median", "local_scale_n"),
                ("positive_distance_over_local_scale", "positive_mean_distance_over_local_scale",
                 "positive_median_distance_over_local_scale", "positive_ratio_n"),
                ("negative_distance_over_local_scale", "negative_mean_distance_over_local_scale",
                 "negative_median_distance_over_local_scale", "negative_ratio_n"))
            for key, mean_name, median_name, n_name in distributions:
                distribution = row[key]
                output[mean_name] = distribution["mean"]
                output[median_name] = distribution["median"]
                output[n_name] = distribution["n"]
            for k in (5, 10, 20, 50):
                for key, name in (("positive_coverage_at_K", "positive"), ("matched_negative_coverage_at_K", "negative")):
                    stat = row["neighborhood"][key][str(k)]
                    output[f"{name}_coverage_at_{k}"] = stat["value"]
                    output[f"{name}_coverage_at_{k}_n"] = stat["n"]
            for prefix in ("L1", "L1_L2", "L1_L2_L3"):
                for metric in ("p_share_pos", "p_share_neg"):
                    output[f"{prefix}_{metric}"] = row["prefix_sharing_and_er"][prefix][metric]
                    output[f"{prefix}_{metric}_n"] = (row["prefix_sharing_and_er"]["sample_counts"]["positive_pairs"]
                                                       if metric.endswith("pos") else
                                                       row["prefix_sharing_and_er"]["sample_counts"]["negative_pairs"])
                output[f"{prefix}_er"] = row["prefix_sharing_and_er"][prefix]["er"]
            for group, label in (("1", "1"), ("2+", "2plus")):
                values = row["multiplicity_strata"][group]
                output.update({f"multiplicity_{label}_anchors": values["anchor_count"],
                               f"multiplicity_{label}_triples": values["triple_count"],
                               f"multiplicity_{label}_pncr": values["pncr"]["value"],
                               f"multiplicity_{label}_error_rate": values["error_rate_margin"]["value"],
                               f"multiplicity_{label}_auc": values["behavioral_auc"]["value"]})
            writer.writerow(output)
    print(f"arms={len(arms)} rows={len(all_rows)} pairs={len(pairs['anchors'])}")
    print(f"local_scale=median distance to fixed {len(ref_ids)}-item random sample, seed={SEED}")
    print(f"table={OUTPUT / 'competition_metrics.csv'}")
    print(f"full_metrics={out_json}")
    with (OUTPUT / "competition_metrics.csv").open() as stream:
        print(stream.read(), end="")
    labels = ("Exp1a", "TIGER")
    for label in labels:
        print(_decomposition_statement(label, [row for row in result["rows"] if row["arm"] == label]))
    random_rows = [row for row in result["rows"] if row["arm"].startswith("crowd_random")]
    print(_decomposition_statement("crowd_random control (mean across available arms/levels)", random_rows))


if __name__ == "__main__":
    run()
