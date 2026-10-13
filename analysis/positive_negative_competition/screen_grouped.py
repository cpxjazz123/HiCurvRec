"""Measure capacity arms with the fixed direct-competition suite and stratifications."""
from __future__ import annotations

import json
import shutil
import sys
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
import competition_grouped as G
import screen_competition as S

ARMS_ROOT = ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
ARM_LABELS = ("cap_d8", "cap_d16", "cap_d64", "resample")
CAPACITY = ROOT / "results/positive_negative_competition/capacity"
RESIDUALS = CAPACITY / "residuals"
CACHED_RESIDUALS = ROOT / "results/positive_negative_competition/residuals"
EMBEDDINGS = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
VALID = ROOT / "results/stage0_build_parquet/valid.parquet"
MINIMUM = G.MINIMUM_STRATUM_SIZE
K_VALUES = (5, 10, 20, 50)


def _popularity(frame, item_count):
    counts = np.zeros(item_count, dtype=np.int64)
    for history in frame["history"]:
        ids, frequencies = np.unique(np.asarray(history, dtype=np.int64), return_counts=True)
        counts[ids] += frequencies
    return counts


def _codebook_size(label):
    return 256


def _group_full_metrics(vectors, pairs, ref_ids, mask, successors, neighbors, codes, min_size,
                        local_scales):
    anchors, positives, negatives = C._indices(pairs)
    nrows = int(np.sum(mask))
    result = {"pair_rows": nrows, "anchor_count": int(len(np.unique(anchors[mask]))),
              "triple_count": int(nrows * negatives.shape[1]), "minimum_pair_rows": int(min_size)}
    if nrows < min_size:
        result["undefined_reason"] = f"pair_rows<{min_size}"
        result["metrics"] = "undefined"
        result["local_density"] = "undefined"
        result["code_metrics"] = "undefined"
        result["prefix_sharing_and_er"] = "undefined"
        return result
    sub = {"anchors": anchors[mask], "positives": positives[mask], "negatives": negatives[mask]}
    measured = C.measure_level(vectors, sub, ref_ids, local_scales=local_scales[mask])
    measured["hard_competition_rate"] = G.hard_competition_rate(measured["margins"])
    result["metrics"] = {k: v for k, v in measured.items()
                         if k not in ("margins", "local_scales", "positive_distances", "negative_distances")}
    result["neighborhood"] = C.neighborhood_metrics(vectors, sub, k_values=K_VALUES,
                                                      successor_lists=successors,
                                                      precomputed_neighbors=neighbors)
    item_ids = np.unique(np.concatenate((anchors[mask], positives[mask].ravel(), negatives[mask].ravel())))
    result["code_metrics"] = G.code_metrics(codes, item_ids, _codebook_size(""))
    from eval_behavior_metrics import metrics_for
    prefix_pairs = {**sub, "positives": sub["positives"][:, 0]}
    pmetrics = metrics_for(codes, prefix_pairs)
    return result


def _capacity_metadata(label, codes):
    return {"arm": label, "codebook_size_per_level": _codebook_size(label),
            "codebook_dimension": {"cap_d8": 8, "cap_d16": 16, "cap_d64": 64,
                                   "resample": 32}[label],
            "items": int(len(codes)), "levels": 3}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else "undefined"
    return value


def _flatten_rows(results):
    output = []
    for arm in results:
        for level, row in arm["levels"].items():
            base = row["overall"]
            output.append({"arm": arm["arm"], "level": level, "stratification": "overall", "stratum": "all",
                           "pair_rows": base["pair_rows"], **_headline(base)})
            for axis, groups in row["strata"].items():
                for name, values in groups.items():
                    output.append({"arm": arm["arm"], "level": level, "stratification": axis,
                                   "stratum": name, "pair_rows": values["pair_rows"], **_headline(values)})
    return output


def _headline(row):
    m = row.get("metrics")
    if m == "undefined":
        return {"status": "undefined"}
    neighborhood = row.get("neighborhood", {})
    triples = m
    return {"status": "measured", "pncr_005": triples["pncr_by_epsilon"]["0.05"],
            "pncr_010": triples["pncr_by_epsilon"]["0.1"], "pncr_020": triples["pncr_by_epsilon"]["0.2"],
            "error_rate_m_lt_0": triples["error_rate_margin"],
            "hard_competition_m_lt_01": triples["hard_competition_rate"],
            "auc": triples["behavioral_auc"], "tightest_margin": triples["tightest_negative_margin"],
            "loosest_margin": triples["loosest_negative_margin"],
            "intrusion": neighborhood.get("hard_negative_intrusion", "undefined"),
            "positive_coverage_at_5_10_20": neighborhood.get("positive_coverage_at_K", {}),
            "matched_negative_coverage_at_5_10_20": neighborhood.get("matched_negative_coverage_at_K", {}),
            "knn_density": row.get("local_density", "undefined"),
            "code_metrics": row.get("code_metrics", "undefined"),
            "prefix_sharing_and_er": row.get("prefix_sharing_and_er", "undefined")}


def _stage_and_extract(label, arm_dir):
    cached_label = "Exp1a" if label == "resample" else label
    cached = CACHED_RESIDUALS / f"residuals_{cached_label}.npz"
    if cached.is_file():
        return cached
    RESIDUALS.mkdir(parents=True, exist_ok=True)
    output = RESIDUALS / f"residuals_{label}.npz"
    if output.is_file():
        return output
    import hashlib
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location(
        "capacity_residual_extractor", ROOT / "analysis/crowding_diagnostic/residuals.py")
    residuals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(residuals)
    trainer = residuals._trainer_module()
    from model import RQVAE
    checkpoint = arm_dir / "rqvae_best.pth"
    staged_dir = Path("/tmp/positive_negative_competition_capacity") / label
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / "rqvae_best.pth"
    if not staged.exists():
        shutil.copy2(checkpoint, staged)
    arm_config = trainer._CAPACITY_ARMS[label]
    trainer.CODEBOOK_DIM = int(arm_config["codebook_dim"])
    trainer.CODEBOOK_SIZE = tuple(arm_config["codebook_size"])
    trainer.CODEBOOK_NUM = len(trainer.CODEBOOK_SIZE)
    trainer.HIDDEN_SIZES = tuple(arm_config["hidden"])
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    blob = torch.load(staged, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    ids = np.array(sorted(map(int, json.loads((arm_dir / "item_sids.json").read_text()).keys())),
                   dtype=np.int64)
    selected = np.asarray(embeddings if embeddings.shape[0] == len(ids) else embeddings[ids],
                          dtype=np.float32)
    trace = residuals.residual_trace_from_model(model, selected, trainer)
    with torch.no_grad():
        tensor = torch.tensor(selected, dtype=torch.float32)
        encoded = model.encoder(tensor)
        _, _, _, model_codes = model.rq(encoded)
        codes = model_codes.cpu().numpy().astype(np.int64)
        reconstruction = np.zeros_like(encoded.cpu().numpy())
        residual = encoded
        for layer in model.rq.vq_layers:
            quantized, _, _, _ = layer(residual)
            reconstruction += quantized.cpu().numpy()
            residual = residual - quantized
    sid = trainer._extend_collisions(codes, list(model.config.codebook_size))
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    np.savez_compressed(output, item_ids=ids, residuals=trace.astype(np.float32), codes=codes,
                        cumulative_reconstruction=reconstruction.astype(np.float32), sid=sid,
                        checkpoint_sha256=np.asarray(digest))
    return output


def _persist_results(results):
    payload = {"protocol": {"pair_split": "validation", "seed": 42, "reference_sample_size": 512,
                            "minimum_stratum_pair_rows": MINIMUM,
                            "hard_competition_definition": "fraction of matched-pair margins m < 0.1",
                            "hardness_definition": "anchor's minimum matched-negative distance divided by its positive distance; rank terciles",
                            "popularity_definition": "item-occurrence count in validation histories; stable item-level rank deciles",
                            "successor_definition": "distinct real validation successors, including canonical validation pairs"},
               "arms": results}
    output = CAPACITY / "capacity_grouped_metrics.json"
    output.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    pd.DataFrame(_flatten_rows(results)).to_csv(CAPACITY / "capacity_grouped_metrics.csv", index=False)


def run():
    CAPACITY.mkdir(parents=True, exist_ok=True)
    pairs = S._load_pairs()
    frame = pd.read_parquet(VALID)
    successors = S._successors(frame, pairs)
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    reference_ids = S._reference_ids(len(embeddings))
    pop_by_item = _popularity(frame, len(embeddings))
    popularity_deciles = G._rank_bins(pop_by_item, 10)
    anchors, positives, negatives = C._indices(pairs)
    successor_counts = np.asarray([len(successors.get(int(a), ())) for a in anchors], dtype=np.int64)
    results = []
    for label in ARM_LABELS:
        arm_dir = ARMS_ROOT / label
        residual_path = _stage_and_extract(label, arm_dir)
        ids, spaces, codes = S._load_levels(label, residual_path)
        if len(ids) != len(embeddings):
            raise ValueError(f"{label}: item count differs from embedding item universe")
        per_level = {}
        for level, vectors in spaces.items():
            print(f"measuring arm={label} level={level}", flush=True)
            neighbors = C._knn(vectors, K_VALUES, workers=-1)
            measured = C.measure_level(vectors, pairs, reference_ids)
            hard_relative = G._hardness_distance(vectors, pairs)
            strata = G.make_strata(pop_by_item[anchors], successor_counts, hard_relative,
                                   popularity_deciles=popularity_deciles[anchors])
            overall = _group_full_metrics(vectors, pairs, reference_ids, np.ones(len(anchors), dtype=bool),
                                          successors, neighbors, codes, minimum_for_all(), measured["local_scales"])
            grouped = {axis: {name: _group_full_metrics(vectors, pairs, reference_ids, mask, successors,
                                                        neighbors, codes, minimum_for_all(), measured["local_scales"])
                              for name, mask in groups.items()} for axis, groups in strata.items()}
            full = C.neighborhood_metrics(vectors, pairs, K_VALUES, successor_lists=successors,
                                          precomputed_neighbors=neighbors)
            overall["neighborhood"] = full
            overall["local_density"] = G.local_density(vectors, anchors, K_VALUES)
            overall["code_metrics"] = G.code_metrics(codes, np.arange(len(codes)), _codebook_size(label))
            from eval_behavior_metrics import metrics_for
            prefix = metrics_for(codes, pairs)
            overall["prefix_sharing_and_er"] = {k: prefix[k] for k in ("L1", "L1_L2", "L1_L2_L3", "L2_given_L1", "L3_given_L2")}
            overall["metrics"] = {k: v for k, v in measured.items()
                                   if k not in ("margins", "local_scales", "positive_distances", "negative_distances")}
            overall["metrics"]["hard_competition_rate"] = G.hard_competition_rate(measured["margins"])
            per_level[level] = {"overall": overall, "strata": grouped,
                                "all_item_code_metrics": G.code_metrics(codes, np.arange(len(codes)), _codebook_size(label)),
                                "local_scale": C._distribution(measured["local_scales"])}
        results.append({**_capacity_metadata(label, codes), "levels": per_level})
        _persist_results(results)
        print(f"completed arm={label}", flush=True)
    _persist_results(results)
    print(f"wrote {CAPACITY / 'capacity_grouped_metrics.json'}", flush=True)
    for arm in results:
        for level, row in arm["levels"].items():
            metric = row["overall"]["metrics"]
            print(f"reference arm={arm['arm']} level={level} PNCR@0.05={metric['pncr_by_epsilon']['0.05']['value']:.4f} "
                  f"error={metric['error_rate_margin']['value']:.4f} AUC={metric['behavioral_auc']['value']:.4f}", flush=True)


def minimum_for_all():
    return MINIMUM


if __name__ == "__main__":
    run()
