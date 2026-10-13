from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hyperbolic_utilisation"
ARMS = ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/positive_negative_competition"))
sys.path.insert(0, str(ROOT / "analysis/crowding_diagnostic"))
import competition as C
import competition_grouped as G
import screen_competition as S
import screen_grouped as SG
import measurements as M


def _latent(label, checkpoint_path, embeddings):
    module, torch, model_class, checkpoint, filtered, dropped, _ = S._load_compatible_checkpoint(
        label, ARMS / label, checkpoint_override=checkpoint_path)
    trainer = module._trainer_module()
    model = model_class(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    model.load_state_dict(filtered, strict=True)
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.as_tensor(np.asarray(embeddings), dtype=torch.float32))
    return z.detach().cpu().numpy(), int(len(dropped))


def _distances(a, p, n):
    return np.linalg.norm(a - p, axis=1), np.linalg.norm(a[:, None] - n, axis=2)


def _geo_distances(x, ids, pair):
    a, p, n = (x[pair[k]] for k in ("anchors", "positives", "negatives"))
    da, dn = _distances(a, p, n)
    return np.c_[da, dn[:, 0]], np.c_[da, dn[:, 0]]


def _metric_dists(x, pair, hyperbolic):
    a, p, n = (x[pair[k]] for k in ("anchors", "positives", "negatives"))
    pos = np.linalg.norm(a - p, axis=1)
    neg = np.linalg.norm(a[:, None] - n, axis=2)
    if hyperbolic:
        # Closed-form curvature-one Poincare geodesic; all points already expmap-mapped.
        def geo(u, v):
            diff2 = np.sum((u - v) ** 2, axis=-1)
            nu = np.sum(u * u, axis=-1)
            nv = np.sum(v * v, axis=-1)
            arg = 1 + 2 * diff2 / np.maximum((1 - nu) * (1 - nv), 1e-20)
            return np.arccosh(np.maximum(arg, 1.0))
        pos = geo(a, p)
        neg = geo(a[:, None], n)
    return np.c_[pos, neg[:, 0]], np.c_[pos, neg[:, 0]]


def _competition(vectors, pairs, refids, strata):
    measured = C.measure_level(vectors, pairs, refids)
    measured["hard_competition_rate"] = G.hard_competition_rate(measured["margins"])
    neighborhood = C.neighborhood_metrics(vectors, pairs, k_values=(5, 10), workers=-1) if False else None
    anchors, positives, negatives = C._indices(pairs)
    # Reuse the suite's per-level kNN/coverage and intrusion implementation.
    neighbors = C._knn(vectors, (5, 10), workers=-1)
    neighborhood = C.neighborhood_metrics(vectors, pairs, k_values=(5, 10),
                                           precomputed_neighbors=neighbors)
    stratified = {}
    for axis, groups in strata.items():
        stratified[axis] = {}
        for name, mask in groups.items():
            sub = {"anchors": anchors[mask], "positives": positives[mask], "negatives": negatives[mask]}
            part = C.measure_level(vectors, sub, refids, local_scales=measured["local_scales"][mask])
            m = part["margins"]
            part["hard_competition_rate"] = G.hard_competition_rate(m)
            stratified[axis][name] = {"pair_rows": int(mask.sum()), "triple_count": int(m.size),
                "pncr_005": part["pncr_by_epsilon"]["0.05"], "pncr_010": part["pncr_by_epsilon"]["0.1"],
                "pncr_020": part["pncr_by_epsilon"]["0.2"], "error_rate": part["error_rate_margin"],
                "hard_competition_rate": part["hard_competition_rate"], "auc": part["behavioral_auc"],
                "margin_distribution": part["margin_distribution"]}
    slim = {k: v for k, v in measured.items() if k not in ("margins", "local_scales", "positive_distances", "negative_distances")}
    slim["hard_competition_rate"] = measured["hard_competition_rate"]
    return {"competition": slim, "margin_distribution": measured["margin_distribution"],
            "neighborhood": neighborhood, "strata": stratified,
            "local_scales": measured["local_scales"], "margins": measured["margins"]}


def _geom_space_stats(space, mapped, clamp_info):
    radius = M.radius_statistics(mapped)
    return {**radius, "angular": M.pairwise_angle_statistics(mapped),
            "angle_radius_dependence": M.angle_radius_dependence(mapped),
            "mapping_clamp": clamp_info}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residuals").mkdir(exist_ok=True)
    pairs = S._load_pairs()
    embeddings = np.load(S.EMBEDDINGS_PATH, mmap_mode="r")
    frame = pd.read_parquet(S.VALID_PATH)
    successors = S._successors(frame, pairs)
    anchors, positives, negatives = C._indices(pairs)
    refids = S._reference_ids(len(embeddings))
    pop = SG._popularity(frame, len(embeddings))
    popbins = G._rank_bins(pop[anchors], 10)
    succ_count = np.asarray([len(successors.get(int(a), ())) for a in anchors])
    results = {"protocol": {"pairs": "behavior_pairs_valid_s42.npz", "pair_rows": len(anchors),
                "negative_count_per_row": negatives.shape[1], "reference_sample_size": len(refids),
                "local_scale": "per-anchor median distance to the same seed-42 uniform 512-item reference ID set, self excluded; distances are measured in each arm's own geometry (no global multiplier or cross-metric unit conversion)",
                "clamp": "mapped radius projected to <0.99999; cap tangent norm at 2 before tanh", "arms": {}}, "arms": {}}
    for label in ("E0", "H1", "E1"):
        print(f"START {label}", flush=True)
        ckpt_src = ARMS / label / "rqvae_best.pth"
        staged = Path("/tmp") / f"hyperbolic_utilisation_{label}.pth"
        shutil.copy2(ckpt_src, staged)
        residual_file, dropped = S._extract(label, ARMS / label, output_dir=OUT / "residuals",
                                             force=False, checkpoint_override=staged)
        with np.load(residual_file) as data:
            ids = data["item_ids"].astype(np.int64)
            residuals = data["residuals"].astype(np.float64)
            codes = data["codes"].astype(np.int64)
            reconstruction = data["cumulative_reconstruction"].astype(np.float64)
            sid = data["sid"].astype(np.int64)
        z, dropped = _latent(label, staged, embeddings)
        spaces = {f"L{i+1}": residuals[:, i] for i in range(3)}
        spaces["encoder"] = z
        geom = {"spaces": {}, "triple_geometry": {}, "pipeline": {},
                "codebook": M.codebook_statistics(codes), "loader_dropped_keys": dropped}
        hyper = label == "H1"
        mapped_spaces, clamp_infos = {}, {}
        raw_for_comparison = {}
        for name, vector in spaces.items():
            mapped, clamp_info = M.map_to_ball(vector, radial_scale=1.0 if label == "E0" else 2.0)
            mapped_spaces[name], clamp_infos[name] = mapped, clamp_info
            effective = vector if label == "E0" else mapped
            geom["spaces"][name] = {"raw_norm": M.radius_statistics(vector),
                "effective_radius_and_conformal": M.radius_statistics(effective),
                "angles": M.pairwise_angle_statistics(effective),
                "angle_radius_dependence": M.angle_radius_dependence(effective),
                "mapping_clamp": clamp_info, "nonfinite_coordinates": int(vector.size - np.isfinite(vector).sum())}
        for level, vector in spaces.items():
            actual = vector if label == "E0" else mapped_spaces[level]
            raw_for_comparison[level] = actual
            result = _competition(actual, pairs, refids, G.make_strata(
                pop[anchors], succ_count, G._hardness_distance(actual, pairs), popularity_deciles=popbins))
            geom["triple_geometry"][level] = {k: v for k, v in result.items()
                if k not in ("local_scales", "margins")}
            # Geometry ranking contrast is evaluated on exactly the same mapped coordinates.
            if label != "E0":
                d_e, _ = _metric_dists(actual, pairs, False)
                d_g, _ = _metric_dists(actual, pairs, True)
                geom["triple_geometry"][level]["geodesic_vs_mapped_euclidean_ranking"] = M.ranking_difference(d_e, d_g)
        # Same ranking contrast for encoder representation.
        actual_z = z if label == "E0" else mapped_spaces["encoder"]
        if label != "E0":
            geom["triple_geometry"]["encoder"] = {"geodesic_vs_mapped_euclidean_ranking":
                M.ranking_difference(*_metric_dists(actual_z, pairs, True))}
        pipeline = {"continuous_encoder": z, "per_level_residual": spaces,
                    "cumulative_reconstruction": reconstruction, "semantic_id": codes}
        from eval_behavior_metrics import metrics_for
        for stage, vectors in pipeline.items():
            if stage == "per_level_residual":
                vals = {level: C.behavioral_auc(*_metric_dists(v, pairs, False)[0].T.reshape(2, -1)) for level, v in vectors.items()}
            elif stage == "semantic_id":
                vals = metrics_for(codes, pairs)
            else:
                d = _metric_dists(vectors, pairs, False)[0]
                vals = {"behavioral_auc": C.behavioral_auc(d[:, 0], d[:, 1])}
            geom["pipeline"][stage] = vals
        # Code-based prefix/ER and per-depth prefix sharing.
        code_metrics = metrics_for(codes, pairs)
        geom["pipeline"]["semantic_id"] = {k: code_metrics[k] for k in
            ("L1", "L1_L2", "L1_L2_L3", "L2_given_L1", "L3_given_L2", "codebook")}
        geom["pipeline"]["information_lost_stage1_to_stage4_auc"] = (
            geom["pipeline"]["continuous_encoder"]["behavioral_auc"] if isinstance(geom["pipeline"]["continuous_encoder"], dict) else "see stage-specific rows")
        geom["pipeline"]["stage1_to_stage4_information_loss_note"] = "Behavioral AUC information loss = continuous encoder AUC minus final SID AUC; SID AUC is not defined because IDs are categorical, so final-stage positive/negative prefix sharing and ER are reported instead."
        results["arms"][label] = geom
        print(f"DONE {label}", flush=True)
    (OUT / "hyperbolic_utilisation.json").write_text(json.dumps(results, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")


if __name__ == "__main__":
    run()
