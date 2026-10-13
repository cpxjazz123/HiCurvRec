from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"))
sys.path.insert(0, str(ROOT / "analysis/crowding_diagnostic"))
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/scripts"))
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


def _point_distance(u, v, hyperbolic=False):
    if not hyperbolic:
        return np.linalg.norm(u - v, axis=-1)
    delta2 = np.sum((u - v) ** 2, axis=-1)
    nu, nv = np.sum(u * u, axis=-1), np.sum(v * v, axis=-1)
    argument = 1 + 2 * delta2 / np.maximum((1 - nu) * (1 - nv), 1e-20)
    return np.arccosh(np.maximum(argument, 1.0))


def _metric_dists(x, pair, hyperbolic):
    a, p, n = (x[pair[k]] for k in ("anchors", "positives", "negatives"))
    pos = _point_distance(a, p, hyperbolic)
    neg = _point_distance(a[:, None], n, hyperbolic)
    return np.concatenate((pos[:, None], neg), axis=1)


def _fast_headlines(pos, neg, margins):
    flat = margins.reshape(-1)
    return {"triple_count": int(flat.size),
        "pncr_by_epsilon": {str(e): {"value": float(np.mean(np.abs(flat) < e)), "n": int(flat.size)}
                            for e in C.EPSILONS},
        "error_rate_margin": {"value": float(np.mean(flat < 0)), "n": int(flat.size)},
        "hard_competition_rate": G.hard_competition_rate(flat),
        "behavioral_auc": {"value": C.behavioral_auc(pos, neg), "n": int(flat.size)},
        "margin_distribution": M._distribution(flat)}
def _competition(vectors, pairs, local_scales, strata, hyperbolic):
    anchors, positives, negatives = C._indices(pairs)
    pos = _point_distance(vectors[anchors], vectors[positives[:, 0]], hyperbolic)
    neg = _point_distance(vectors[anchors, None], vectors[negatives], hyperbolic)
    margins = (neg - pos[:, None]) / local_scales[:, None]
    measured = C._triple_metrics(pos, neg, margins, anchors, C.EPSILONS)
    measured["hard_competition_rate"] = G.hard_competition_rate(margins)
    neighborhood = C.neighborhood_metrics(vectors, pairs, k_values=(5, 10))
    stratified = {}
    for axis, groups in strata.items():
        stratified[axis] = {}
        for name, mask in groups.items():
            sub = _fast_headlines(pos[mask], neg[mask], margins[mask])
            stratified[axis][name] = {"pair_rows": int(mask.sum()), "triple_count": int(margins[mask].size),
                "pncr_005": sub["pncr_by_epsilon"]["0.05"], "pncr_010": sub["pncr_by_epsilon"]["0.1"],
                "pncr_020": sub["pncr_by_epsilon"]["0.2"], "error_rate": sub["error_rate_margin"],
                "hard_competition_rate": sub["hard_competition_rate"], "auc": sub["behavioral_auc"],
                "margin_distribution": sub["margin_distribution"]}
    return {"competition": measured, "margin_distribution": measured["margin_distribution"],
            "neighborhood": neighborhood, "strata": stratified,
            "local_scales": local_scales, "margins": margins}


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
    common_local_scales = C._local_scales(np.asarray(embeddings), anchors, refids)
    pop = SG._popularity(frame, len(embeddings))
    popbins = G._rank_bins(pop[anchors], 10)
    succ_count = np.asarray([len(successors.get(int(a), ())) for a in anchors])
    results = {"protocol": {"pairs": "behavior_pairs_valid_s42.npz", "pair_rows": len(anchors),
                "negative_count_per_row": negatives.shape[1], "reference_sample_size": len(refids),
                "local_scale": "shared per-anchor median distance to the same seed-42 reference IDs measured once in the frozen 768-D Stage-1 item-embedding space; native metric distances use this single scale without per-arm/per-metric rescaling",
                "clamp": "mapped radius projected to <0.99999; cap tangent norm at 2 before tanh", "arms": {}}, "arms": {}}
    for label in ("E0", "H1", "E1"):
        print(f"START {label}", flush=True)
        ckpt_src = ARMS / label / "rqvae_best.pth"
        staged = Path("/tmp") / f"hyperbolic_utilisation_{label}.pth"
        shutil.copy2(ckpt_src, staged)
        residual_file, dropped = S._extract(label, ARMS / label, output_dir=OUT / "residuals",
                                             force=False, checkpoint_override=staged)
        with np.load(residual_file) as data:
            residuals = data["residuals"].astype(np.float64)
            codes = data["codes"].astype(np.int64)
            reconstruction = data["cumulative_reconstruction"].astype(np.float64)
        z, dropped = _latent(label, staged, embeddings)
        spaces = {f"L{i+1}": residuals[:, i] for i in range(3)}
        spaces["encoder"] = z
        geom = {"spaces": {}, "triple_geometry": {}, "pipeline": {},
                "codebook": M.codebook_statistics(codes), "loader_dropped_keys": dropped}
        geom["training_gradient_context"] = {
            "geometry_encoder_grad_norm": {"E0": 0.1963, "H1": 0.1842, "E1": 0.1311}[label],
            "codebook_geometry_gradient_norm": 0.0,
            "note": "terminal training values supplied with the experiment specification; not re-measured in this analysis"}
        geom["exp1a_occupancy_reference"] = {"L1": "241/256", "L2": "256/256", "L3": "256/256"}
        for name, vector in spaces.items():
            mapped, clamp_info = M.map_to_ball(vector, radial_scale=1.0 if label == "E0" else 2.0)
            mapped_spaces[name] = mapped
            effective = vector if label == "E0" else mapped
            effective_stats = M.radius_statistics(effective)
            if label == "E0":
                effective_stats["conformal_factor"] = None
            geom["spaces"][name] = {"raw_norm": M.radius_statistics(vector),
                "effective_radius_and_conformal": effective_stats,
                "angles": M.pairwise_angle_statistics(effective),
                "angle_radius_dependence": M.angle_radius_dependence(effective),
                "mapping_clamp": clamp_info, "nonfinite_coordinates": int(vector.size - np.isfinite(vector).sum())}
        for level, vector in spaces.items():
            actual = vector if label == "E0" else mapped_spaces[level]
            metric_dists = _metric_dists(actual, pairs, label == "H1")
            hardness = metric_dists[:, 1:].min(axis=1) / np.maximum(metric_dists[:, 0], np.finfo(float).tiny)
            result = _competition(actual, pairs, common_local_scales,
                G.make_strata(pop[anchors], succ_count, hardness, popularity_deciles=popbins), label == "H1")
            geom["triple_geometry"][level] = {k: v for k, v in result.items()
                if k not in ("local_scales", "margins")}
            # Geometry ranking contrast is evaluated on exactly the same mapped coordinates.
            if label != "E0":
                d_e = _metric_dists(actual, pairs, False)
                d_g = _metric_dists(actual, pairs, True)
                geom["triple_geometry"][level]["geodesic_to_euclidean_distance_ratio"] = \
                    M._distribution((d_g / np.maximum(d_e, 1e-12)).reshape(-1))
                geom["triple_geometry"][level]["geodesic_vs_mapped_euclidean_ranking"] = M.ranking_difference(d_e, d_g)
        actual_z = z if label == "E0" else mapped_spaces["encoder"]
        if label != "E0":
            enc_e = _metric_dists(actual_z, pairs, False)
            enc_g = _metric_dists(actual_z, pairs, True)
            geom["triple_geometry"]["encoder"] = {
                "geodesic_to_euclidean_distance_ratio": M._distribution(
                    (enc_g / np.maximum(enc_e, 1e-12)).reshape(-1)),
                "geodesic_vs_mapped_euclidean_ranking": M.ranking_difference(enc_e, enc_g)}
        from eval_behavior_metrics import metrics_for
        def auc_for(vectors):
            points = vectors if label == "E0" else M.map_to_ball(vectors, 2.0)[0]
            d = _metric_dists(points, pairs, label == "H1")
            return C.behavioral_auc(d[:, 0], d[:, 1:])
        stage1_auc = auc_for(z)
        pipeline = {"continuous_encoder": {"behavioral_auc": stage1_auc}, "per_level_residual": {},
                    "cumulative_reconstruction": {}, "semantic_id": {}}
        for level, vectors in spaces.items():
            if level != "encoder":
                pipeline["per_level_residual"][level] = {"behavioral_auc": auc_for(vectors)}
        pipeline["cumulative_reconstruction"] = {"behavioral_auc": auc_for(reconstruction)}
        code_distance_pos = np.count_nonzero(codes[anchors] != codes[positives], axis=1)
        code_distance_neg = np.count_nonzero(codes[anchors, None] != codes[negatives], axis=2)
        sid_auc = C.behavioral_auc(code_distance_pos, code_distance_neg)
        code_metrics = metrics_for(codes, pairs)
        pipeline["semantic_id"] = {"hamming_behavioral_auc": sid_auc,
            **{k: code_metrics[k] for k in
               ("L1", "L1_L2", "L1_L2_L3", "L2_given_L1", "L3_given_L2", "codebook")}}
        pipeline["information_lost_stage1_to_stage4_auc"] = stage1_auc - sid_auc
        geom["pipeline"] = pipeline
        results["arms"][label] = geom
        print(f"DONE {label}", flush=True)
    (OUT / "hyperbolic_utilisation.json").write_text(json.dumps(results, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")


if __name__ == "__main__":
    run()
