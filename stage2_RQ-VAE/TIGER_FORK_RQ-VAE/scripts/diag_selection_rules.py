"""Offline comparison of L1 SELECTION RULES. No training.

Step 4 of the root-cause programme. Step 3 located the defect in the selection
rule, not in the codebook update: in the Poincare metric the distance between two
points grows with both radii, so for items pinned near the boundary the nearest
codeword is the one with the smallest norm whatever its angle. On the healthy
control the full geodesic rule keeps 51 of 256 codewords and puts 79.67% of items
on ten of them, while an angular-only rule on the same codebook keeps 247.

This compares candidate replacement rules on the same frozen checkpoints, each
under the normalizer its run actually trained with, with the same table that
convicted the current rule: distinct codewords, top-ten share, the share landing on
the single furthest-out codeword, and the mean/std norm of the chosen codewords.

Rules:
  euclid                argmin ||z - e||^2                       (the control)
  geodesic              argmin d_c(enc(z), enc(e))              (the current rule)
  angular_cos           argmin angle between z and e directions (radius removed)
  geodesic_radius_norm  argmin d_c(z, e) / d_c(0, e)            (per-codeword radial normalisation)
  chart_tangent         argmin ||log_0 stuff||: the plain chart distance after
                        projecting both to a COMMON radius (radius-free comparison)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import hyperbolic as H  # noqa: E402
import train_rqvae as T  # noqa: E402
from model import RQVAE  # noqa: E402

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE2 = REPO / "results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE"
LOGS = ROOT / "logs"
ARMS = ["nce_euclid_group", "hyper_l1_s42"]
CURV = T.L1_GEODESIC_CURVATURE


def recorded_normalizer(arm: str):
    mp = LOGS / arm / "training_metrics.jsonl"
    if not mp.is_file():
        return None
    for line in reversed(mp.read_text(encoding="utf-8").splitlines()):
        if "ball_normalizer" in line:
            rec = json.loads(line)
            if rec.get("ball_normalizer"):
                return float(rec["ball_normalizer"])
    return None


def rule_stats(a: np.ndarray, e_norm: np.ndarray, n: int) -> dict:
    radii = e_norm[a]
    srt = np.sort(np.bincount(a, minlength=n))[::-1]
    return {
        "distinct_codewords": int(len(np.unique(a))),
        "top10_share": float(srt[:10].sum() / len(a)),
        "share_on_single_furthest_out": float((a == int(e_norm.argmax())).mean()),
        "chosen_norm_mean": float(radii.mean()),
        "chosen_norm_std": float(radii.std()),
    }


def main() -> None:
    T.set_seed(T.SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    all_emb = torch.from_numpy(emb)
    dev = torch.device("cuda:0")
    report = {}
    for arm in ARMS:
        ck = STAGE2 / arm / "rqvae_best.pth"
        if not ck.is_file():
            report[arm] = {"status": "missing"}
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}
        m = RQVAE(T._tokenizer_config(cfg), in_dim=emb.shape[1]).to(dev)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        layer = m.rq.vq_layers[0]
        nz = recorded_normalizer(arm) or float(T.resolve_normalizer(m, all_emb.to(dev)))
        layer.geodesic_normalizer = nz

        with torch.no_grad():
            z = m.encoder(all_emb.to(dev))
            e = layer.get_code_embs()
            n = e.shape[0]
            z_b, e_b = H.encode_ball(z, CURV, nz), H.encode_ball(e, CURV, nz)
            d_geo = H.pair_poincare_distance(z_b, e_b, CURV)
            d_euc = z.pow(2).sum(1, keepdim=True) - 2 * z @ e.t() + e.pow(2).sum(1)
            zu = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            eu = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            d_ang = 1.0 - zu @ eu.t()
            # per-codeword radial normalisation: divide the geodesic distance by the
            # codeword's own distance from the origin
            d_origin = H.pair_poincare_distance(
                torch.zeros(1, e_b.shape[1], device=dev), e_b, CURV)[0]
            d_radnorm = d_geo / d_origin.clamp_min(1e-6).unsqueeze(0)
            # radius-free comparison: project BOTH sides to a common radius, then
            # take the chart distance (this is the angular channel expressed as a
            # distance rather than a cosine)
            r_common = 0.5
            def to_radius(x_b):
                return x_b / x_b.norm(dim=-1, keepdim=True).clamp_min(1e-12) * r_common
            d_chart = (to_radius(z_b)[:, None, :] - to_radius(e_b)[None, :, :]).pow(2).sum(-1)

            rules = {
                "euclid": d_euc,
                "geodesic": d_geo,
                "angular_cos": d_ang,
                "geodesic_radius_norm": d_radnorm,
                "chart_tangent": d_chart,
            }
            e_norm = e.norm(dim=-1).cpu().numpy()
            assigns = {k: v.argmin(-1).cpu().numpy() for k, v in rules.items()}

        entry = {
            "normalizer_recorded": recorded_normalizer(arm),
            "item_ball_radius_median": float(z_b.norm(dim=-1).median()),
            "codeword_norm_median": float(np.median(e_norm)),
            "codeword_ball_radius_median": float(e_b.norm(dim=-1).median()),
            "rules": {k: rule_stats(a, e_norm, n) for k, a in assigns.items()},
        }
        base = assigns["euclid"]
        entry["agreement_with_euclid"] = {
            k: float((a == base).mean()) for k, a in assigns.items()}
        report[arm] = entry

    print(json.dumps(report, indent=2))
    out = STAGE2 / "selection_rule_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()