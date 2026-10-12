"""Frozen-representation diagnostic: where does the L1 collapse actually happen?

No training. For each stored Stage 2 checkpoint the encoder and the three
codebooks are loaded as they were saved, the SAME representation and the SAME
codewords are then assigned twice:

  euclid   argmin_k ||z - e_k||^2            (the objective the codebook is trained by)
  geodesic argmin_k d_ball(enc(z), enc(e_k)) (the selection the arm uses)

Everything reported is measured on those two assignments of one frozen model, so
a difference between the arms cannot come from training and a difference between
the two assignments inside one arm cannot come from a different model.

Reported per arm and per geometry: distinct L1 codewords, occupancy deciles, the
largest codeword's share, the nearest-vs-second-nearest prototype margin, the
prototype norms and the representation norms, and the share of items whose L1
codeword changes between the two assignments.
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
ARMS = [
    ("nce_euclid_group", "matched Euclidean control"),
    ("hyper_l1_s42", "L1 geodesic selection"),
    ("hyper_l1_br_euclid_s42", "L1 geodesic selection + Euclidean boundary repulsion"),
]
CURV = 1.0


def occupancy_stats(idx: np.ndarray, n_codes: int) -> dict:
    counts = np.bincount(idx, minlength=n_codes)
    used = counts > 0
    srt = np.sort(counts[used])[::-1]
    total = int(counts.sum())
    return {
        "distinct_codes": int(used.sum()),
        "capacity": int(n_codes),
        "utilisation": float(used.sum() / n_codes),
        "largest_code_share": float(srt[0] / total),
        "top10_code_share": float(srt[:10].sum() / total),
        "median_code_count": float(np.median(srt)),
        "used_codes_share_of_items_ge_100": int((srt >= 100).sum()),
    }


def assign_and_measure(model: RQVAE, lat: torch.Tensor, layer, normalizer: float) -> dict:
    n_codes = layer.n_embed
    e = layer.get_code_embs().detach()
    z = lat.detach()

    d_euc = z.pow(2).sum(1, keepdim=True) - 2 * z @ e.t() + e.pow(2).sum(1)
    a_euc = d_euc.argmin(-1)

    z_b = H.encode_ball(z, CURV, normalizer)
    e_b = H.encode_ball(e, CURV, normalizer)
    d_geo = H.pair_poincare_distance(z_b, e_b, CURV)
    a_geo = d_geo.argmin(-1)

    # nearest-vs-second-nearest margin, both geometries
    def margin(d: torch.Tensor) -> torch.Tensor:
        top2 = d.topk(2, dim=-1, largest=False).values
        return (top2[:, 1] - top2[:, 0]).detach()

    out = {
        "euclid": occupancy_stats(a_euc.numpy(), n_codes),
        "geodesic": occupancy_stats(a_geo.numpy(), n_codes),
        "assignment_change_rate": float((a_euc != a_geo).float().mean()),
        "euclid_nn_margin_mean": float(margin(d_euc).mean()),
        "euclid_nn_margin_median": float(margin(d_euc).median()),
        "geodesic_nn_margin_mean": float(margin(d_geo).mean()),
        "geodesic_nn_margin_median": float(margin(d_geo).median()),
        "prototype_norm_mean": float(e.norm(dim=-1).mean()),
        "prototype_norm_max": float(e.norm(dim=-1).max()),
        "representation_norm_median": float(z.norm(dim=-1).median()),
        "representation_norm_p99": float(torch.quantile(z.norm(dim=-1), 0.99)),
        "ball_radius_median": float(z_b.norm(dim=-1).median()),
        "normalizer_used": float(normalizer),
    }
    return out


def main() -> None:
    T.set_seed(T.SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    all_emb = torch.from_numpy(emb)
    report = {}
    for arm, note in ARMS:
        ckpt_path = STAGE2 / arm / "rqvae_best.pth"
        if not ckpt_path.is_file():
            report[arm] = {"status": "missing checkpoint"}
            continue
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        geo_cfg = {"normalizer": 1.0, "curvature": T.L1_GEODESIC_CURVATURE}
        model = RQVAE(T._tokenizer_config(geo_cfg), in_dim=emb.shape[1])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        layer = model.rq.vq_layers[0]
        # The run constant the arm actually trained with, recomputed from the
        # checkpoint's own encoder, so nothing here depends on a file outside the
        # checkpoint.
        normalizer = T.resolve_normalizer(model, all_emb)
        with torch.no_grad():
            lat = model.encoder(all_emb)
        entry = {"note": note, "checkpoint": str(ckpt_path)}
        entry.update(assign_and_measure(model, lat, layer, normalizer))
        report[arm] = entry

        e = layer.get_code_embs().detach()
        z = lat.detach()
        z_b = H.encode_ball(z, CURV, normalizer)
        e_b = H.encode_ball(e, CURV, normalizer)
        # Where do the items sit relative to their nearest codeword, in the ball?
        d_geo = H.pair_poincare_distance(z_b, e_b, CURV)
        d_geo_sorted = d_geo.topk(2, dim=-1, largest=False).values
        entry["ball_geometry"] = {
            "item_ball_radius_median": float(z_b.norm(dim=-1).median()),
            "codeword_ball_radius_median": float(e_b.norm(dim=-1).median()),
            "nn1_geodesic_median": float(d_geo_sorted[:, 0].median()),
            "nn2_geodesic_median": float(d_geo_sorted[:, 1].median()),
            "nn1_over_nn2_median": float((d_geo_sorted[:, 0] / d_geo_sorted[:, 1].clamp_min(1e-9)).median()),
            "geodesic_distance_p99": float(torch.quantile(d_geo_sorted[:, 1], 0.99)),
        }

    print(json.dumps(report, indent=2))
    out = STAGE2 / "frozen_assignment_diagnostic.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()