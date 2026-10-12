"""Is the geodesic L1 assignment dominated by the codeword RADIUS? Offline.

Step 3 of the root-cause programme. The measured ordering says the L1 collapse is
complete by epoch 200 while the encoder norm has grown only 9% and the
reconstruction is still improving, and the prototype norm freezes at 0.1822 right
after. So the candidate is the assignment itself: in the Poincare metric, when the
codewords sit near the boundary and the items sit well inside, the geodesic
distance is dominated by the RADIAL gap, so the nearest codeword is decided mostly
by which codeword is furthest out rather than by angle.

Measured on frozen checkpoints, under the normalizer each run actually trained with:

  * how much of the geodesic assignment is explained by the codeword's radius
    alone -- the share of items whose chosen codeword is the single furthest-out
    codeword, and the Spearman correlation between the chosen codeword's radius
    rank and the radius rank of the item;
  * the same quantities for the Euclidean assignment, which is the control;
  * the spread of the assigned codewords' radii under each geometry;
  * the angular-only assignment (nearest by cosine after removing radius) so the
    angle channel can be separated from the radial one.
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
ARMS = ["nce_euclid_group", "hyper_l1_s42"]
CURV = T.L1_GEODESIC_CURVATURE


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


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
        nz = float(T.resolve_normalizer(m, all_emb.to(dev)))   # only for the printout
        recorded = None
        mp = ROOT / "logs" / arm / "training_metrics.jsonl"
        if mp.is_file():
            for line in reversed(mp.read_text(encoding="utf-8").splitlines()):
                if "ball_normalizer" in line and json.loads(line).get("ball_normalizer"):
                    recorded = float(json.loads(line)["ball_normalizer"])
                    break
        nz = recorded if recorded is not None else nz
        layer.geodesic_normalizer = nz

        with torch.no_grad():
            z = m.encoder(all_emb.to(dev))
            e = layer.get_code_embs()
            z_b = H.encode_ball(z, CURV, nz)
            e_b = H.encode_ball(e, CURV, nz)
            d_geo = H.pair_poincare_distance(z_b, e_b, CURV)
            a_geo = d_geo.argmin(-1)
            d_euc = z.pow(2).sum(1, keepdim=True) - 2 * z @ e.t() + e.pow(2).sum(1)
            a_euc = d_euc.argmin(-1)
            # angular-only: cosine between the item direction and the codeword
            # direction, both after removing radius (i.e. unit vectors)
            zu = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            eu = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            a_ang = (zu @ eu.t()).argmax(-1)
            zr = z.norm(dim=-1).cpu().numpy()
            er = e.norm(dim=-1).cpu().numpy()
            ebr = e_b.norm(dim=-1).cpu().numpy()

        def stats(a, name):
            a = a.cpu().numpy()
            radii = er[a]
            outer = int(er.argmax())
            return {
                "geometry": name,
                "distinct_codewords": int(len(np.unique(a))),
                "share_on_single_furthest_out_codeword": float((a == outer).mean()),
                "chosen_radius_mean": float(radii.mean()),
                "chosen_radius_std": float(radii.std()),
                "spearman_item_radius_vs_chosen_radius": spearman(zr, radii),
                "spearman_item_radius_vs_chosen_ball_radius": spearman(zr, ebr[a]),
                "top10_codewords_share": float(
                    np.sort(np.bincount(a, minlength=len(er)))[::-1][:10].sum() / len(a)),
            }

        report[arm] = {
            "normalizer_recorded": recorded,
            "normalizer_reestimated": float(T.resolve_normalizer(m, all_emb.to(dev))),
            "codeword_norm_median": float(np.median(er)),
            "codeword_norm_std": float(er.std()),
            "codeword_ball_radius_median": float(np.median(ebr)),
            "item_norm_median": float(np.median(zr)),
            "item_ball_radius_median": float(z_b.norm(dim=-1).median()),
            "assignments": {
                "euclid": stats(a_euc, "euclid"),
                "geodesic": stats(a_geo, "geodesic"),
                "angular_only": stats(a_ang, "angular_only"),
            },
            "geodesic_vs_euclid_change_rate": float((a_geo != a_euc).float().mean()),
            "geodesic_vs_angular_change_rate": float((a_geo != a_ang).float().mean()),
        }
    print(json.dumps(report, indent=2))
    out = STAGE2 / "assignment_radius_diagnostic.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()