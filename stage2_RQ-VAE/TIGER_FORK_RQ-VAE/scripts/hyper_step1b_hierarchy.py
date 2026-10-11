"""Step 1b: which of the three candidate causes is the real one.

Step 1 measured that the geodesic and Euclidean metrics order the 256 L1
codewords almost identically: per-item rank correlation 0.9602, and 90.7% of items
keep the same nearest codeword. A mechanism built on the geodesic metric cannot
then move the grouping boundary, so before any training the three candidates
have to be separated:

  (a) not enough radial variation in the ITEM representation
  (b) the distance normalisation cancelling the curvature
  (c) the codeword layout is not hierarchical

Step 1 already ruled out (a) for the items: none is above radius 0.95 and the
radius spread is 0.0779, so the ball map is neither saturated nor degenerate.
That leaves the codebook side.

The question this script answers is whether the 256 L1 prototypes sit at
DIFFERENT radii in the ball. The curvature of the Poincare metric buys anything
only through radius: for a fixed angle, the geodesic distance grows with the
separation of the radii. If every prototype lands at the same radius, then for
any item the distance ordering is decided by angle alone, the curvature factor is
a common multiplier, and it cancels in a ranking. That would make (c) the cause
and would mean step 2 can only re-derive the assignment already in use.

Measured here, all frozen:
  prototype radius spread relative to the item radius spread
  how much of the item-prototype distance ordering is attributable to radius
  what actually changed for the items whose assignment did flip
  the same comparison under several curvatures, to check whether the conclusion
  is a fixed-normaliser artefact rather than a property of the layout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402
import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball, poincare_distance  # noqa: E402

SEED = 42
CKPT = AP.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
OUT = MM.OUT_DIR / "hyper_step1b_hierarchy.json"


@torch.no_grad()
def frozen():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T.set_seed(SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=emb.shape[1]).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev)["state_dict"])
    model.eval()
    z = model.encoder(torch.from_numpy(emb).to(dev)).detach()
    e1 = model.rq.vq_layers[0].get_code_embs().detach()
    return z, e1, dev


def main() -> None:
    z, e1, dev = frozen()
    n, K = z.shape[0], e1.shape[0]
    # The repository fixes the normaliser at initialisation; recover it the same
    # way training does so the map is identical.
    normaliser = float(
        (np.arctanh(0.7) / float(torch.linalg.vector_norm(z, dim=-1).median().item()))
    )
    rep = {"seed": SEED, "normalizer_used": normaliser, "n_items": n, "n_codewords": K}

    for c in (1.0, 0.25, 4.0):
        zb = encode_ball(z, c, normaliser)
        eb = encode_ball(e1, c, normaliser)
        br = ball_radius(c)
        ri = torch.linalg.vector_norm(zb, dim=-1)
        rk = torch.linalg.vector_norm(eb, dim=-1)
        d_h = poincare_distance(zb[:, None, :], eb[None, :, :], c)
        d_e = torch.cdist(z, e1)
        a_h = d_h.argmin(1)
        a_e = d_e.argmin(1)

        # (c) hierarchical layout: how spread are the prototype radii?
        proto = {
            "radius_mean": float(rk.mean()),
            "radius_std": float(rk.std()),
            "radius_cv": float(rk.std() / rk.mean()),
            "radius_min": float(rk.min()),
            "radius_max": float(rk.max()),
        }
        item = {
            "radius_mean": float((ri / br).mean()),
            "radius_std": float((ri / br).std()),
        }
        # How much of the spread of the 256 distances is radial? Decompose the
        # item-prototype squared distance into its radial and tangential parts in
        # the BALL, which is where the metric lives.
        ui = zb / ri.unsqueeze(1).clamp_min(1e-12)
        cos = (ui @ (eb / rk.unsqueeze(1).clamp_min(1e-12)).T).clamp(-1, 1)
        dr = (ri.unsqueeze(1) - rk.unsqueeze(0))
        # angular chord contribution, to scale comparable with the radial one
        chord_ang = (2.0 * (1.0 - cos)).clamp_min(0).sqrt()
        radial_var = float(dr.var(dim=1).mean())
        angular_var = float((chord_ang ** 2).var(dim=1).mean())

        # What flipped, for the items that did?
        flip = a_e != a_h
        idx = torch.where(flip)[0]
        rank_e = d_e[idx].argsort(1).argsort(1).float()
        rank_h = d_h[idx].argsort(1).argsort(1).float()
        rc = []
        for j in range(0, len(idx), 29):
            a = rank_e[j] - rank_e[j].mean()
            b = rank_h[j] - rank_h[j].mean()
            rc.append(float((a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()))

        rep[f"curvature_{c}"] = {
            "prototype_radius": proto,
            "item_radius": item,
            "radial_var_share_of_distance_spread":
                radial_var / max(radial_var + angular_var, 1e-12),
            "assignment_change_rate": float(flip.float().mean()),
            "flipped_rank_agreement_mean": float(np.mean(rc)) if rc else None,
        }

    # The decisive one: is the conclusion an artefact of the fixed normaliser?
    # Re-run at several normalisers, which changes how far out the items sit and
    # therefore how much radial room the map has.
    sweep = {}
    for mult in (0.5, 1.0, 2.0, 4.0):
        nz = normaliser * mult
        zb = encode_ball(z, 1.0, nz)
        eb = encode_ball(e1, 1.0, nz)
        d_h = poincare_distance(zb[:, None, :], eb[None, :, :], 1.0)
        d_e = torch.cdist(z, e1)
        sweep[f"normalizer_x{mult}"] = {
            "item_radius_p50": float((torch.linalg.vector_norm(zb, dim=-1)).median()),
            "frac_items_outside_ball": float(
                (torch.linalg.vector_norm(zb, dim=-1) >= 1.0).float().mean()),
            "assignment_change_rate": float((d_h.argmin(1) != d_e.argmin(1)).float().mean()),
            "prototype_radius_cv": float(
                torch.linalg.vector_norm(eb, dim=-1).std()
                / torch.linalg.vector_norm(eb, dim=-1).mean()),
        }
    rep["normalizer_sweep"] = sweep

    OUT.write_text(json.dumps(rep, indent=2))
    print("=== 原型径向布局 (曲率 1.0) ===")
    g = rep["curvature_1.0"]
    print(f"  原型半径 mean={g['prototype_radius']['radius_mean']:.5f} "
          f"std={g['prototype_radius']['radius_std']:.5f} "
          f"CV={g['prototype_radius']['radius_cv']:.5f}")
    print(f"  原型半径范围 [{g['prototype_radius']['radius_min']:.5f}, "
          f"{g['prototype_radius']['radius_max']:.5f}]")
    print(f"  商品半径(p50归一) mean={g['item_radius']['radius_mean']:.5f} "
          f"std={g['item_radius']['radius_std']:.5f}")
    print(f"  距离展宽中径向占比 {g['radial_var_share_of_distance_spread']:.4f}")
    print(f"  分配变化率 {g['assignment_change_rate']:.4f}  "
          f"翻转商品的秩一致度 {g['flipped_rank_agreement_mean']:.4f}")
    print("=== 曲率扫描: 分配变化率 ===")
    for c in (1.0, 0.25, 4.0):
        k = rep[f"curvature_{c}"]
        print(f"  c={c:<5} 变化率={k['assignment_change_rate']:.4f} "
              f"原型半径CV={k['prototype_radius']['radius_cv']:.4f} "
              f"径向占比={k['radial_var_share_of_distance_spread']:.4f}")
    print("=== 归一化扫描 (曲率 1.0) ===")
    for k, v in sweep.items():
        print(f"  {k:<18} 商品半径p50={v['item_radius_p50']:.4f} "
              f"越界={v['frac_items_outside_ball']:.4f} "
              f"变化率={v['assignment_change_rate']:.4f} "
              f"原型CV={v['prototype_radius_cv']:.4f}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()