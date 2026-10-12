"""Offline comparison of the geometry-consistent L1 objectives. No training.

Loads stored Stage 2 checkpoints, and for the level-1 codebook measures, for each
candidate objective:

  grad norm on the selected codewords, and its cosine against the objective the
  arm actually trains with (the pre-map Euclidean MSE);
  the effect of ONE optimiser step at the arm's own learning rate, measured by
  the ball distance between a codeword and the item it represents;
  the boundary-saturation share (mapped radius > 0.99) before and after.

The point is to decide between formulations WITHOUT spending GPU hours: a
formulation whose gradient is orders of magnitude larger, or whose one step moves
the codebook out of the usable range, is not a candidate however good its
direction is.

Objectives compared:
  premap        mse(e_sel, sg[z]) + beta * mse(z, sg[e_sel])          (current)
  ball          mse(enc(e_sel), sg[enc(z)]) + beta * mse(enc(z), sg[enc(e_sel)])
  ball_over_nz2 the same, divided by normalizer**2
  tangent       ||log_{enc(e_sel)}( enc(z) )||^2  at the mapped codeword
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
BETA = 0.25


def mobius_sub(x, y, c):
    """x -_c y = x (+) (-y): the ball is a group under (+) and -y is y's inverse."""
    return H.mobius_add(x, -y, c)


def log_map_at(p, q, c):
    """log_p(q): the tangent vector at p whose exponential map is q."""
    d = mobius_sub(q, p, c)
    nd = (c ** 0.5) * torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(H.NORM_FLOOR)
    scale = torch.atanh(nd.clamp(max=1.0 - H.BOUNDARY_EPS)) / nd
    return scale * d


def enc(x, c, nz):
    return H.encode_ball(x, c, nz)


def ball_dist(z_b, e_b, c):
    return float(H.poincare_distance(z_b, e_b, c))


def objectives(z, e_sel, layer, nz):
    c = layer.geodesic_curvature
    z_b, e_b = enc(z, c, nz), enc(e_sel, c, nz)
    return {
        "premap": F.mse_loss(e_sel, z.detach()) + BETA * F.mse_loss(z, e_sel.detach()),
        "ball": F.mse_loss(e_b, z_b.detach()) + BETA * F.mse_loss(z_b, e_b.detach()),
        "ball_over_nz2": (F.mse_loss(e_b, z_b.detach()) + BETA * F.mse_loss(z_b, e_b.detach()))
        / (nz ** 2),
        # NOT detached: the prototype is the base point of the log map, so the
        # codebook must be able to move through it as well as through the vector.
        "tangent": (log_map_at(e_b, z_b, c) ** 2).sum(-1).mean(),
    }


def selected_rows(layer, z):
    idx = layer._distances(z).argmin(-1)
    return idx, layer.get_code_embs()[idx]


def saturation(layer, nz):
    c = layer.geodesic_curvature
    return {
        "item_r_gt_0.99": float((enc(layer._probe_z, c, nz).norm(dim=-1) > 0.99).float().mean()),
        "code_r_gt_0.99": float((enc(layer.get_code_embs(), c, nz).norm(dim=-1) > 0.99).float().mean()),
        "code_r_median": float(enc(layer.get_code_embs(), c, nz).norm(dim=-1).median()),
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
            report[arm] = {"status": "missing checkpoint"}
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}
        m = RQVAE(T._tokenizer_config(cfg), in_dim=emb.shape[1]).to(dev)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        layer = m.rq.vq_layers[0]
        nz = float(T.resolve_normalizer(m, all_emb.to(dev)))
        layer.geodesic_normalizer = nz
        with torch.no_grad():
            z = m.encoder(all_emb[:4096].to(dev))
            layer._probe_z = z
        entry = {"normalizer": nz, "curvature": CURV, "before": saturation(layer, nz)}

        # per-objective gradient on the SELECTED codewords, and its cosine vs premap
        grads = {}
        enc_norm, loss_val = {}, {}
        for name in ("premap", "ball", "ball_over_nz2", "tangent"):
            layer.zero_grad(set_to_none=True)
            # z must carry a grad_fn here; the no_grad block above was only for
            # the saturation measurement.
            z_grad = m.encoder(all_emb[:4096].to(dev))
            idx, e_sel = selected_rows(layer, z_grad)
            loss = objectives(z_grad, e_sel, layer, nz)[name]
            loss.backward()
            g = layer.get_code_embs().grad.detach().clone()
            enc_grad = torch.cat([p.grad.reshape(-1) for p in m.encoder.parameters()
                                  if p.grad is not None])
            grads[name] = g
            enc_norm[name] = float(enc_grad.norm())
            loss_val[name] = float(loss.detach())
        # The reconstruction term alone, for the relative-weight measurement. The
        # encoder gradient is shared between it and the codebook term, so the ratio
        # of their contributions to that gradient IS the effective step each
        # objective gets -- and it is scale-invariant in neither direction.
        m.zero_grad(set_to_none=True)
        z_rec = m.encoder(all_emb[:4096].to(dev))
        rec = F.mse_loss(m.decoder(z_rec), all_emb[:4096].to(dev))
        rec.backward()
        rec_norm = float(torch.cat([p.grad.reshape(-1) for p in m.encoder.parameters()
                                    if p.grad is not None]).norm())
        base = grads["premap"].reshape(-1)
        rows = {}
        for name, g in grads.items():
            flat = g.reshape(-1)
            denom = float(flat.norm()) * float(base.norm())
            rows[name] = {
                "grad_norm": float(flat.norm()),
                "grad_norm_ratio_vs_premap": float(flat.norm()) / max(float(base.norm()), 1e-30),
                "cosine_vs_premap_grad": float((flat * base).sum() / denom) if denom > 0 else float("nan"),
            }
        rows["_reconstruction_only"] = {"encoder_grad_norm": rec_norm,
                                        "loss_value": float(rec.detach())}
        for name in ("premap", "ball", "ball_over_nz2", "tangent"):
            rows[name]["encoder_grad_norm"] = enc_norm[name]
            rows[name]["loss_value"] = loss_val[name]
            rows[name]["encoder_grad_ratio_vs_reconstruction"] = (
                enc_norm[name] / max(rec_norm, 1e-30))
        entry["gradient"] = rows

        # ONE step at the arm's own learning rate (AdamW default lr 1e-3 here),
        # starting each time from the same weights, measured by the ball distance.
        lr = 1e-3
        c = layer.geodesic_curvature
        for name in ("premap", "ball", "ball_over_nz2", "tangent"):
            m.load_state_dict(blob["state_dict"])  # full model, not the layer
            layer.geodesic_normalizer = nz
            with torch.no_grad():
                z = m.encoder(all_emb[:4096].to(dev))
                layer._probe_z = z
                _, e_sel = selected_rows(layer, z)
                before = ball_dist(enc(z[0:1], c, nz), enc(e_sel[0:1], c, nz), c)
            opt = torch.optim.AdamW([layer.get_code_embs()], lr=lr, weight_decay=0.0)
            opt.zero_grad(set_to_none=True)
            idx, e_sel = selected_rows(layer, z)
            objectives(z, e_sel, layer, nz)[name].backward()
            opt.step()
            with torch.no_grad():
                z = m.encoder(all_emb[:4096].to(dev))
                _, e_sel = selected_rows(layer, z)
                after = ball_dist(enc(z[0:1], c, nz), enc(e_sel[0:1], c, nz), c)
            rows[name]["one_step_ball_dist_before"] = before
            rows[name]["one_step_ball_dist_after"] = after
            rows[name]["one_step_ball_dist_delta"] = after - before
        entry["after_tangent_step"] = saturation(layer, nz)
        report[arm] = entry

    print(json.dumps(report, indent=2))
    out = STAGE2 / "geometry_objective_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()