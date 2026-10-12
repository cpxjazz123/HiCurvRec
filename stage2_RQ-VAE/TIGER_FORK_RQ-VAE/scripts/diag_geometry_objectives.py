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
LOGS = REPO / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/logs"
ARMS = ["nce_euclid_group", "hyper_l1_s42"]
CURV = T.L1_GEODESIC_CURVATURE
BETA = 0.25


def mobius_sub(x, y, c):
    """q -_c p = (-p) (+) q. The ball is a group under Boachy-Mobius (+), the
    inverse of p is -p, and (+) is commutative, so (-p) (+) q == q (+) (-p)."""
    return H.mobius_add(-y, x, c)


def log_map_p(p, q, c):
    """The plain chart log map: the tangent vector at p whose exponential map is q.

        log_p(q) = arctanh(sqrt(c) ||q -_c p||) / sqrt(c) * (q -_c p) / ||q -_c p||
    """
    d = mobius_sub(q, p, c)
    nd = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(H.NORM_FLOOR)
    return (torch.atanh((c ** 0.5 * nd).clamp(max=1.0 - H.BOUNDARY_EPS))
            / (c ** 0.5 * nd)) * d


def riemannian_log_p(p, q, c):
    """The RIEMANNIAN logarithm at p, in the standard Poincare-ball form.

        log_p(q) = 2 / (sqrt(c) * lambda_p) * arctanh(sqrt(c)||-p (+) q||)
                   * (-p (+) q) / ||-p (+) q||
        lambda_p = 2 / (1 - c ||p||^2)   the conformal (metric) factor at p

    which simplifies to  ((1 - c||p||^2) / sqrt(c)) * arctanh(...) * (-p (+) q)/||-p (+) q||.

    Two things matter and are both checked by the regression test:
      * the direction is (-p) (+) q, and the Euclidean gradient of d_c(., p) is a
        NEGATIVE constant multiple of this vector -- verified against central
        differences of the geodesic distance, so the direction is not assumed;
      * the metric factor (1 - c||p||^2) is what separates it from the plain chart
        log map: log^riem = (1 - c||p||^2) * log^plain.
    """
    d = mobius_sub(q, p, c)
    nd = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(H.NORM_FLOOR)
    lam_p = 2.0 / (1.0 - c * p.pow(2).sum(-1, keepdim=True)).clamp_min(H.NORM_FLOOR)
    return ((2.0 / (c ** 0.5 * lam_p))
            * torch.atanh((c ** 0.5 * nd).clamp(max=1.0 - H.BOUNDARY_EPS))
            * d / nd)


def read_training_normalizer(arm: str):
    """The ball normalizer the arm ACTUALLY trained with, read from its own record.

    It is resolved once from the initial encoder and frozen for the run; reading
    it back from the run's record is the only way to reproduce the geometry the
    checkpoint sits in. Re-estimating it from the FINAL encoder gives a different
    number, because the encoder's norm has moved.
    """
    mpath = LOGS / arm / "training_metrics.jsonl"
    if not mpath.is_file():
        return None
    for line in reversed(mpath.read_text(encoding="utf-8").splitlines()):
        if "ball_normalizer" not in line:
            continue
        rec = json.loads(line)
        if rec.get("ball_normalizer") is not None:
            return float(rec["ball_normalizer"])
    return None


def enc(x, c, nz):
    return H.encode_ball(x, c, nz)


def ball_dist(z_b, e_b, c):
    return float(H.poincare_distance(z_b, e_b, c))


def candidate_objectives(z, e_sel, layer, nz):
    """All candidates for ONE CURRENT, non-accumulated backward.

    The caller must call model.zero_grad(set_to_none=True) before every one of
    them -- the encoder gradient from a previous candidate would otherwise
    accumulate and every measurement after the first would be wrong.
    """
    c = layer.geodesic_curvature
    z_b, e_b = enc(z, c, nz), enc(e_sel, c, nz)
    ball = F.mse_loss(e_b, z_b.detach()) + BETA * F.mse_loss(z_b, e_b.detach())
    return {
        "premap": F.mse_loss(e_sel, z.detach()) + BETA * F.mse_loss(z, e_sel.detach()),
        "ball": ball,
        "ball_over_nz2": ball / (nz ** 2),
        # the RIEMANNIAN logarithm, not the plain log map, and the prototype is
        # the base point so it is NOT detached
        "tangent": (riemannian_log_p(e_b, z_b, c) ** 2).sum(-1).mean(),
        # squared geodesic distance between mapped item and mapped codeword, the
        # same metric the selection argmin uses
        "geo2": (H.poincare_distance(z_b, e_b, c) ** 2).mean(),
    }


def real_reconstruction_gradient(m, x):
    """The encoder gradient of the REAL objective on this batch.

    Forward through the actual RQ-VAE (encode -> residual quantise -> decode),
    then the reconstruction loss. A decoder applied to a bare encoder latent skips
    the straight-through estimator and the commitment terms, and measures a
    different function than the one training minimises.
    """
    m.zero_grad(set_to_none=True)
    reconstructed, _quant_loss, _, _ = m(x)
    loss = F.mse_loss(reconstructed, x.detach())
    loss.backward()
    return (torch.cat([p.grad.reshape(-1) for p in m.encoder.parameters()
                       if p.grad is not None]),
            float(loss.detach()))


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
    names = ["premap", "ball", "ball_over_nz2", "tangent", "geo2"]
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
        # The normalizer the run TRAINED with, from the run's own record. Re-estimating
        # it from this final encoder would put the checkpoint in a geometry it never
        # saw.
        nz_recorded = read_training_normalizer(arm)
        nz_reestimated = float(T.resolve_normalizer(m, all_emb.to(dev)))
        nz = nz_recorded if nz_recorded is not None else nz_reestimated
        layer.geodesic_normalizer = nz
        entry = {
            "normalizer_used": nz,
            "normalizer_recorded": nz_recorded,
            "normalizer_reestimated_from_final_encoder": nz_reestimated,
        }
        x = all_emb[:4096].to(dev)
        with torch.no_grad():
            z_meas = m.encoder(x)
            layer._probe_z = z_meas
        entry["before"] = saturation(layer, nz)
        entry["ball_radius_with_recorded_normalizer"] = {
            "item_median": float(enc(z_meas, CURV, nz).norm(dim=-1).median()),
            "item_share_gt_0.99": float((enc(z_meas, CURV, nz).norm(dim=-1) > 0.99).float().mean()),
            "code_median": float(enc(layer.get_code_embs(), CURV, nz).norm(dim=-1).median()),
            "code_share_gt_0.99": float((enc(layer.get_code_embs(), CURV, nz).norm(dim=-1) > 0.99).float().mean()),
        }

        # ---- gradient table: zero_grad before EVERY candidate, per candidate
        rec_grad, rec_loss = real_reconstruction_gradient(m, x)
        rows = {}
        for name in names:
            m.zero_grad(set_to_none=True)          # <- the defect this fixes
            z = m.encoder(x)                       # carries grad_fn
            idx, e_sel = selected_rows(layer, z)
            loss = candidate_objectives(z, e_sel, layer, nz)[name]
            loss.backward()
            cb = layer.get_code_embs().grad.detach().clone()
            en = torch.cat([p.grad.reshape(-1) for p in m.encoder.parameters()
                            if p.grad is not None]).detach().clone()
            rows[name] = {
                "loss_value": float(loss.detach()),
                "codebook_grad_norm": float(cb.reshape(-1).norm()),
                "encoder_grad_norm": float(en.norm()),
                "encoder_over_reconstruction": float(en.norm()) / max(float(rec_grad.norm()), 1e-30),
            }
        base = None
        for name in names:
            m.zero_grad(set_to_none=True)
            z = m.encoder(x)
            idx, e_sel = selected_rows(layer, z)
            candidate_objectives(z, e_sel, layer, nz)[name].backward()
            en = torch.cat([p.grad.reshape(-1) for p in m.encoder.parameters()
                            if p.grad is not None]).detach()
            if base is None:
                base = en.clone()
            denom = float(en.norm()) * float(base.norm())
            rows[name]["cosine_encoder_grad_vs_premap"] = (
                float((en * base).sum()) / denom if denom > 0 else float("nan"))
        rows["_reconstruction_only"] = {"loss_value": rec_loss,
                                        "encoder_grad_norm": float(rec_grad.norm())}
        entry["gradient"] = rows

        # ---- one CONTROLLED step per parameter group, measured, not inferred.
        # AdamW's step is not a function of the gradient norm, so the effect is
        # measured by the change it produces, separately for encoder and codebook.
        c = layer.geodesic_curvature
        lr = 1e-3
        for name in names:
            for group in ("codebook", "encoder"):
                m.load_state_dict(blob["state_dict"])
                layer.geodesic_normalizer = nz
                with torch.no_grad():
                    z0 = m.encoder(x)
                    _, e0 = selected_rows(layer, z0)
                    d0 = ball_dist(enc(z0[0:1], c, nz), enc(e0[0:1], c, nz), c)
                    r0 = float(enc(z0, c, nz).norm(dim=-1).median())
                params = [layer.get_code_embs()] if group == "codebook" else list(m.encoder.parameters())
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
                m.zero_grad(set_to_none=True)
                z = m.encoder(x)
                idx, e_sel = selected_rows(layer, z)
                candidate_objectives(z, e_sel, layer, nz)[name].backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                with torch.no_grad():
                    z1 = m.encoder(x)
                    _, e1 = selected_rows(layer, z1)
                    d1 = ball_dist(enc(z1[0:1], c, nz), enc(e1[0:1], c, nz), c)
                    r1 = float(enc(z1, c, nz).norm(dim=-1).median())
                    finite = bool(torch.isfinite(z1).all())
                rows[name][f"step_{group}"] = {
                    "ball_dist_before": d0, "ball_dist_after": d1, "ball_dist_delta": d1 - d0,
                    "item_radius_median_before": r0, "item_radius_median_after": r1,
                    "finite": finite,
                }
        report[arm] = entry

    print(json.dumps(report, indent=2))
    out = STAGE2 / "geometry_objective_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
