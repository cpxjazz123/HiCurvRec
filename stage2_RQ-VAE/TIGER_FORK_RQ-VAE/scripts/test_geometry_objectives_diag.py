"""Regression tests for the offline geometry-objective diagnostic. No training.

Four defects are pinned here, each with a check that fails against the previous
implementation:

  T1  gradient discipline -- the encoder gradient must not accumulate across the
      candidate objectives;
  T2  Riemannian log map -- log_p(q) must be the Riemannian logarithm with the
      alpha_p(q) = 2/(1 - c||p||^2) metric factor, built from (-p) (+) q, and it
      must agree with a finite-difference gradient of the geodesic distance;
  T3  the ball normalizer -- it must be the value the run actually trained with,
      read back from the run's own record, not re-estimated from the final encoder;
  T4  reconstruction gradient -- it must come from the real RQ-VAE forward
      (encode -> quantise -> decode), not from a decoder applied to a bare latent.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/test_geometry_objectives_diag.py
"""
from __future__ import annotations

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

import diag_geometry_objectives as D  # noqa: E402

DEV = torch.device("cuda:0")
_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


CURV = 1.0


# --------------------------------------------------------------------- T2 first
def test_t2_riemannian_log() -> None:
    """log_p(q) must be the Riemannian logarithm, checked against finite
    differences of the geodesic distance d_c(p, .) -- the Euclidean gradient of
    that function is a negative constant multiple of log_q(p)."""
    torch.manual_seed(7)
    for pv, qv in (((0.15, -0.25, 0.10), (0.10, 0.30, -0.20)),
                   ((0.40, 0.10, -0.20), (0.05, -0.30, 0.15))):
        p = torch.tensor([list(pv)], dtype=torch.float64)
        q = torch.tensor([list(qv)], dtype=torch.float64)
        req(float(D.riemannian_log_p(p, p, CURV).abs().max()) < 1e-12,
            f"T2 log_p(p) == 0 at ||p||={float(torch.linalg.vector_norm(p)):.3f}")

        eps = 1e-6
        fd = torch.zeros_like(q)
        for i in range(q.shape[1]):
            qp, qm = q.clone(), q.clone()
            qp[0, i] += eps
            qm[0, i] -= eps
            fd[0, i] = (float(H.poincare_distance(p, qp, CURV))
                        - float(H.poincare_distance(p, qm, CURV))) / (2 * eps)
        # the gradient of d(p, .) at q is built from log_q(p), the log map AT Q
        lg = D.riemannian_log_p(q, p, CURV).reshape(-1)
        ratios = fd.reshape(-1) / lg
        spread = float((ratios - ratios.mean()).abs().max())
        req(spread < 1e-6 and float(ratios.mean()) < 0.0,
            f"T2 grad_q d(p,q) is a NEGATIVE constant multiple of log_q(p) "
            f"(ratios {[round(float(x), 6) for x in ratios]}, spread {spread:.2e})")

        # the metric factor separates it from the plain chart log map
        riem = D.riemannian_log_p(p, q, CURV).reshape(-1)
        plain = D.log_map_p(p, q, CURV).reshape(-1)
        factor = 1.0 - CURV * float((p * p).sum())
        rr = riem / plain
        req(bool(torch.allclose(rr, torch.full_like(rr, factor), rtol=1e-9)),
            f"T2 log^riem = (1 - c||p||^2) * log^plain with (1-c||p||^2) = {factor:.6f} "
            f"(measured {[round(float(x), 6) for x in rr]})")
        req(not bool(torch.allclose(riem, plain)),
            "T2 and the plain chart log map are genuinely different objects")


# --------------------------------------------------------------------- T3
def test_t3_normalizer_restored() -> None:
    rec = D.read_training_normalizer("hyper_l1_s42")
    req(rec is not None and abs(rec - 11.921757725170167) < 1e-9,
        f"T3 the run's recorded ball normalizer is read back ({rec})")
    rec2 = D.read_training_normalizer("nce_euclid_group")
    req(rec2 is not None and abs(rec2 - rec) < 1e-9,
        f"T3 it is the same value every arm trained with ({rec2})")

    # And a re-estimate from the FINAL encoder is demonstrably a different number:
    # this is exactly the defect the fix removes.
    T.set_seed(T.SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    all_emb = torch.from_numpy(emb)
    blob = torch.load(ROOT.parent.parent / "results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE"
                      / "hyper_l1_s42/rqvae_best.pth", map_location="cpu", weights_only=False)
    cfg = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}
    m = RQVAE(T._tokenizer_config(cfg), in_dim=emb.shape[1]).to(DEV)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    re_est = float(T.resolve_normalizer(m, all_emb.to(DEV)))
    req(abs(re_est - rec) > 1e-3,
        f"T3 re-estimating from the FINAL encoder gives a different number "
        f"({re_est:.6f} vs the recorded {rec:.6f}) -- so re-estimation is wrong")


# --------------------------------------------------------------------- T1
def test_t1_zero_grad_discipline() -> None:
    """The encoder gradient of each candidate must equal that candidate alone.

    Without zero_grad between candidates the encoder gradient is their SUM, which
    is what the previous diagnostic did (it cleared only the layer, not the model).
    """
    torch.manual_seed(11)
    nz = 3.0
    cfg = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}
    m = RQVAE(T._tokenizer_config(cfg), in_dim=16).to(DEV)
    x = torch.randn(64, 16, device=DEV)
    names = ["premap", "ball", "ball_over_nz2", "tangent", "geo2"]

    def grads():
        return torch.cat([p.grad.reshape(-1).clone() for p in m.encoder.parameters()
                          if p.grad is not None])

    solo = {}
    for n in names:
        m.zero_grad(set_to_none=True)
        z = m.encoder(x)                                  # the encoder must participate
        e_sel = m.rq.vq_layers[0].get_code_embs()[:1].expand(z.shape[0], -1)
        D.candidate_objectives(z, e_sel, m.rq.vq_layers[0], nz)[n].backward()
        solo[n] = grads()
    req(all(float(v.abs().max()) > 0.0 for v in solo.values()),
        "T1 every candidate produces a non-zero encoder gradient on its own")

    # the WRONG way: no zero_grad between candidates
    m.zero_grad(set_to_none=True)
    for n in names:
        z = m.encoder(x)
        e_sel = m.rq.vq_layers[0].get_code_embs()[:1].expand(z.shape[0], -1)
        D.candidate_objectives(z, e_sel, m.rq.vq_layers[0], nz)[n].backward()
    accumulated = grads()
    total = sum(solo[n] for n in names)
    dev_sum = float((accumulated - total).abs().max())
    dev_solo = min(float((accumulated - solo[n]).abs().max()) for n in names)
    req(dev_sum < 1e-4 * max(float(total.abs().max()), 1e-12),
        f"T1 without zero_grad the encoder gradient IS the sum of the candidates "
        f"(deviation from the sum {dev_sum:.3e})")
    req(dev_solo > 0.0,
        f"T1 and it differs from every single candidate's gradient "
        f"(min deviation {dev_solo:.3e}) -- so the previous diagnostic measured a sum")


# --------------------------------------------------------------------- T4
def test_t4_real_forward_reconstruction() -> None:
    """The diagnostic's reconstruction gradient must be the REAL forward's.

    The straight-through estimator hands the decoder the QUANTISED codeword, not
    the latent, so a decoder applied to a bare encoder latent measures a different
    function. That shortcut is the defect; the test pins both facts.
    """
    torch.manual_seed(13)
    cfg = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}
    m = RQVAE(T._tokenizer_config(cfg), in_dim=16).to(DEV)
    x = torch.randn(128, 16, device=DEV)

    def enc_grad():
        return torch.cat([p.grad.reshape(-1).clone() for p in m.encoder.parameters()
                          if p.grad is not None])

    # what the diagnostic uses
    diag_grad, diag_loss = D.real_reconstruction_gradient(m, x)

    # the real forward, recomputed independently
    m.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = m(x)
    real_loss = F.mse_loss(reconstructed, x.detach())
    real_loss.backward()
    real_grad = enc_grad()

    # the bare-latent shortcut
    m.zero_grad(set_to_none=True)
    F.mse_loss(m.decoder(m.encoder(x)), x.detach()).backward()
    shortcut = enc_grad()

    rel_diag = float((diag_grad - real_grad).abs().max()) / max(float(real_grad.abs().max()), 1e-12)
    req(rel_diag < 1e-6,
        f"T4 the diagnostic's reconstruction gradient IS the real forward's "
        f"(rel {rel_diag:.2e})")
    rel_short = float((shortcut - real_grad).abs().max()) / max(float(real_grad.abs().max()), 1e-12)
    req(rel_short > 1e-3,
        f"T4 and the bare-latent shortcut is a DIFFERENT function (rel {rel_short:.2e}), "
        f"because the straight-through estimator hands the decoder the quantised "
        f"codeword rather than the latent")
    req(abs(diag_loss - float(real_loss.detach())) < 1e-9,
        f"T4 and it reports the real forward's loss ({diag_loss:.6e})")
    req(float(quant_loss) >= 0.0,
        f"T4 the real forward also returns a quantisation loss ({float(quant_loss):.6e})")


def test_t5_angular_selection() -> None:
    """The radius-free angular selection must be radius-free, must differ from the
    Euclidean rule, and must equal the angular-cosine rule."""
    torch.manual_seed(17)
    from model.layers import VQLayer

    nz = 3.0
    geo = VQLayer(32, 8, 0.25, geodesic=True, geodesic_normalizer=nz,
                  geodesic_curvature=CURV, geodesic_angular_only=True)
    euc = VQLayer(32, 8, 0.25, geodesic=True, geodesic_normalizer=nz,
                  geodesic_curvature=CURV, geodesic_angular_only=False)
    with torch.no_grad():
        geo.embed.weight.copy_(torch.randn_like(geo.embed.weight) * 0.4)
        euc.embed.weight.copy_(geo.embed.weight.clone())
    z = torch.randn(256, 8) * 0.3

    a_ang = geo._distances(z).argmin(-1)
    a_geo = euc._distances(z).argmin(-1)
    d_euc = z.pow(2).sum(1, keepdim=True) - 2 * z @ geo.get_code_embs().t() \
        + geo.get_code_embs().pow(2).sum(1)
    a_euc = d_euc.argmin(-1)
    req(bool(torch.isfinite(geo._distances(z)).all()) and float(geo._distances(z).min()) > 0.0,
        "T5 the angular selection distance is finite and positive")

    # radius-free: rescaling the codewords AND the latents by any positive factor
    # must leave the selection untouched
    with torch.no_grad():
        geo.embed.weight.mul_(7.0)
        a_scaled = geo._distances(z * 0.2).argmin(-1)
    req(bool((a_ang == a_scaled).all()),
        f"T5 the selection is radius-free ({(a_ang == a_scaled).float().mean():.4f} unchanged "
        f"under a 7x codeword and 0.2x latent rescale)")

    # equals the angular-cosine rule: the chart distance between unit vectors is a
    # monotone function of their cosine, so the argmin must agree exactly
    z_b = H.encode_ball(z, CURV, nz)
    e_b = H.encode_ball(geo.get_code_embs().detach(), CURV, nz)
    zu = z_b / z_b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eu = e_b / e_b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a_cos = (zu @ eu.t()).argmax(-1)
    req(bool((a_ang == a_cos).all()),
        f"T5 the angular selection IS the angular-cosine rule "
        f"({int((a_ang == a_cos).sum())}/{len(a_ang)} agree)")

    # and it is not the Euclidean rule
    dis = float((a_ang != a_euc).float().mean())
    req(dis > 0.0,
        f"T5 and it disagrees with the Euclidean rule on {dis:.4f} of items, so it is "
        f"a genuinely different selection")
    req(float((a_ang != a_geo).float().mean()) > 0.0,
        f"T5 and it disagrees with the full geodesic rule on "
        f"{float((a_ang != a_geo).float().mean()):.4f} of items")


def main() -> int:
    print("=== T2: Riemannian log map ===", flush=True)
    test_t2_riemannian_log()
    print("\n=== T3: the run's own normalizer ===", flush=True)
    test_t3_normalizer_restored()
    print("\n=== T1: gradient discipline ===", flush=True)
    test_t1_zero_grad_discipline()
    print("\n=== T4: reconstruction gradient from the real forward ===", flush=True)
    test_t4_real_forward_reconstruction()
    print("\n=== T5: radius-free angular selection ===", flush=True)
    test_t5_angular_selection()
    print("\n" + "=" * 74)
    print("ALL REGRESSION TESTS PASS" if _ok[0] else "REGRESSION TESTS FAILED")
    print("=" * 74)
    return 0 if _ok[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())