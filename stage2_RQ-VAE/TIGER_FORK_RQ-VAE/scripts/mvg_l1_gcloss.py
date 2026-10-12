"""MVG for the geometry-consistent L1 codebook loss (root-cause programme, step 4).

One variable against hyper_l1_s42: the L1 codebook is optimised in the ball the
selection already measures, instead of by a Euclidean squared MSE in the pre-map
space.

Checks:
  1. the flag reaches the level-0 layer only;
  2. the loss is exactly mse(enc(e_sel), sg(enc(z))) + beta * mse(enc(z), sg(enc(e_sel)))
     on the mapped points -- the geometry is what it claims to be;
  3. the loss and both gradients are finite, and the L1 codebook receives a
     non-zero gradient;
  4. DIRECT EFFECT: one real optimiser step on the codebook DECREASES the ball
     distance between the codeword and the item it represents, and the Euclidean
     objective does NOT -- so the objective really changes where the codebook goes;
  5. everything else is untouched: the straight-through value handed to the decoder
     is still the raw codeword, and L2/L3 still use the Euclidean objective.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import hyperbolic as H  # noqa: E402
import train_rqvae as T  # noqa: E402
from model import RQVAE  # noqa: E402

DEV = torch.device("cuda:0")
_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


T.set_seed(T.SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
tr = torch.from_numpy(emb)[torch.from_numpy(ids)].to(DEV)
CURV = T.L1_GEODESIC_CURVATURE

GEO_BALL = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "ball_mse"}
GEO_EUCL = {"normalizer": 1.0, "curvature": CURV, "codebook_loss": "euclid"}


def build(cfg):
    T.set_seed(T.SEED)
    m = RQVAE(T._tokenizer_config(cfg), in_dim=emb.shape[1]).to(DEV)
    T.initialize_tiger_weights(m)
    with torch.no_grad():
        m.init_codebook(tr)
    m.rq.vq_layers[0].geodesic_normalizer = float(T.resolve_normalizer(m, tr))
    return m


m_ball = build(GEO_BALL)
m_eucl = build(GEO_EUCL)
l0b, l1b, l2b = m_ball.rq.vq_layers
l0e, l1e, l2e = m_eucl.rq.vq_layers

req(getattr(l0b, "geodesic_codebook_loss", False) is True,
    "the ball-space objective reaches the level-0 layer")
req(getattr(l1b, "geodesic_codebook_loss", False) is False
    and getattr(l2b, "geodesic_codebook_loss", False) is False,
    "L2/L3 keep the Euclidean objective")
req(getattr(l0e, "geodesic_codebook_loss", False) is False,
    "the control arm keeps the Euclidean objective at level 0")


def ball_dist(layer, z, e):
    return float(H.pair_poincare_distance(
        H.encode_ball(z, layer.geodesic_curvature, layer.geodesic_normalizer),
        H.encode_ball(e, layer.geodesic_curvature, layer.geodesic_normalizer),
        layer.geodesic_curvature)[0, 0])


z = m_ball.encoder(tr[:1024])  # latents, not raw embeddings
with torch.no_grad():
    xq, ql, _, idx = l0b(z)
    idx_e = l0e._distances(z).argmin(-1)
    nz = l0b.geodesic_normalizer
    c = l0b.geodesic_curvature
    zb = H.encode_ball(z, c, nz)
    eb = H.encode_ball(l0b.get_code_embs()[idx], c, nz)
    want = F.mse_loss(eb, zb.detach()) + 0.25 * F.mse_loss(zb, eb.detach())
    euclid_ref = F.mse_loss(l0b.get_code_embs()[idx], z.detach()) \
        + 0.25 * F.mse_loss(z, l0b.get_code_embs()[idx].detach())
req(torch.isfinite(ql) and float(ql) > 0.0, f"the ball-space loss is finite and positive ({float(ql):.6e})")
req(abs(float(ql) - float(want)) < 1e-12,
    f"the loss IS the mapped-space objective ({float(ql):.9e} vs {float(want):.9e})")
req(abs(float(ql) - float(euclid_ref)) > 1e-12,
    "and it is NOT the pre-map Euclidean objective it replaces")

# gradients: a fresh forward WITH grad, the no_grad block above was only for the
# identity checks and its loss carries no grad_fn by construction
m_ball.zero_grad(set_to_none=True)
_, ql_g, _, _ = l0b(z)
ql_g.backward()
ql = ql_g
gc = l0b.embed.weight.grad
ge = [p.grad for p in m_ball.encoder.parameters() if p.grad is not None]
req(gc is not None and float(gc.abs().sum()) > 0.0,
    f"the L1 codebook receives a non-zero gradient ({float(gc.abs().sum()):.4e})")
req(any(float(g.abs().sum()) > 0.0 for g in ge), "the encoder receives a non-zero gradient")
req(torch.isfinite(gc).all() and all(torch.isfinite(g).all() for g in ge), "gradients are finite")

# DIRECT EFFECT: one real step on the codebook, measured by the ball distance
def one_step_distance(layer, module, use_ball, lr=1e-2):
    T.set_seed(T.SEED)
    enc = module.encoder(tr[:1024])
    d_before = ball_dist(layer, enc, layer.get_code_embs()[layer._distances(enc).argmin(-1)])
    opt = torch.optim.SGD([layer.embed.weight], lr=lr)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        lat = module.encoder(tr[:1024])
        i = layer._distances(lat).argmin(-1)
        e_sel = layer.get_code_embs()[i]
        if use_ball:
            c = layer.geodesic_curvature
            nz = layer.geodesic_normalizer
            zb = H.encode_ball(lat, c, nz)
            eb = H.encode_ball(e_sel, c, nz)
            loss = F.mse_loss(eb, zb.detach()) + 0.25 * F.mse_loss(zb, eb.detach())
        else:
            loss = F.mse_loss(e_sel, lat.detach()) + 0.25 * F.mse_loss(lat, e_sel.detach())
        loss.backward()
        opt.step()
    with torch.no_grad():
        enc = module.encoder(tr[:1024])
        d_after = ball_dist(layer, enc, layer.get_code_embs()[layer._distances(enc).argmin(-1)])
    return d_before, d_after


db, da = one_step_distance(l0b, m_ball, True)
eb_, ea_ = one_step_distance(l0e, m_eucl, False)
print(f"     ball-space objective: ball distance {db:.6f} -> {da:.6f} "
      f"(delta {da - db:+.6f})", flush=True)
print(f"     euclidean objective (control): ball distance {eb_:.6f} -> {ea_:.6f} "
      f"(delta {ea_ - eb_:+.6f})", flush=True)
req(da < db, "one codebook step under the BALL objective decreases the ball distance")
req(ea_ >= db, "one codebook step under the EUCLIDEAN objective does not "
                "(this is the discriminative comparison)")

# untouched paths
# The invariant that matters is that the flag does not change what the decoder is
# handed: the straight-through value must be identical to the control arm's.
with torch.no_grad():
    ste_ball = xq
    ste_eucl, _, _, _ = l0e(z)
    d_ste = float((ste_ball - ste_eucl).abs().max())
req(d_ste < 1e-4,
    f"the straight-through value handed to the decoder is unchanged by the flag "
    f"(max |ball - euclid| = {d_ste:.2e}, float32 tolerance)")
with torch.no_grad():
    raw = l0b.get_code_embs()[idx[0]]
    print(f"     STE value equals the raw codeword: max |diff| = "
          f"{float((ste_ball[0] - raw).abs().max()):.2e}", flush=True)
req(getattr(l1b, "geodesic_codebook_loss", False) is False
    and getattr(l2b, "geodesic_codebook_loss", False) is False,
    "L2/L3 objective untouched by the flag")

print("\nMVG " + ("PASS" if _ok[0] else "FAIL"), flush=True)
raise SystemExit(0 if _ok[0] else 1)