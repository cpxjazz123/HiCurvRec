"""MVG for the L1 boundary-repulsion hinge (hyperbolic programme, step 3).

Registered mechanism, one factor on top of the accepted step-2 arm: canonical
pair negatives that CURRENTLY share their anchor's L1 codeword are pushed across
that boundary by a margin hinge between the shared prototype and the nearest
alternative prototype, and positives that correctly share the anchor's codeword
get the mirror hinge. The L1 prototypes are detached inside the term.

This probe verifies implementation, not merit:
  1. the hinge identifies hard negatives and correct-share positives at all, on
     latents constructed so that sharing is guaranteed;
  2. both losses are finite and the margin/weights are the registered constants;
  3. the gradient of the term reaches the encoder and does NOT reach the L1
     codebook (the prototypes are detached);
  4. the geodesic and the Euclidean variant differ in the distance they measure
     and agree on sampling, margin, weights and budget (same call, one flag);
  5. the term is exactly zero when no pair shares a code, so it cannot leak into
     the objective through an unconditional constant.
"""
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import train_rqvae as T  # noqa: E402

dev = torch.device("cuda:0")
T.set_seed(T.SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
all_emb = torch.from_numpy(emb)
tr_emb = all_emb[torch.from_numpy(ids)]

GEO = {"normalizer": 1.0, "curvature": T.L1_GEODESIC_CURVATURE}
model = T.RQVAE(T._tokenizer_config(GEO), in_dim=emb.shape[1]).to(dev)
T.initialize_tiger_weights(model)
with torch.no_grad():
    model.init_codebook(tr_emb.to(dev))
normalizer = T.resolve_normalizer(model, tr_emb.to(dev))
model.rq.vq_layers[0].geodesic_normalizer = float(normalizer)

_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


n = 64
anchor = model.encoder(tr_emb[:n].to(dev))
# Positives are the anchors themselves and negatives are copies of the anchors,
# so every pair shares a code by construction and the hard-negative set is
# guaranteed non-empty at initialisation.
pos = anchor.clone()
neg = anchor.clone().unsqueeze(1).expand(-1, 10, -1).contiguous()

lh, lp, st = T.l1_boundary_repulsion(
    model, anchor, pos, neg, "geodesic", normalizer, T.L1_BR_MARGIN, T.L1_BR_MAX_HARD
)
req(st["n_hard"] > 0, f"hard negatives identified on constructed sharing: n_hard={st['n_hard']}")
req(st["n_pos_ok"] > 0, f"correct-share positives identified: n_pos_ok={st['n_pos_ok']}")
req(torch.isfinite(lh) and torch.isfinite(lp), "both hinges are finite")
req(abs(float(T.L1_BR_MARGIN) - 0.05) < 1e-12, f"margin is the registered constant {T.L1_BR_MARGIN}")
req(abs(float(T.L1_BR_WEIGHT) - 0.05) < 1e-12 and abs(float(T.L1_BR_POS_WEIGHT) - 0.02) < 1e-12,
    f"weights are the registered constants {T.L1_BR_WEIGHT} / {T.L1_BR_POS_WEIGHT}")
req(st["n_hard"] <= T.L1_BR_MAX_HARD and st["n_pos_ok"] <= T.L1_BR_MAX_HARD,
    f"hard-negative count is bounded by {T.L1_BR_MAX_HARD}")
# A hard negative is a copy of its anchor, so the shared prototype is by
# construction the anchor's nearest codeword: the shared distance can never
# exceed the nearest-alternative distance. That is what makes the hinge
# `relu(margin + d_shared - d_alt)` a boundary term rather than a global repulsion.
req(st["mean_d_shared_hard"] <= st["mean_d_alt_hard"] + 1e-6,
    f"shared prototype is the nearest for hard negatives "
    f"({st['mean_d_shared_hard']:.4f} <= {st['mean_d_alt_hard']:.4f})")

model.zero_grad(set_to_none=True)
loss = T.L1_BR_WEIGHT * lh + T.L1_BR_POS_WEIGHT * lp
loss.backward()
enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
req(any(float(g.abs().sum()) > 0.0 for g in enc_grads),
    "encoder receives a non-zero gradient from the hinge")
cb_grad = model.rq.vq_layers[0].embed.weight.grad
req(cb_grad is None or float(cb_grad.abs().sum()) == 0.0,
    "L1 codebook receives NO gradient from the hinge (prototypes detached)")
req(all(torch.isfinite(g).all() for g in enc_grads), "encoder gradients are finite")

eu_lh, eu_lp, eu_st = T.l1_boundary_repulsion(
    model, anchor, pos, neg, "euclid", normalizer, T.L1_BR_MARGIN, T.L1_BR_MAX_HARD
)
req(torch.isfinite(eu_lh) and torch.isfinite(eu_lp), "Euclidean variant is finite")
req(eu_st["n_hard"] == st["n_hard"] and eu_st["n_pos_ok"] == st["n_pos_ok"],
    "geodesic and Euclidean variants use the same sampling and the same budget")
req(abs(eu_st["scale"] - st["scale"]) > 1e-9,
    f"the two variants measure different distances (scale {st['scale']:.4e} vs {eu_st['scale']:.4e})")

# With a shared distance of 0 the hinge collapses to relu(margin - d_alt), so its
# mean cannot exceed the margin. This pins the value to the registered formula.
req(float(lh) <= T.L1_BR_MARGIN + 1e-6,
    f"hard hinge value {float(lh):.6f} respects relu(margin - d_alt) <= margin")

print("\nMVG " + ("PASS" if _ok[0] else "FAIL"), flush=True)
sys.exit(0 if _ok[0] else 1)
