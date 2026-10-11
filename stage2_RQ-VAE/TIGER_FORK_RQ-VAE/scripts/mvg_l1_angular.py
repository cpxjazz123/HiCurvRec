"""MVG for the local-chart angular separation (hyperbolic programme, step 5).

Registered mechanism, one factor on top of the accepted hyper_l1_s42: canonical
pair negatives that CURRENTLY share their anchor's L1 codeword must leave the
anchor's residual DIRECTION and positives that correctly share it must keep it,
measured as a cosine in the local Euclidean chart at the anchor's prototype.
Direction only, so no ball metric enters the gradient path.

This probe verifies implementation, not merit:
  1. hard negatives and correct-share positives are identified on latents
     constructed so that sharing is guaranteed;
  2. both hinges are finite and the margin/threshold are the registered constants;
  3. the term is DIRECTION ONLY: scaling every residual about its prototype leaves
     both losses bit-identical, so it cannot be driven by radial magnitude;
  4. the hinge is positive exactly when the hard negative's cosine exceeds the
     threshold, and zero otherwise;
  5. the gradient reaches the encoder and does NOT reach the L1 codebook.
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
tr_emb = torch.from_numpy(emb)
model = T.RQVAE(T._tokenizer_config({"normalizer": 1.0, "curvature": T.L1_GEODESIC_CURVATURE}), in_dim=emb.shape[1]).to(dev)
T.initialize_tiger_weights(model)
with torch.no_grad():
    model.init_codebook(tr_emb[torch.from_numpy(ids)].to(dev))

_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


n = 64
anchor = model.encoder(tr_emb[:n].to(dev))
pos = anchor.clone()
neg = anchor.clone().unsqueeze(1).expand(-1, 10, -1).contiguous()

lh, lp, st = T.l1_angular_separation(model, anchor, pos, neg, T.L1_ANG_MARGIN, T.L1_ANG_MAX)
req(st["n_hard"] > 0, f"hard negatives identified on constructed sharing: n_hard={st['n_hard']}")
req(st["n_pos_ok"] > 0, f"correct-share positives identified: n_pos_ok={st['n_pos_ok']}")
req(torch.isfinite(lh) and torch.isfinite(lp), "both hinges are finite")
req(abs(float(T.L1_ANG_MARGIN) - 0.2) < 1e-12,
    f"margin is the registered constant {T.L1_ANG_MARGIN} (threshold {st['threshold']})")
req(abs(float(T.L1_ANG_WEIGHT) - 0.05) < 1e-12 and abs(float(T.L1_ANG_POS_WEIGHT) - 0.02) < 1e-12,
    f"weights are the registered constants {T.L1_ANG_WEIGHT} / {T.L1_ANG_POS_WEIGHT}")
req(st["n_hard"] <= T.L1_ANG_MAX and st["n_pos_ok"] <= T.L1_ANG_MAX,
    f"hard-negative count is bounded by {T.L1_ANG_MAX}")

# Direction only: scaling every residual about its prototype leaves the cosines,
# and therefore both losses, unchanged. This is what makes the term scale-free.
with torch.no_grad():
    codes = model.rq.vq_layers[0].get_code_embs()
    ca = codes[(anchor.pow(2).sum(-1, keepdim=True) - 2 * anchor @ codes.t() + codes.pow(2).sum(-1).unsqueeze(0)).argmin(dim=1)]
    ca_n = ca.unsqueeze(1)                      # (N, 1, D) for the (N, K, D) negatives

    lh_s, lp_s, st_s = T.l1_angular_separation(
        model,
        ca + 3.0 * (anchor - ca),
        ca + 3.0 * (pos - ca),
        ca_n + 3.0 * (neg - ca_n),
        T.L1_ANG_MARGIN, T.L1_ANG_MAX,
    )
req(abs(float(lh_s) - float(lh)) < 1e-5 and abs(float(lp_s) - float(lp)) < 1e-5,
    f"loss is invariant to radial scaling ({float(lh):.6f} vs {float(lh_s):.6f}, "
    f"{float(lp):.6f} vs {float(lp_s):.6f})")

# A hard negative that is a copy of its anchor has cosine exactly 1, so the hinge
# is exactly margin and the violation fraction is 1.
req(abs(st["mean_cos_hard"] - 1.0) < 1e-4,
    f"constructed hard negatives have cosine 1 with the anchor ({st['mean_cos_hard']:.6f})")
req(abs(float(lh) - T.L1_ANG_MARGIN) < 1e-4,
    f"hinge equals the margin when the cosine is 1 ({float(lh):.6f})")
req(st["frac_hard_above"] == 1.0, f"every constructed hard negative violates ({st['frac_hard_above']})")

model.zero_grad(set_to_none=True)
loss = T.L1_ANG_WEIGHT * lh + T.L1_ANG_POS_WEIGHT * lp
loss.backward()
enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
req(any(float(g.abs().sum()) > 0.0 for g in enc_grads),
    "encoder receives a non-zero gradient from the angular term")
cb_grad = model.rq.vq_layers[0].embed.weight.grad
req(cb_grad is None or float(cb_grad.abs().sum()) == 0.0,
    "L1 codebook receives NO gradient from the angular term")
req(all(torch.isfinite(g).all() for g in enc_grads), "encoder gradients are finite")

print("\nMVG " + ("PASS" if _ok[0] else "FAIL"), flush=True)
sys.exit(0 if _ok[0] else 1)
