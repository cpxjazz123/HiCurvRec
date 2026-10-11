"""MVG for the L1 codeword-SELECTION geodesic (hyperbolic programme, step 2).

Registered mechanism: the ONLY change against the plain recipe is that the L1
codeword-selection distance is the Poincare geodesic instead of the squared
Euclidean one. The residual chain, the codebook capacity, the training budget,
the data split and Stage 3 are untouched; L2 and L3 keep the Euclidean distance.

This probe verifies implementation, not merit:
  1. the flag really is set on level 0 and nowhere else;
  2. the curvature is the registered constant and the normalizer is the run
     constant, i.e. the map is batch-independent;
  3. the returned matrix is finite and the level-0 layer's distance really is
     the geodesic (it differs from the Euclidean one on the same latents);
  4. the intervention has a measurable computational effect: some items change
     their L1 index relative to an identical Euclidean model;
  5. the training objective's gradient reaches both the L1 codebook and the
     encoder, and the total loss has a live grad_fn.
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


l0, l1, l2 = model.rq.vq_layers
req(getattr(l0, "geodesic", False) is True, "L1 selection distance is the Poincare geodesic")
req(
    getattr(l1, "geodesic", False) is False and getattr(l2, "geodesic", False) is False,
    "L2/L3 keep the Euclidean selection distance",
)
req(
    abs(float(l0.geodesic_curvature) - float(T.L1_GEODESIC_CURVATURE)) < 1e-12
    and abs(float(T.CURVATURE) - 1.0) < 1e-12,
    f"curvature fixed: L1={float(l0.geodesic_curvature)} global={float(T.CURVATURE)}",
)
req(
    abs(float(l0.geodesic_normalizer) - float(normalizer)) < 1e-9,
    f"L1 normalizer is the run constant {float(normalizer):.6f}",
)

with torch.no_grad():
    enc = model.encoder(tr_emb.to(dev))
    d_head = l0._distances(enc[:64])
    idx = torch.cat([torch.arange(64), torch.arange(4000, 5000)], dim=0)
    d_head_b = l0._distances(enc[idx])[:64]
    d_l2 = l1._distances(enc[:64])
    manual = (
        enc[:64].pow(2).sum(1, keepdim=True)
        - 2 * enc[:64] @ l1.get_code_embs().t()
        + l1.get_code_embs().pow(2).sum(1)
    )
req(torch.isfinite(d_head).all() and torch.isfinite(d_head_b).all(),
    "L1 geodesic distance matrix is finite")
req(float((d_head - d_head_b).abs().max()) < 1e-6,
    "L1 distance is batch-independent (max |delta| < 1e-6 over a re-batched set)")
req(float((d_l2 - manual).abs().max()) < 1e-4,
    "L2 distance still equals the squared Euclidean formula")

eu = T.RQVAE(T._tokenizer_config(None), in_dim=emb.shape[1]).to(dev)
eu.load_state_dict(model.state_dict())
with torch.no_grad():
    e2 = eu.encoder(tr_emb.to(dev))
    i_geo = l0._distances(enc).argmin(-1)
    i_euc = eu.rq.vq_layers[0]._distances(e2).argmin(-1)
changed = int((i_geo != i_euc).sum())
req(changed > 0,
    f"L1 assignment changes for {changed}/{len(i_geo)} items vs an identical Euclidean model")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch = tr_emb[:1024].to(dev)
reconstructed, quant_loss, _, _ = model(batch)
_, recon = model.compute_loss(batch, reconstructed, quant_loss)
loss = recon + quant_loss
opt.zero_grad()
loss.backward()
req(loss.requires_grad and loss.grad_fn is not None, "total_loss has a live grad_fn")
cb_grad = l0.embed.weight.grad
req(cb_grad is not None and float(cb_grad.abs().sum()) > 0.0,
    "L1 codebook receives a non-zero gradient")
enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
enc_nonzero = any(float(g.abs().sum()) > 0.0 for g in enc_grads)
req(enc_nonzero, "encoder receives a non-zero gradient through the quantizer")
req(all(torch.isfinite(g).all() for g in enc_grads) and torch.isfinite(cb_grad).all(),
    "gradients are finite")

print("\nMVG " + ("PASS" if _ok[0] else "FAIL"), flush=True)
sys.exit(0 if _ok[0] else 1)
