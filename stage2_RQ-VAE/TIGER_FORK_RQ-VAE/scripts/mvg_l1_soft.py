"""MVG for the L1 soft-assignment loss, under the FULL training objective.

An earlier version of this probe optimised the soft-assignment term ALONE and
every formulation collapsed: the overlap form to a uniform softmax, the
anchor-codeword form to a single codeword. That probe was not the training
configuration. In real training the reconstruction term is always present and
the encoder cannot collapse without paying for it, so the only question that
matters is whether the assignment sharpens when the soft term is ADDED to the
objective it will actually be added to.

So this probe optimises

    recon_loss + weight * l1_soft_assignment_loss

on a fixed item batch and a fixed pair batch, and reports the L1 sharing rates
on the canonical pairs as they evolve. The mechanism passes if the gap widens
and L1 codebook usage stays high; it fails if either the gap stays flat or the
codebook collapses.
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
model = T.RQVAE(T._tokenizer_config(), in_dim=emb.shape[1]).to(dev)
T.initialize_tiger_weights(model)
with torch.no_grad():
    model.init_codebook(tr_emb.to(dev))

can = T.load_canonical_pairs(T.CANONICAL_PAIRS)
rng = np.random.default_rng(42)
rows = rng.choice(len(can["anchors"]), 512, replace=False)
a = torch.as_tensor(can["anchors"][rows], device=dev)
p = torch.as_tensor(can["positives"][rows], device=dev)
n = torch.as_tensor(can["negatives"][rows], device=dev)
item_batch = tr_emb[:1024].to(dev)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
WEIGHT = T.L1_SOFT_WEIGHT

print(f"weight={WEIGHT}  items={len(item_batch)}  pairs={len(rows)}")
print(f"{'step':>6} {'recon':>10} {'soft':>9} {'p_pos':>9} {'p_neg':>9} "
      f"{'gap':>10} {'L1used':>7} {'k*':>4}")

for step in range(1, 1201):
    # The reconstruction path the quantiser already had, unchanged.
    reconstructed, quant_loss, _, _ = model(item_batch)
    _, recon = model.compute_loss(item_batch, reconstructed, quant_loss)

    al = model.encoder(all_emb[a.cpu()].to(dev))
    pl = model.encoder(all_emb[p.cpu()].to(dev))
    nl = model.encoder(
        all_emb[n.reshape(-1).cpu()].to(dev)
    ).view(len(rows), n.shape[1], -1)
    soft, st = T.l1_soft_assignment_loss(model, al, pl, nl)

    loss = recon + WEIGHT * soft
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step in (1, 25, 100, 250, 500, 800, 1200):
        with torch.no_grad():
            enc = model.encoder(tr_emb.to(dev))
            ind = torch.argmin(model.rq.vq_layers[0]._distances(enc), dim=-1)
            used = int(ind.unique().numel())
        print(f"{step:>6} {float(recon):>10.6f} {float(soft):>9.5f} "
              f"{st['p_share_pos']:>9.6f} {st['p_share_neg']:>9.6f} "
              f"{st['gap']:>+10.6f} {used:>7} "
              f"{st.get('n_l1_codewords_targeted', 0):>4}")

with torch.no_grad():
    enc = model.encoder(tr_emb.to(dev))
    ind = torch.argmin(model.rq.vq_layers[0]._distances(enc), dim=-1)
    used = int(ind.unique().numel())
    al = model.encoder(all_emb[a.cpu()].to(dev))
    pl = model.encoder(all_emb[p.cpu()].to(dev))
    nl = model.encoder(
        all_emb[n.reshape(-1).cpu()].to(dev)
    ).view(len(rows), n.shape[1], -1)
    _, st = T.l1_soft_assignment_loss(model, al, pl, nl)
print(f"\nfinal gap {st['gap']:+.6f}  L1 codewords used {used}/256")
print("PASS" if (st["gap"] > 0 and used > 200) else "FAIL")
