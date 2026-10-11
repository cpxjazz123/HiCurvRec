"""MVG for the cluster-consistency L1 term, under the full objective.

The question is whether pulling each item's L1 soft assignment toward its
behaviour group actually moves the assignment, and whether it does so without
the collapse that killed step 3. Step 3 failed because its target,
argmax q_a, was a quantity the encoder controlled and could drive to one
codeword. Here the target is the cluster index, fixed in advance and spread over
199 of the 256 codewords, so the mechanism has no attractor at a single entry.

Three numbers decide it:

  target_prob          mean q_i[cluster(i)], which the loss raises directly
  frac_on_target       share of items whose ARGMAX L1 codeword is already their
                       cluster index; this is the discrete claim
  L1 codewords used    must stay high. A term that raises the target probability
                       by collapsing the codebook is exactly the failure mode the
                       programme's step 7 forbids.

Optimised as it will actually be trained: reconstruction plus the cluster term.
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

cluster = np.load(T.BEHAVIOUR_CLUSTER_PATH)["cluster"].astype(np.int64)
item_batch = tr_emb[:2048].to(dev)
batch_ids = ids[:2048]
tgt = torch.as_tensor(cluster[batch_ids], dtype=torch.long, device=dev)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
W = T.CLUSTER_WEIGHT

print(f"weight={W}  clusters={len(np.unique(cluster))}  items={len(batch_ids)}")
print(f"{'step':>6} {'recon':>10} {'cl_loss':>9} {'d_tgt':>9} "
      f"{'d_other':>9} {'viol':>7} {'L1used':>7}")

for step in range(1, 1201):
    reconstructed, quant_loss, _, _ = model(item_batch)
    _, recon = model.compute_loss(item_batch, reconstructed, quant_loss)
    lat = model.encoder(item_batch)
    cl, st = T.cluster_assign_loss(model, lat, tgt)
    loss = recon + W * cl
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step in (1, 25, 100, 250, 500, 800, 1200):
        with torch.no_grad():
            enc = model.encoder(tr_emb.to(dev))
            ind = torch.argmin(model.rq.vq_layers[0]._distances(enc), dim=-1)
            used = int(ind.unique().numel())
        print(f"{step:>6} {float(recon):>10.6f} {float(cl):>9.5f} "
              f"{st['mean_d_target']:>9.6f} {st['mean_d_best_other']:>9.6f} "
              f"{st['frac_violating']:>7.4f} {used:>7}")

with torch.no_grad():
    enc = model.encoder(tr_emb.to(dev))
    ind = torch.argmin(model.rq.vq_layers[0]._distances(enc), dim=-1)
    used = int(ind.unique().numel())
    _, st = T.cluster_assign_loss(model, model.encoder(item_batch), tgt)
with torch.no_grad():
    d2 = (model.encoder(item_batch).pow(2).sum(-1, keepdim=True)
          - 2 * model.encoder(item_batch) @ model.rq.vq_layers[0].get_code_embs().detach().t()
          + model.rq.vq_layers[0].get_code_embs().detach().pow(2).sum(-1).unsqueeze(0))
    hard = d2.argmin(dim=-1)
    on_target = float((hard == tgt).float().mean())
print(f"\nfinal on_target {on_target:.4f}  violating {st['frac_violating']:.4f}  "
      f"L1 used {used}/256")
print("PASS" if (on_target > 0.5 and used > 200) else "FAIL")