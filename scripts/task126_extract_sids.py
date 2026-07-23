"""
Extract semantic IDs from trained phonism RQ-VAE checkpoint.

Loads embeddings directly from the cached parquet file (avoids re-encoding),
runs RQ-VAE encode -> quantize on all items, saves SID matrix.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/genrec")

from genrec.models.rqvae import RqVae, QuantizeForwardMode

CKPT_PATH = "/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys/rqvae/checkpoint_19999.pt"
PARQUET_PATH = "/home/wlia0047/ar57/wenyu/genrec/dataset/amazon/processed/toys/item_emb_sentence-t5-base.parquet"
OUT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task126_phonism_rqvae/semantic_ids.pt"
DEVICE = "cuda:0"

# RQ-VAE config (must match training config)
vae_input_dim = 768
vae_embed_dim = 32
vae_hidden_dims = [512, 256, 128, 64]
vae_codebook_size = 256
vae_n_layers = 3
commitment_weight = 0.25

print("Building RQ-VAE model...")
model = RqVae(
    input_dim=vae_input_dim,
    embed_dim=vae_embed_dim,
    hidden_dims=vae_hidden_dims,
    codebook_size=vae_codebook_size,
    codebook_kmeans_init=False,
    codebook_normalize=False,
    codebook_sim_vq=False,
    codebook_mode=QuantizeForwardMode.STE,
    codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
    n_layers=vae_n_layers,
    n_cat_features=0,
    commitment_weight=commitment_weight,
)
print(f"Loading checkpoint from {CKPT_PATH}")
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
if "model" in state:
    model.load_state_dict(state["model"])
else:
    model.load_state_dict(state)
print(f"Checkpoint loaded (iter {state.get('iter', 'unknown')})")

model = model.to(DEVICE)
model.eval()

print(f"Loading embeddings from {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
df = df.sort_values("ItemID").reset_index(drop=True)
print(f"  Loaded {len(df)} items, embedding dim={len(df.iloc[0]['embedding'])}")
embeddings = np.stack(df["embedding"].values).astype(np.float32)
item_ids = df["ItemID"].values

# Run RQ-VAE encode + quantize in batches
batch_size = 1024
all_sids = []
with torch.no_grad():
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = torch.tensor(embeddings[start:end], dtype=torch.float32, device=DEVICE)
        out = model.get_semantic_ids(batch, gumbel_t=0.001)
        sids = out.sem_ids.cpu()  # (batch, n_layers)
        all_sids.append(sids)

# Concat rows from all batches (handles variable batch sizes)
semantic_ids_rows = []
for t in all_sids:
    semantic_ids_rows.extend(t.unbind(0))
semantic_ids = torch.stack(semantic_ids_rows, dim=0)  # (N, n_layers)
print(f"semantic_ids shape: {semantic_ids.shape}")
print(f"Unique SIDs per layer:")
for i in range(vae_n_layers):
    print(f"  Layer {i}: {len(semantic_ids[:, i].unique())} / {vae_codebook_size}")

n_unique_total = len(set(map(tuple, semantic_ids.tolist())))
print(f"Total unique 3-token SIDs: {n_unique_total} / {len(df)} = {n_unique_total/len(df):.4f}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save({
    "semantic_ids": semantic_ids,
    "item_ids": torch.tensor(item_ids, dtype=torch.long),
    "checkpoint_iter": state.get("iter", "unknown"),
    "n_items": len(df),
    "n_layers": vae_n_layers,
    "codebook_size": vae_codebook_size,
}, OUT_PATH)
print(f"Saved to {OUT_PATH}")