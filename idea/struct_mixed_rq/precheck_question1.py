"""question1: does the structure term move the hard SID at all?  CPU pre-check."""
import json, random, sys
sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_struct_mixed_rq")
import numpy as np, pandas as pd, torch
import train_rqvae as T, structure as S, training_hooks as H
from model.model import RQVAE

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
S_SCALE, LAMBDA = 0.041920, 3.5211e-06
BATCH, STEPS, CHECK = 512, 100, (50, 100)

emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
items = torch.from_numpy(emb).float()
model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
if T.XAVIER_INIT: T.initialize_tiger_weights(model)
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
with torch.no_grad(): model.init_codebook(items[torch.from_numpy(ids[:2048])])
batch = items[torch.from_numpy(ids[:BATCH])]
graph = S.build_graph(T.TRAIN_FILE, seed=T.SEED)

edges = []
for node, neighbours in graph.neighbours.items():
    for other in neighbours:
        if node < other: edges.append((node, other))
edges = np.array(edges[:200000])
# edges carry global item ids; the tokens are indexed by batch row, so keep only
# the edges whose both endpoints are in this batch and remap them
position = {int(gid): row for row, gid in enumerate(ids[:BATCH])}
kept = [(position[a], position[b]) for a, b in edges
        if int(a) in position and int(b) in position]
edge_index = torch.tensor(kept, dtype=torch.long).T if kept else torch.zeros(2, 0, dtype=torch.long)

opt = torch.optim.AdamW(model.parameters(), lr=float(T.LR), weight_decay=float(T.WEIGHT_DECAY))
rng = random.Random(T.SEED + 1000)

def tokens(infer):
    with torch.no_grad(): return model.get_indices(batch, infer_use_sk=infer).clone()

def prefix_stats(tok):
    out = {}
    for level, name in enumerate(("L1", "L2", "L3")):
        keys = [tuple(row[: level + 1].tolist()) for row in tok]
        counts = {}
        for key in keys: counts[key] = counts.get(key, 0) + 1
        same = sum(c for c in counts.values() if c > 1) / len(keys)
        dup_item = sum(c for c in counts.values() if c > 1) / len(keys)
        distinct_measure = 1.0 - len(counts) / len(keys)
        out[name] = {"distinct_prefixes": len(counts),
                     "same_prefix_item_share": round(same, 6),
                     "l3_collision_item_weighted": round(dup_item, 6),
                     "l3_collision_one_minus_distinct": round(distinct_measure, 6)}
    return out

def edge_agreement(tok):
    out = {}
    for level, name in enumerate(("L1", "L2", "L3")):
        grouped = tok[:, : level + 1]
        same = (grouped[edge_index[0]] == grouped[edge_index[1]]).all(dim=1)
        out[name] = round(float(same.float().mean()), 6)
    return out

base_tok_export = tokens(False); base_tok_sk = tokens(True)
cb = [l.get_code_embs() for l in model.rq.vq_layers]
cb0 = [p.detach().clone() for p in cb]
model.train()
recon0 = None; clip = 0; report = {}
for step in range(1, STEPS + 1):
    recon, quant, _, _ = model.forward(batch)
    base = model.compute_loss(batch, recon, quant)[0]
    if recon0 is None: recon0 = float(((recon - batch) ** 2).mean().detach())
    term = H.structure_term(model, items, graph, rng, scale=S_SCALE, size=T.STRUCT_TRIPLES)
    loss = base + LAMBDA * term
    opt.zero_grad(set_to_none=True); loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    clip += int(norm > 1.0)
    opt.step()
    if step in CHECK:
        export = tokens(False); sk = tokens(True)
        recon_now = float(((model.forward(batch)[0] - batch) ** 2).mean().detach())
        report[step] = {
            "changed_tokens_argmin": [int((export[:, l] != base_tok_export[:, l]).sum()) for l in range(3)],
            "changed_tokens_sinkhorn": [int((sk[:, l] != base_tok_sk[:, l]).sum()) for l in range(3)],
            "train_export_branch_diff": [int((export[:, l] != sk[:, l]).sum()) for l in range(3)],
            "prefix": prefix_stats(export),
            "edge_same_prefix": edge_agreement(export),
            "recon_mse": {"step0": round(recon0, 8), "now": round(recon_now, 8),
                          "ratio": round(recon_now / recon0, 6)},
            "codebook_shift": [round(float((p.detach() - q).norm()), 8) for p, q in zip(cb, cb0)],
            "clip_frac": round(clip / step, 6),
            "accept_rate": None,
        }
print(json.dumps({"batch": BATCH, "steps": STEPS, "scale": S_SCALE, "lambda": LAMBDA,
                  "edges_sampled": int(edges.shape[0]),
                  "step0": {"prefix": prefix_stats(base_tok_export),
                            "edge_same_prefix": edge_agreement(base_tok_export),
                            "train_export_branch_diff": [int((base_tok_export[:, l] != base_tok_sk[:, l]).sum()) for l in range(3)]},
                  "checkpoints": report}, indent=1))
