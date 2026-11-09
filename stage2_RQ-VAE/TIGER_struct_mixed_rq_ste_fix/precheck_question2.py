"""question2: corrected measurement and fixed-state wiring evidence.

CPU only, no optimizer step.  Everything is measured on the delivered Stage2
checkpoint; nothing is trained.
"""
import hashlib, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mixed_geometry as G
import probe_measurement as M
import structure as S
import train_rqvae as T
import training_hooks as H
from model.model import RQVAE

SEED = T.SEED
BATCH, TRIPLES = 64, 64
CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.manual_seed(SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, SEED)
items = torch.from_numpy(emb).float()
model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
state = torch.load(CKPT, map_location="cpu")
state = state.get("state_dict", state) if isinstance(state, dict) else state
missing, unexpected = model.load_state_dict(state, strict=False)

frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:BATCH]
batch_ids = [int(i) for i in ids]
batch = items[torch.tensor(batch_ids)]
graph = S.build_graph(T.TRAIN_FILE, seed=SEED)

triples = S.sample_triples(graph, TRIPLES, random.Random(SEED))
item_ids = [item for c, a, b, _ in triples for item in (c, a, b)]

out = {
    "execution": {
        "script": str(Path(__file__)),
        "command": "python3.9 precheck_question2.py",
        "checkpoint": str(CKPT),
        "checkpoint_sha256": sha(CKPT),
        "sid_sha256": "7c147fc634cc2939367d3fc83bc0a8ebfaa2d96a92cf23e060c7c88b4ac390a1",
        "source_sha256": {name: sha(ROOT / name) for name in
                          ("model/layers.py", "model/model.py", "mixed_geometry.py",
                           "native_product.py", "structure.py", "training_hooks.py")},
        "seed": SEED, "batch_ids": batch_ids, "batch_size": BATCH,
        "triples_requested": TRIPLES, "triples_accepted": len(triples),
        "acceptance_rate": len(triples) / TRIPLES,
        "behaviour_on": bool(T.BEHAVIOUR_ON), "geometry_on": bool(G.GEOMETRY_ON),
        "load_state_dict": {"missing": list(missing), "unexpected": list(unexpected)},
    },
}

# A. branch modes and corrected occupancy
train_tokens = M.branch_tokens(model, batch, "train")
export_tokens = M.branch_tokens(model, batch, "export")
out["branch"] = {
    "train_mode": "train", "export_mode": "eval + infer_use_sk=False",
    "train_tokens": train_tokens.tolist(),
    "export_tokens": export_tokens.tolist(),
    "differing_positions": [int((train_tokens[:, l] != export_tokens[:, l]).sum())
                            for l in range(3)],
    "occupancy_train": M.occupancy(train_tokens),
    "occupancy_export": M.occupancy(export_tokens),
}

# edges whose two endpoints are in this batch
position = {gid: row for row, gid in enumerate(batch_ids)}
kept = [(position[a], position[b]) for a, b in
        ((n, m) for n, ns in graph.neighbours.items() for m in ns if n < m)
        if a in position and b in position]
edge_index = torch.tensor(kept, dtype=torch.long).T if kept else torch.zeros(2, 0, dtype=torch.long)
out["edges"] = M.edge_stats(train_tokens, edge_index)

# B. fixed-state gradients, no optimizer step
scale = 0.041920
lam = 3.5211e-06
recon, quant, _, _ = model.forward(batch)
base_loss = model.compute_loss(batch, recon, quant)[0]
term = H.structure_term(model, items, graph, random.Random(SEED), scale=scale, size=TRIPLES)


def block_named_grads(loss):
    blocks = {"encoder": list(model.encoder.parameters()),
              "codebooks": [p for layer in model.rq.vq_layers for p in layer.parameters()],
              "decoder": list(model.decoder.parameters())}
    out = {}
    for name, params in blocks.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, grads)])
        out[name] = {"norm": float(flat.norm()), "finite": bool(torch.isfinite(flat).all())}
    return out


g_base = block_named_grads(base_loss)
g_struct = block_named_grads(lam * term)
g_both = block_named_grads(base_loss + lam * term)
angles = {}
for name in g_base:
    gb = torch.autograd.grad(base_loss, [p for p in model.parameters() if p.requires_grad],
                             retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(term, [p for p in model.parameters() if p.requires_grad],
                             retain_graph=True, allow_unused=True)
    vb = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip([p for p in model.parameters() if p.requires_grad], gb)])
    vs = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip([p for p in model.parameters() if p.requires_grad], gs)])
    cos = float(torch.dot(vb, vs) / (vb.norm() * vs.norm())) if vb.norm() > 0 and vs.norm() > 0 else None
    angles[name] = {"cosine_all_parameters": cos}
    break
out["fixed_state"] = {
    "scale": scale, "lambda": lam,
    "base_grads": g_base, "structure_grads": g_struct, "combined_grads": g_both,
    "cosine": angles,
    "base_loss": float(base_loss.detach()), "structure_term": float(term.detach()),
}

# switch checks: forward/gradient only, no optimizer step
switches = {}
G.GEOMETRY_ON = False
try:
    off_tokens = M.branch_tokens(model, batch, "export")
    off_recon, off_quant, _, _ = model.forward(batch)
    off_term = H.structure_term(model, items, graph, random.Random(SEED), scale=scale, size=TRIPLES)
    gs_off = torch.autograd.grad(off_term, [p for p in model.rq.vq_layers[0].parameters()],
                                 retain_graph=True, allow_unused=True)
    switches["geometry_off"] = {
        "tokens_equal_to_on": bool(torch.equal(off_tokens, export_tokens)),
        "changed_positions": [int((off_tokens[:, l] != export_tokens[:, l]).sum()) for l in range(3)],
        "structure_term": float(off_term.detach()),
        "codebook1_grad_norm": float(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                                for p, g in zip(model.rq.vq_layers[0].parameters(), gs_off)]).norm()),
    }
finally:
    G.GEOMETRY_ON = True
term_off = H.structure_term(model, items, graph, random.Random(SEED), scale=scale, size=TRIPLES,
                            behaviour_on=False)
switches["behaviour_off"] = {"structure_term": float(term_off.detach()),
                             "requires_grad": bool(term_off.requires_grad)}
out["switches"] = switches

# C. alignment with the formal run
out["alignment"] = {
    "precheck": {"s": scale, "lambda": lam, "batch": BATCH, "triples": TRIPLES},
    "formal_run": {"s": 0.04444597661495209, "lambda": 9.226019983543665e-07,
                   "batch": 4096, "triples": T.STRUCT_TRIPLES},
    "lambda_ratio": lam / 9.226019983543665e-07,
    "note": "initialisation, sample, batch and DDP state all differ; attribution UNKNOWN",
}
print(json.dumps(out, indent=1)[:200])
Path("/tmp/q2_out.json").write_text(json.dumps(out, indent=1))
print("written /tmp/q2_out.json")
