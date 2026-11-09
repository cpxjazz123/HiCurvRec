"""Real TRAIN batch on the fixed tree: E/H split gradients per block."""
import hashlib, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mixed_geometry as G, structure as S, train_rqvae as T, training_hooks as H
from model.model import RQVAE

SEED = T.SEED; BATCH, TRIPLES = 64, 64
S_SCALE, LAMBDA = 0.04444597661495209, 9.226019983543665e-07
CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
OUT = Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q5"); OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, SEED)
items = torch.from_numpy(emb).float()
model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state.get("state_dict", state), strict=False)
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:BATCH]
batch = items[torch.from_numpy(ids)]
graph = S.build_graph(T.TRAIN_FILE, seed=SEED)
triples = S.sample_triples(graph, TRIPLES, random.Random(SEED))

blocks = {"encoder": list(model.encoder.parameters())}
for i, layer in enumerate(model.rq.vq_layers):
    blocks[f"C{i+1}"] = list(layer.parameters())
blocks["decoder"] = list(model.decoder.parameters())

recon, quant, _, _ = model.forward(batch)
recon_loss = torch.nn.functional.mse_loss(recon, batch)
base = model.compute_loss(batch, recon, quant)[0]
term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)
total = base + LAMBDA * term


def grads(loss):
    flat = [p for ps in blocks.values() for p in ps]
    out = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    res, cur = {}, 0
    for name, ps in blocks.items():
        res[name] = out[cur:cur + len(ps)]
        cur += len(ps)
    return res


def split(ps, gs):
    e = h = 0.0; none = 0
    for p, g in zip(ps, gs):
        if g is None:
            none += 1; continue
        # parameters shaped (out, in): the 32 latent coordinates are the *input*
        if g.dim() == 2:
            e += float(g[:, :16].pow(2).sum()); h += float(g[:, 16:].pow(2).sum())
        else:
            e += float(g.pow(2).sum())   # bias: not split by latent coordinate
    return {"E_norm": e ** 0.5, "H_norm": h ** 0.5, "none": none}


report = {"identity": {
    "tree": str(ROOT), "checkpoint_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
    "sid_sha256": hashlib.sha256(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json").read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "layers_sha256": hashlib.sha256(Path(ROOT/"model/layers.py").read_bytes()).hexdigest(),
    "s": S_SCALE, "lambda": LAMBDA, "seed": SEED, "batch_ids": [int(i) for i in ids],
    "triples_accepted": len(triples), "mode": "train", "geometry_on": bool(G.GEOMETRY_ON),
    "behaviour_on": bool(T.BEHAVIOUR_ON)},
    "losses": {"recon": float(recon_loss.detach()), "quant": float(quant.detach()),
               "structure_weighted": float((LAMBDA*term).detach()), "total": float(total.detach())},
    "E_H_split": {}}
for name, loss in (("recon", recon_loss), ("quant", quant), ("structure", LAMBDA*term), ("total", total)):
    if loss is None: continue
    g = grads(loss)
    report["E_H_split"][name] = {b: split(blocks[b], g[b]) for b in blocks}

# one cheap both-on step, then report the update
codebooks = [l.get_code_embs() for l in model.rq.vq_layers]
before = [p.detach().clone() for p in codebooks]
opt = torch.optim.AdamW(model.parameters(), lr=float(T.LR), weight_decay=float(T.WEIGHT_DECAY))
total.backward(); opt.step()
report["update"] = {"codebook_delta": [float((p.detach()-b).norm()) for p,b in zip(codebooks, before)],
                    "clip_norm_configured": float(T.GRADIENT_CLIP_NORM)}
Path(OUT/"eh_split.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1)[:1200])
print("written", OUT/"eh_split.json")
