"""structure-vs-base block cosines, decoder precision, and mode labelling."""
import hashlib, json, random, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mixed_geometry as G, structure as S, train_rqvae as T, training_hooks as H
from model.model import RQVAE

SEED = T.SEED
BATCH, TRIPLES = 64, 64
S_SCALE, LAMBDA = 0.04444597661495209, 9.226019983543665e-07
CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")

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

blocks = {"encoder": list(model.encoder.parameters())}
for i, layer in enumerate(model.rq.vq_layers):
    blocks[f"C{i+1}"] = list(layer.parameters())
blocks["decoder"] = list(model.decoder.parameters())

recon, quant, _, _ = model.forward(batch)
base = model.compute_loss(batch, recon, quant)[0]
term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)


def grads(loss):
    flat = [p for ps in blocks.values() for p in ps]
    out = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    res, cur = {}, 0
    for name, ps in blocks.items():
        res[name] = out[cur:cur + len(ps)]
        cur += len(ps)
    return res


gb, gs = grads(base), grads(term)
report = {"identity": {"checkpoint_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
                       "sid_sha256": hashlib.sha256(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json").read_bytes()).hexdigest(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "s": S_SCALE, "lambda": LAMBDA, "seed": SEED,
                       "batch_ids": [int(i) for i in ids],
                       "triples_accepted": len(S.sample_triples(graph, TRIPLES, random.Random(SEED)))},
          "structure_vs_base": {}, "decoder_precision": {}}
for name in blocks:
    a = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(blocks[name], gb[name])])
    b = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(blocks[name], gs[name])])
    denom = (a.norm() * b.norm())
    cos = float(torch.dot(a, b) / denom) if float(denom) > 0 else "UNDEFINED"
    report["structure_vs_base"][name] = {"base_norm": float(a.norm()), "structure_norm": float(b.norm()),
                                         "cosine": cos}
    if name == "decoder":
        report["decoder_precision"] = {"cosine_float32_path": cos}

print(json.dumps(report, indent=1)[:1500])
Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q4").mkdir(parents=True, exist_ok=True)
Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q4/extras.json").write_text(json.dumps(report, indent=1))
print("written extras.json")
