"""question3: per-block, per-layer fixed-state wiring at the delivered checkpoint.

CPU, no optimizer step, no recalibration.  s and lambda are the formal run's own
calibration values, recorded in the run's training log.
"""
import hashlib, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mixed_geometry as G
import probe_blocks as B
import structure as S
import train_rqvae as T
import training_hooks as H
from model.model import RQVAE

SEED = T.SEED
BATCH, TRIPLES = 64, 64
S_SCALE, LAMBDA = 0.04444597661495209, 9.226019983543665e-07
CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
OUT = Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q3")
OUT.mkdir(parents=True, exist_ok=True)


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
batch_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:BATCH]
batch = items[torch.from_numpy(batch_ids)]
graph = S.build_graph(T.TRAIN_FILE, seed=SEED)
triples = S.sample_triples(graph, TRIPLES, random.Random(SEED))

blocks = {"encoder": list(model.encoder.parameters())}
for level, layer in enumerate(model.rq.vq_layers):
    blocks[f"C{level + 1}"] = list(layer.parameters())
blocks["decoder"] = list(model.decoder.parameters())


def grads_of(loss):
    flat = [p for params in blocks.values() for p in params]
    out = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    per, cursor = {}, 0
    for name, params in blocks.items():
        chunk = []
        for _ in params:
            chunk.append(out[cursor])
            cursor += 1
        per[name] = chunk
    return per


def tokens(mode):
    was = model.training
    model.train(mode == "train")
    try:
        with torch.no_grad():
            return model.get_indices(batch, infer_use_sk=False).clone()
    finally:
        model.train(was)


recon, quant, _, _ = model.forward(batch)
recon_loss = torch.nn.functional.mse_loss(recon, batch)
quant_loss = quant
term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)
struct_loss = LAMBDA * term
total = recon_loss + quant_loss + struct_loss

train_tokens, export_tokens = tokens("train"), tokens("export")
reference = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                       for g in grads_of(total)["encoder"]]) if False else None

report = {
    "identity": {
        "checkpoint": str(CKPT), "checkpoint_sha256": sha(CKPT),
        "sid": str(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json")),
        "sid_sha256": sha("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json"),
        "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)),
        "sources": {p: sha(ROOT / p) for p in ("model/layers.py", "model/model.py",
                                               "mixed_geometry.py", "native_product.py",
                                               "structure.py", "training_hooks.py",
                                               "probe_blocks.py")},
        "inputs": {"train": sha(T.TRAIN_FILE), "embedding": sha(T.EMBEDDING_FILE)},
        "seed": SEED, "batch_ids": [int(i) for i in batch_ids], "batch": BATCH,
        "triples_requested": TRIPLES, "triples_accepted": len(triples),
        "acceptance_rate": len(triples) / TRIPLES,
        "s": S_SCALE, "lambda": LAMBDA,
        "sources_of_s_lambda": "formal run calibration (training log)",
        "switches": {"behaviour_on": bool(T.BEHAVIOUR_ON), "geometry_on": bool(G.GEOMETRY_ON)},
        "load_state_dict": {"missing": list(missing), "unexpected": list(unexpected)},
        "run_id": "UNAVAILABLE",
    },
    "branch": {"train_vs_export": [int((train_tokens[:, l] != export_tokens[:, l]).sum())
                                   for l in range(3)]},
    "losses": {"recon": float(recon_loss.detach()), "quant": float(quant_loss.detach()),
               "structure_weighted": float(struct_loss.detach()),
               "structure_raw": float(term.detach()), "total": float(total.detach())},
    "gradients": {},
}

for name, loss in (("recon", recon_loss), ("quant", quant_loss),
                   ("structure", struct_loss), ("total", total)):
    g = grads_of(loss)
    report["gradients"][name] = {
        "blocks": B.describe_blocks(blocks, g),
        "cosines_vs_total": None,
    }

total_grads = grads_of(total)
flat_total = torch.cat([(torch.zeros_like(p) if gg is None else gg).reshape(-1)
                        for params in blocks.values() for p, gg in zip(params, total_grads[list(blocks)[0]])]) if False else None
# reference vector: the whole-model total gradient, in block order
ref = torch.cat([(torch.zeros_like(p) if gg is None else gg).reshape(-1)
                 for name in blocks
                 for p, gg in zip(blocks[name], total_grads[name])])
for name, loss in (("recon", recon_loss), ("quant", quant_loss),
                   ("structure", struct_loss)):
    g = grads_of(loss)
    per_block = {}
    for block in blocks:
        ref_block = torch.cat([(torch.zeros_like(p) if gg is None else gg).reshape(-1)
                               for p, gg in zip(blocks[block], total_grads[block])])
        per_block[block] = B.block_cosines(ref_block, {block: blocks[block]},
                                           {block: g[block]})[block]
    report["gradients"][name]["block_cosines_vs_total"] = per_block

# geometry on/off at one assignment mode, and readback differences
off = {}
G.GEOMETRY_ON = False
try:
    off_tokens_export = tokens("export")
    off_recon, off_quant, _, _, off_levels = model.forward_with_codes(batch)
    off_term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)
finally:
    G.GEOMETRY_ON = True
with torch.no_grad():
    _, _, _, _, on_levels = model.forward_with_codes(batch)
off["token_diff"] = [int((off_tokens_export[:, l] != export_tokens[:, l]).sum()) for l in range(3)]
off["readback_max_abs_diff"] = [float((a - b).abs().max()) for a, b in zip(off_levels, on_levels)]
off["structure_term_off"] = float(off_term.detach())
report["geometry_on_off_same_assignment_mode"] = off

(OUT / "diagnostics.json").write_text(json.dumps(report, indent=1))
(OUT / "precheck_question3.py").write_text(Path(__file__).read_text())
print(json.dumps(report, indent=1)[:300])
print("written", OUT / "diagnostics.json")
