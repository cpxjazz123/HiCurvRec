"""Corrected factor attribution on the real TRAIN batch (fix tree)."""
import hashlib, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mixed_geometry as G, structure as S, train_rqvae as T, training_hooks as H
import probe_factors as F
from model.model import RQVAE

SEED = T.SEED; BATCH, TRIPLES = 64, 64
S_SCALE, LAMBDA = 0.04444597661495209, 9.226019983543665e-07
CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
OUT = Path("/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q6"); OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, SEED)
items = torch.from_numpy(emb).float()
model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
_state = torch.load(CKPT, map_location="cpu")
_state = _state.get("state_dict", _state) if isinstance(_state, dict) else _state
_missing, _unexpected = model.load_state_dict(_state, strict=False)
if _missing or _unexpected:
    raise SystemExit(f"checkpoint mismatch: missing={_missing[:3]} unexpected={_unexpected[:3]}")
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:BATCH]
batch = items[torch.from_numpy(ids)]
graph = S.build_graph(T.TRAIN_FILE, seed=SEED)
triples = S.sample_triples(graph, TRIPLES, random.Random(SEED))

blocks = {"encoder": list(model.encoder.parameters())}
for i, layer in enumerate(model.rq.vq_layers):
    blocks[f"C{i+1}"] = [layer.get_code_embs()]
blocks["decoder"] = list(model.decoder.parameters())

recon, quant, _, _ = model.forward(batch)
recon_loss = torch.nn.functional.mse_loss(recon, batch)
base = model.compute_loss(batch, recon, quant)[0]
term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)
struct_loss = LAMBDA * term
total = base + struct_loss

# latent-space VJP for the encoder: differentiate each loss w.r.t. the encoder output
encoded = model.encoder(batch)
encoded.retain_grad()
model.zero_grad(set_to_none=True)


def whole_grads(loss):
    flat = [p for ps in blocks.values() for p in ps]
    out = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    res, cur = {}, 0
    for name, ps in blocks.items():
        res[name] = out[cur:cur + len(ps)]
        cur += len(ps)
    return res


def latent_grad(kind):
    """Latent-space E/H gradient, captured from the encoder output in the same graph."""
    if kind != "structure":
        return "NOT_MEASURED"
    captured = []

    def hook(module, inputs, output):
        output.retain_grad()
        captured.append(output)

    handle = model.encoder.register_forward_hook(hook)
    try:
        recon, quant, _, _ = model.forward(batch)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        base = model.compute_loss(batch, recon, quant)[0]
        term = H.structure_term(model, items, graph, random.Random(SEED), scale=S_SCALE, size=TRIPLES)
        losses = {"recon": recon_loss, "quant": quant, "structure": LAMBDA * term}
        model.zero_grad(set_to_none=True)
        losses[kind].backward(retain_graph=False)
        rows = batch.shape[0] if kind in ("recon", "quant") else 3 * TRIPLES
        chosen = [z for z in captured if z.shape[0] == rows]
        if not chosen:
            return "NOT_CONNECTED"
        z = chosen[0]
        return z.grad.detach().clone() if z.grad is not None else None
    finally:
        handle.remove()


report = {"identity": {
    "tree": str(ROOT), "arm_name": T.ARM_NAME,
    "checkpoint_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
    "sid_sha256": hashlib.sha256(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json").read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "layers_sha256": hashlib.sha256(Path(ROOT/"model/layers.py").read_bytes()).hexdigest(),
    "s": S_SCALE, "lambda": LAMBDA, "note": "old checkpoint, new wiring; not the fix version's calibration",
    "seed": SEED, "batch_ids": [int(i) for i in ids], "triples_accepted": len(triples),
    "mode": "train", "geometry_on": bool(G.GEOMETRY_ON), "behaviour_on": bool(T.BEHAVIOUR_ON)},
    "losses": {"recon": float(recon_loss.detach()), "quant": float(quant.detach()),
               "structure_weighted": float(struct_loss.detach()), "total": float(total.detach())},
    "gradients": {}, "latent": {}}

for name, loss in (("recon", recon_loss), ("quant", quant), ("structure", struct_loss), ("total", total)):
    g = whole_grads(loss)
    named = {}
    for block, ps in blocks.items():
        kind = "code" if block.startswith("C") else "parameter"
        if kind == "code":
            named[block] = (g[block][0], "code")
        else:
            named[block] = (torch.cat([(torch.zeros_like(p) if gg is None else gg).reshape(-1)
                                       for p, gg in zip(ps, g[block])]), "parameter")
    report["gradients"][name] = F.block_report(named)
    lg = latent_grad(name)
    if isinstance(lg, torch.Tensor):
        report["latent"][name] = F.factor_norms("latent", lg, "latent")
    else:
        report["latent"][name] = {"split": str(lg)}

# composition check on the same graph: base + lambda*struct == total
g_base, g_struct = whole_grads(base), whole_grads(struct_loss)
g_total = whole_grads(total)
worst = 0.0
per_tensor = {}
for block in blocks:
    for idx, (p, a, b, t) in enumerate(zip(blocks[block], g_base[block],
                                           g_struct[block], g_total[block])):
        av = torch.zeros_like(p) if a is None else a
        bv = torch.zeros_like(p) if b is None else b
        tv = torch.zeros_like(p) if t is None else t
        dev = float((av + bv - tv).abs().max())
        worst = max(worst, dev)
        per_tensor[f"{block}[{idx}]"] = dev
report["composition"] = {"checked": "base + structure per tensor", "worst_abs_dev": worst,
                         "per_tensor": per_tensor}

Path(OUT/"factors.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1)[:1600])
print("written", OUT/"factors.json")
