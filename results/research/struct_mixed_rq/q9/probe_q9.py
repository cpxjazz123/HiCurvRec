"""q9: one sampling path, full report, no writes to history roots on import."""
from __future__ import annotations

import json
from pathlib import Path

import torch

E = 16


def split_latent(g):
    return {"E_norm": float(g[..., :E].pow(2).sum()) ** 0.5,
            "H_norm": float(g[..., E:].pow(2).sum()) ** 0.5}


def report_blocks(blocks, grads):
    out = {}
    for name, params in blocks.items():
        gs = grads.get(name, [None] * len(params))
        none = sum(1 for g in gs if g is None)
        zero = sum(1 for g in gs if g is not None and float(g.abs().sum()) == 0.0)
        nonzero = sum(1 for g in gs if g is not None and float(g.abs().sum()) > 0.0)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, gs)])
        out[name] = {"tensors": len(params), "none": none, "zero": zero,
                     "nonzero": nonzero, "norm": float(flat.norm())}
    return out


def run(out_root: Path) -> dict:
    import hashlib
    import random
    import numpy as np
    import pandas as pd
    import mixed_geometry as G
    import structure as S
    import train_rqvae as T
    import training_hooks as H
    from model.model import RQVAE

    seed = T.SEED
    requested, batch_n = 64, 64
    scale, lam = 0.04444597661495209, 9.226019983543665e-07
    ckpt = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
    torch.manual_seed(seed)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, seed)
    items = torch.from_numpy(emb).float()
    model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: {missing[:3]} {unexpected[:3]}")
    frame = pd.read_parquet(T.TRAIN_FILE)
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:batch_n]
    batch = items[torch.from_numpy(ids)]
    graph = S.build_graph(T.TRAIN_FILE, seed=seed)

    # ONE sampling: the returned list is what the structure loss uses
    used = S.sample_triples(graph, requested, random.Random(seed))
    kinds = [k for *_, k in used]
    item_ids = [it for c, a, b, _ in used for it in (c, a, b)]

    captures = {"base": [], "struct": []}
    current = ["base"]

    def hook(module, inputs, output):
        output.retain_grad()
        captures[current[0]].append(output)

    handle = model.encoder.register_forward_hook(hook)
    try:
        current[0] = "base"
        recon, quant, _, _ = model.forward(batch)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        base = model.compute_loss(batch, recon, quant)[0]
        current[0] = "struct"
        positions = H._positions(model, items, item_ids).reshape(len(used), 3, -1)
        h = S.triple_targets(kinds)
        term = S.structure_loss(positions, h, scale=scale)
        struct = lam * term
        total = base + struct
        z_base = captures["base"][0]
        z_struct = captures["struct"][0]

        blocks = {"encoder": list(model.encoder.parameters())}
        for i, layer in enumerate(model.rq.vq_layers):
            blocks[f"C{i+1}"] = [layer.get_code_embs()]
        blocks["decoder"] = list(model.decoder.parameters())
        flat = [p for ps in blocks.values() for p in ps]
        states, latent = {}, {}
        for name, loss in (("recon", recon_loss), ("quant", quant),
                           ("structure", struct), ("total", total)):
            raw = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
            per, cur = {}, 0
            for block, ps in blocks.items():
                per[block] = raw[cur:cur + len(ps)]
                cur += len(ps)
            states[name] = report_blocks(blocks, per)
            gb, gs = torch.autograd.grad(loss, [z_base, z_struct], retain_graph=True,
                                         allow_unused=True)
            latent[name] = {
                "z_base": None if gb is None else split_latent(gb.detach()),
                "z_struct": None if gs is None else split_latent(gs.detach()),
            }
        tb, bb = torch.autograd.grad(total, z_base, retain_graph=True)[0], \
            torch.autograd.grad(base, z_base, retain_graph=True)[0]
        ts, ss = torch.autograd.grad(total, z_struct, retain_graph=True)[0], \
            torch.autograd.grad(struct, z_struct, retain_graph=True)[0]
        report = {
            "identity": {
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                "sid_sha256": hashlib.sha256(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json").read_bytes()).hexdigest(),
                "layers_sha256": hashlib.sha256(Path(__file__).with_name("model").joinpath("layers.py").read_bytes()).hexdigest(),
                "s": scale, "lambda": lam, "seed": seed,
                "batch_ids": [int(i) for i in ids],
                "triples_requested": requested, "triples_accepted": len(used),
                "acceptance_rate": len(used) / requested,
                "z_base_rows": z_base.shape[0], "z_struct_rows": z_struct.shape[0],
                "rows_check": "3*accepted == z_struct_rows",
                "mode": "train", "behaviour_on": bool(T.BEHAVIOUR_ON),
                "geometry_on": bool(G.GEOMETRY_ON),
            },
            "losses": {"recon": float(recon_loss.detach()), "quant": float(quant.detach()),
                       "structure_weighted": float(struct.detach()),
                       "total": float(total.detach())},
            "states": states, "latent": latent,
            "diffs": {"base_vs_total_z_base": float((tb - bb).abs().max()),
                      "struct_vs_total_z_struct": float((ts - ss).abs().max())},
        }
    finally:
        handle.remove()

    out_dir = out_root / "run-q9"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factors.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q9")
    print(json.dumps(run(root), indent=1)[:600])
